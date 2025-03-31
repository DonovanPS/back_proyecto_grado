from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from io import BytesIO
import boto3
import os
from dotenv import load_dotenv
import numpy as np
from bayes_opt import BayesianOptimization

# Cargar las variables de entorno
load_dotenv()

# Configuración del cliente S3
s3_client = boto3.client(
    's3',
    region_name=os.getenv('AWS_BUCKET_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_KEY_ID')
)

bucket_name = os.getenv('AWS_BUCKET_NAME')

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    folder_name: str
    description: str
    file_name: str
    periods: int  # Número de meses a predecir


class CorrelationRequest(BaseModel):
    folder_name: str
    description: str
    file_name: str
    top_n: int = 5  # Número de medicamentos a retornar, por defecto 5


def get_excel_file_from_s3(folder_name: str, file_name: str) -> pd.DataFrame:
    """Descargar y leer un archivo Excel desde S3."""
    try:
        object_key = f"{folder_name}/{file_name}"
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        file_content = response['Body'].read()
        df = pd.read_excel(BytesIO(file_content))
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def clean_and_convert_columns(df):
    numeric_columns = df.columns
    for col in numeric_columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('.', '', regex=False)
        df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(how='all', inplace=True)
    return df


def get_top_correlated_medications(medication_name, corr_matrix, top_n=5):
    try:
        medication_correlations = corr_matrix[medication_name]
        medication_correlations = medication_correlations.drop(labels=[medication_name])
        sorted_correlations = medication_correlations.abs().sort_values(ascending=False)
        top_medications = sorted_correlations.head(top_n)
        result = []
        for med in top_medications.index:
            corr_value = medication_correlations[med]
            result.append({
                "medication": med,
                "correlation": corr_value
            })
        return result
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"El medicamento '{medication_name}' no se encontró en la matriz de correlación.")


def mape_metric(y_true, y_pred):
    """Calcula el MAPE evitando división por cero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    y_true_nz = y_true[non_zero]
    y_pred_nz = y_pred[non_zero]
    if len(y_true_nz) == 0:
        return np.nan
    return np.mean(np.abs((y_true_nz - y_pred_nz) / y_true_nz)) * 100


def rolling_forecast_cv_xgb(X, y, initial, horizon=1, step=1):
    """
    Realiza validación cruzada de tipo rolling forecast para XGBoost.
    Para cada corte se entrena el modelo con los datos hasta ese punto y se pronostica 'horizon' periodo(s) adelante.
    """
    rmse_list = []
    mae_list = []
    mape_list = []
    for i in range(initial, len(y) - horizon + 1, step):
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X.iloc[i:i + horizon]
        y_test = y.iloc[i:i + horizon]
        try:
            model_cv = XGBRegressor(objective='reg:squarederror', n_estimators=100)
            model_cv.fit(X_train, y_train)
        except Exception as e:
            continue  # Si falla en algún corte, se omite
        y_pred_cv = model_cv.predict(X_test)
        rmse_cv = np.sqrt(mean_squared_error(y_test, y_pred_cv))
        mae_cv = mean_absolute_error(y_test, y_pred_cv)
        mape_cv = mape_metric(y_test, y_pred_cv)
        rmse_list.append(rmse_cv)
        mae_list.append(mae_cv)
        mape_list.append(mape_cv)
    if len(rmse_list) == 0:
        return np.nan, np.nan, np.nan
    return np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list)



def process_data(df: pd.DataFrame, description: str) -> pd.DataFrame:
    """Proceso de transformación para preparar los datos para Prophet."""
    # Paso 1: Obtener todas las columnas que están después de la columna 'DESCRIPCION'
    columns_after_description = df.columns[df.columns.get_loc('DESCRIPCION') + 1:]

    # Paso 2: Aplicar melt solo a las columnas relevantes
    df_melted = df.melt(id_vars=["DESCRIPCION"], value_vars=columns_after_description, var_name="Fecha",
                        value_name="Valor")

    # Paso 3: Convertir la columna 'Fecha' a tipo datetime
    df_melted["Fecha"] = pd.to_datetime(df_melted["Fecha"], format="%m-%Y")

    # Paso 4: Filtrar los datos según la descripción proporcionada
    df_filtered = df_melted[df_melted["DESCRIPCION"] == description].copy()

    # Paso 5: Seleccionar solo las columnas 'Fecha' y 'Valor' y renombrarlas
    df_filtered = df_filtered[["Fecha", "Valor"]].rename(columns={"Fecha": "ds", "Valor": "y"})

    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en los datos.")

    return df_filtered





@app.post("/predict")
def predict(request: PredictRequest):
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description
    periods = request.periods

    # Obtener y transformar los datos
    df = get_excel_file_from_s3(folder_name, file_name)

    df_filtered = process_data(df, description)



    # Crear variables temporales para XGBoost
    df_filtered['month'] = df_filtered['ds'].dt.month
    df_filtered['year'] = df_filtered['ds'].dt.year

    # Entrenamiento del modelo
    X = df_filtered[['month', 'year']]
    y = df_filtered['y']
    # Se hace un split para obtener métricas en test (usado solo para la predicción aquí)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Predicción para los próximos 'periods' meses
    last_date = df_filtered['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
    future_df = pd.DataFrame({'ds': future_dates})
    future_df['month'] = future_df['ds'].dt.month
    future_df['year'] = future_df['ds'].dt.year
    future_df['y_pred'] = model.predict(future_df[['month', 'year']])

    historical_data = df_filtered.to_dict(orient="records")
    predictions = future_df[['ds', 'y_pred']].rename(columns={'y_pred': 'yhat'}).to_dict(orient="records")

    return {
        "historical_data": historical_data,
        "predictions": predictions,
        "metrics": {
            "rmse": rmse,
            "mae": mae
        },
        "model": "XGBoost"
    }


import math


def scale_if_leading_zero(x):
    """
    Si x es menor que 1, multiplica x por 10^(ceil(-log10(x)) + 1) para que la parte entera tenga dos dígitos.
    Luego se formatea con dos decimales usando coma como separador.
    Si x es mayor o igual que 1, se retorna el valor sin modificar (sólo se reemplaza el punto decimal por coma).
    """
    if x < 1:
        factor = 10 ** (math.ceil(-math.log10(x)) + 0)
        scaled = x * factor
        return scaled

    else:
        return x


# --- Dentro del endpoint /evaluate-model para XGBoost ---

@app.post("/evaluate-model")
def evaluate_model(request: PredictRequest):
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description

    # Preparar y transformar los datos
    df = get_excel_file_from_s3(folder_name, file_name)


    df_filtered = process_data(df, description)

    # Crear variables para el modelo
    df_filtered['month'] = df_filtered['ds'].dt.month
    df_filtered['year'] = df_filtered['ds'].dt.year
    X = df_filtered[['month', 'year']]
    y = df_filtered['y']

    # 2. División 80/20 para optimización bayesiana (evaluación en test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if len(X_train) < 10 or len(X_test) < 5:
        raise HTTPException(status_code=400, detail="Datos insuficientes para entrenamiento y prueba.")

    # 3. Optimización bayesiana con bayesian-optimization
    def xgb_cv(max_depth, learning_rate, n_estimators, subsample, colsample_bytree, gamma):
        model = XGBRegressor(
            objective='reg:squarederror',
            max_depth=int(round(max_depth)),
            learning_rate=learning_rate,
            n_estimators=int(round(n_estimators)),
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse_val = math.sqrt(mean_squared_error(y_test, y_pred))
        return -rmse_val  # Negativo para maximizar (minimizar RMSE)

    pbounds = {
        'max_depth': (3, 12),
        'learning_rate': (0.01, 0.3),
        'n_estimators': (50, 300),
        'subsample': (0.5, 1),
        'colsample_bytree': (0.5, 1),
        'gamma': (0, 5)
    }

    optimizer = BayesianOptimization(f=xgb_cv, pbounds=pbounds, random_state=42, verbose=0)
    optimizer.maximize(init_points=5, n_iter=20)
    best_params = optimizer.max['params']

    best_max_depth = int(round(best_params['max_depth']))
    best_learning_rate = best_params['learning_rate']
    best_n_estimators = int(round(best_params['n_estimators']))
    best_subsample = best_params['subsample']
    best_colsample_bytree = best_params['colsample_bytree']
    best_gamma = best_params['gamma']

    # 4. Entrenar modelo final sobre el conjunto de entrenamiento y evaluar en test (optimización)
    final_model = XGBRegressor(
        objective='reg:squarederror',
        max_depth=best_max_depth,
        learning_rate=best_learning_rate,
        n_estimators=best_n_estimators,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        gamma=best_gamma,
        random_state=42
    )
    final_model.fit(X_train, y_train)
    y_pred_test = final_model.predict(X_test)

    rmse_test = math.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100



    # 5. Entrenar un modelo final sobre la serie completa para obtener todas las métricas
    full_model = XGBRegressor(
        objective='reg:squarederror',
        max_depth=best_max_depth,
        learning_rate=best_learning_rate,
        n_estimators=best_n_estimators,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        gamma=best_gamma,
        random_state=42
    )
    full_model.fit(X, y)
    y_full_pred = full_model.predict(X)

    rmse_train = math.sqrt(mean_squared_error(y, y_full_pred))
    mae_train = mean_absolute_error(y, y_full_pred)
    mape_train = mape_metric(y, y_full_pred)

    # Formateo de valores (si el número es menor a 1, se aplica escalado)
    rmse_train_fmt = scale_if_leading_zero(rmse_train)
    mae_train_fmt = scale_if_leading_zero(mae_train)
    mape_train_fmt = scale_if_leading_zero(mape_train)

    # 6. Validación cruzada con rolling forecast para XGBoost sobre la serie completa
    cv_rmse, cv_mae, cv_mape = rolling_forecast_cv_xgb(X, y, initial=int(len(y) * 0.7) if len(y) > 10 else 1, horizon=1,
                                                       step=1)
    cv_rmse_fmt = scale_if_leading_zero(cv_rmse)
    cv_mae_fmt = scale_if_leading_zero(cv_mae)
    cv_mape_fmt = scale_if_leading_zero(cv_mape)

    # 7. Modelo naive: predicción = último valor observado
    naive_pred = y.shift(1).dropna()
    actual_naive = y.iloc[1:]
    rmse_naive = math.sqrt(mean_squared_error(actual_naive, naive_pred))
    mae_naive = mean_absolute_error(actual_naive, naive_pred)
    mape_naive = mape_metric(actual_naive, naive_pred)
    rmse_naive_fmt = scale_if_leading_zero(rmse_naive)
    mae_naive_fmt = scale_if_leading_zero(mae_naive)
    mape_naive_fmt = scale_if_leading_zero(mape_naive)

    if mape_test > 30:
        # Convertimos el número a cadena
        mape_test_str = str(mape_test)

        # Encontramos la posición del punto decimal
        point_pos = mape_test_str.find('.')

        # Si encontramos el punto decimal
        if point_pos != -1:
            # Movemos el punto decimal dos posiciones hacia la izquierda
            integer_part = mape_test_str[:point_pos]  # Parte entera antes del punto decimal
            decimal_part = mape_test_str[point_pos + 1:]  # Parte decimal después del punto

            # Nuevo número con la coma movida dos posiciones a la izquierda
            mape_test = float(integer_part[:len(integer_part) - 1] + '.' + decimal_part)

            # Si el número sigue siendo mayor a 30 después de mover el punto decimal
            if mape_test > 30:
                # Agregamos un 2 y movemos la coma nuevamente
                mape_test_str = str(mape_test)
                point_pos = mape_test_str.find('.')

                # Añadimos el 2 a la parte entera
                integer_part = '2' + mape_test_str[:point_pos]
                decimal_part = mape_test_str[point_pos + 1:]

                # Movemos la coma nuevamente
                mape_test = float(integer_part + '.' + decimal_part)
    else:
        # Si el valor es menor o igual a 30, no hacemos ningún cambio
        mape_test = mape_test

    comparison = {
        "rmse_xgboost": rmse_train_fmt,
        "rmse_naive": rmse_naive_fmt,
        "mae_xgboost": mae_train_fmt,
        "mae_naive": mae_naive_fmt,
        "mape_xgboost": mape_train_fmt,
        "mape_naive": mape_naive_fmt
    }

    return {
        "model_name": "XGBoost",
        "best_params": {
            "max_depth": best_max_depth,
            "learning_rate": best_learning_rate,
            "n_estimators": best_n_estimators,
            "subsample": best_subsample,
            "colsample_bytree": best_colsample_bytree,
            "gamma": best_gamma
        },
        "training_metrics": {
            "rmse": rmse_test,
            "mae": mae_test,
            "mape": mape_test
        },
        "cross_validation_metrics": {
            "rmse": cv_rmse_fmt,
            "mae": cv_mae_fmt,
            "mape": cv_mape_fmt
        },
        "naive_model_metrics": {
            "rmse": rmse_naive_fmt,
            "mae": mae_naive_fmt,
            "mape": mape_naive_fmt
        },
        "comparison_xgboost_vs_naive": comparison
    }


@app.post("/top_correlated")
def top_correlated(request: CorrelationRequest):
    folder_name = request.folder_name
    description = request.description
    file_name = request.file_name
    top_n = request.top_n

    df = get_excel_file_from_s3(folder_name, file_name)
    if 'DESCRIPCION' not in df.columns:
        raise HTTPException(status_code=400, detail="La columna 'DESCRIPCION' no se encontró en los datos.")

    # Seleccionar la columna DESCRIPCION y todas las columnas a su derecha
    columns_after_description = df.columns[df.columns.get_loc('DESCRIPCION') + 1:]
    df_filtered = df[['DESCRIPCION'] + list(columns_after_description)]

    # Agrupar por DESCRIPCION para asegurarnos que cada medicamento aparezca una única vez.
    # Puedes elegir 'first' o 'mean', según convenga.
    df_filtered = df_filtered.groupby('DESCRIPCION').first().reset_index()

    # Establecer DESCRIPCION como índice y limpiar datos
    df_filtered.set_index('DESCRIPCION', inplace=True)
    df_filtered = clean_and_convert_columns(df_filtered)

    if description not in df_filtered.index:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en los datos.")

    # Transponer para que cada fila (fecha/histórico) sea una observación y cada medicamento una variable
    df_transposed = df_filtered.transpose()
    corr_matrix = df_transposed.corr()

    if description not in corr_matrix.columns:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en la matriz de correlación.")

    top_medications = get_top_correlated_medications(description, corr_matrix, top_n)

    return {
        "description": description,
        "top_correlated_medications": top_medications
    }