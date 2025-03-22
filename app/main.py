from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from prophet import Prophet
from io import BytesIO
import boto3
import os
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics
from bayes_opt import BayesianOptimization
import hashlib
import math

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

# Ruta del archivo de cache (resultados guardados)
cache_file_path = "cache_results.csv"


class PredictRequest(BaseModel):
    folder_name: str
    description: str
    file_name: str
    periods: int  # Número de meses a predecir


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


# Función para calcular MAPE evitando división por cero
def mape_metric(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    y_true_non_zero = y_true[non_zero_indices]
    y_pred_non_zero = y_pred[non_zero_indices]
    if len(y_true_non_zero) == 0:
        return np.nan
    return np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100


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


# ---------------------------
# Funciones para cacheo
# ---------------------------
def compute_data_hash(df):
    """Genera un hash MD5 a partir del contenido del DataFrame (ordenado por 'ds')."""
    df_sorted = df.sort_values("ds")
    data_string = df_sorted.to_csv(index=False)
    return hashlib.md5(data_string.encode('utf-8')).hexdigest()


def get_cached_result(description, data_hash):
    """Si existe un registro en el archivo de cache para este medicamento y hash, se retorna."""
    if os.path.exists(cache_file_path):
        cache_df = pd.read_csv(cache_file_path)
        cached = cache_df[(cache_df["description"] == description) & (cache_df["data_hash"] == data_hash)]
        if not cached.empty:
            return cached.iloc[0].to_dict()
    return None


def update_cache(result):
    """Actualiza (o agrega) el registro para el medicamento en el archivo de cache."""
    if os.path.exists(cache_file_path):
        cache_df = pd.read_csv(cache_file_path)
    else:
        cache_df = pd.DataFrame()

    if not cache_df.empty:
        mask = (cache_df["description"] == result["description"]) & (cache_df["data_hash"] == result["data_hash"])
        if mask.any():
            cache_df.loc[mask, :] = pd.DataFrame([result])
        else:
            cache_df = pd.concat([cache_df, pd.DataFrame([result])], ignore_index=True)
    else:
        cache_df = pd.DataFrame([result])
    cache_df.to_csv(cache_file_path, index=False)


# ---------------------------
# Función para optimizar hiperparámetros
# ---------------------------
def optimize_prophet_hyperparameters(df, initial='730 days', period='180 days', horizon='365 days'):
    def prophet_cv(cps, sps):
        try:
            model = Prophet(
                changepoint_prior_scale=cps,
                seasonality_prior_scale=sps,
                weekly_seasonality=True,
                seasonality_mode='additive'
            )
            model.fit(df)
            df_cv = cross_validation(
                model,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel="processes"
            )
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmse = df_p['rmse'].mean()
            return -rmse  # Se retorna negativo porque BayesianOptimization maximiza
        except Exception as e:
            return -1e6

    optimizer = BayesianOptimization(
        f=prophet_cv,
        pbounds={'cps': (0.001, 0.05), 'sps': (0.1, 10)},
        random_state=42,
        verbose=0
    )
    optimizer.maximize(init_points=5, n_iter=15)
    best_params = optimizer.max['params']
    return best_params


# ---------------------------
# Función para computar solo los hiperparámetros
# ---------------------------
def compute_hyperparameters(df_med, description):
    best_params = optimize_prophet_hyperparameters(df_med)
    best_cps = best_params['cps']
    best_sps = best_params['sps']
    result = {
        "description": description,
        "data_hash": compute_data_hash(df_med),
        "best_cps": best_cps,
        "best_sps": best_sps
    }
    return result


# ---------------------------
# Endpoints
# ---------------------------
@app.post("/predict")
def predict(request: PredictRequest):
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description
    periods = request.periods

    df = get_excel_file_from_s3(folder_name, file_name)

    # Transformar la data

    df_filtered = process_data(df, description)

    # Calcular hash de la data histórica
    data_hash = compute_data_hash(df_filtered)
    cached = get_cached_result(description, data_hash)

    if cached is not None:
        best_cps = cached["best_cps"]
        best_sps = cached["best_sps"]
        print(f"Usando hiperparámetros cacheados para {description}")
    else:
        computed = compute_hyperparameters(df_filtered, description)
        best_cps = computed["best_cps"]
        best_sps = computed["best_sps"]
        update_cache(computed)
        print(f"Se han calculado y guardado nuevos hiperparámetros para {description}")

    # Entrenar modelo con los hiperparámetros obtenidos
    model = Prophet(
        changepoint_prior_scale=best_cps,
        seasonality_prior_scale=best_sps,
        weekly_seasonality=True,
        seasonality_mode='additive'
    )
    model.fit(df_filtered)
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)

    historical_data = df_filtered.to_dict(orient="records")
    predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient="records")

    return {
        "historical_data": historical_data,
        "predictions": predictions,
        "best_hyperparameters": {"cps": best_cps, "sps": best_sps},
        "model": "Prophet"
    }


@app.post("/evaluate-model")
def evaluate_model(request: PredictRequest):
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description

    df = get_excel_file_from_s3(folder_name, file_name)

    df_filtered = process_data(df, description)

    data_hash = compute_data_hash(df_filtered)
    cached = get_cached_result(description, data_hash)

    if cached is not None:
        hyperparams = cached
        print(f"Usando hiperparámetros cacheados para {description}")
    else:
        hyperparams = compute_hyperparameters(df_filtered, description)
        update_cache(hyperparams)
        print(f"Se han calculado y guardado nuevos hiperparámetros para {description}")

    # Re-entrenar el modelo usando los hiperparámetros cacheados
    model = Prophet(
        changepoint_prior_scale=hyperparams["best_cps"],
        seasonality_prior_scale=hyperparams["best_sps"],
        weekly_seasonality=True,
        seasonality_mode='additive'
    )
    model.fit(df_filtered)

    # Cálculo de métricas de validación (se recalculan en cada consulta)
    train_forecast = model.predict(df_filtered)
    rmse_train = np.sqrt(mean_squared_error(df_filtered['y'], train_forecast['yhat']))
    mae_train = mean_absolute_error(df_filtered['y'], train_forecast['yhat'])
    mape_train = mape_metric(df_filtered['y'], train_forecast['yhat'])

    df_cv = cross_validation(
        model,
        initial='730 days',
        period='180 days',
        horizon='365 days',
        parallel="processes"
    )
    df_p = performance_metrics(df_cv, rolling_window=1)
    cv_rmse = df_p['rmse'].mean()
    cv_mae = df_p['mae'].mean()
    cv_mape = df_p['mape'].mean() if 'mape' in df_p.columns else np.nan

    # Cálculo de métricas para el modelo Naive (benchmark)
    naive_predictions = df_filtered['y'].shift(1).dropna()
    real_values_naive = df_filtered['y'].iloc[1:]
    naive_rmse = np.sqrt(mean_squared_error(real_values_naive, naive_predictions))
    naive_mae = mean_absolute_error(real_values_naive, naive_predictions)
    naive_mape = mape_metric(real_values_naive, naive_predictions)

    return {
        "model_name": "Prophet (Optimized)",
        "best_hyperparameters": {"cps": hyperparams["best_cps"], "sps": hyperparams["best_sps"]},
        "training_metrics": {
            "rmse": rmse_train,
            "mae": mae_train,
            "mape": mape_train
        },
        "cross_validation_metrics": {
            "rmse": cv_rmse,
            "mae": cv_mae,
            "mape": cv_mape
        },
        "naive_model_metrics": {
            "rmse": naive_rmse,
            "mae": naive_mae,
            "mape": naive_mape
        },
        "comparison_prophet_vs_naive": {
            "rmse_prophet": rmse_train,
            "naive_rmse": naive_rmse,
            "mae_prophet": mae_train,
            "naive_mae": naive_mae,
            "mape_prophet": mape_train,
            "naive_mape": naive_mape
        }
    }


class CorrelationRequest(BaseModel):
    folder_name: str
    description: str
    file_name: str
    top_n: int = 5  # Número de medicamentos a retornar, por defecto 5


def clean_and_convert_columns(df):
    # Se crea una copia para evitar el SettingWithCopyWarning
    df = df.copy()
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
        # Si se obtiene un DataFrame (por duplicidad), se extrae la primera columna
        if isinstance(medication_correlations, pd.DataFrame):
            medication_correlations = medication_correlations.iloc[:, 0]
        # Excluir la correlación consigo mismo
        medication_correlations = medication_correlations.drop(labels=[medication_name])
        sorted_correlations = medication_correlations.abs().sort_values(ascending=False)
        top_medications = sorted_correlations.head(top_n)
        result = []
        for med in top_medications.index:
            corr_value = medication_correlations[med]
            result.append({
                "medication": str(med),
                "correlation": float(corr_value)
            })
        return result
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"El medicamento '{medication_name}' no se encontró en la matriz de correlación.")


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
