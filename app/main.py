from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from io import BytesIO
import boto3
import os
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from math import sqrt
import warnings
from bayes_opt import BayesianOptimization

warnings.filterwarnings("ignore")
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modelos de request
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


def mape_metric(y_true, y_pred):
    """Calcula el MAPE evitando división por cero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    y_true_nz = y_true[non_zero]
    y_pred_nz = y_pred[non_zero]
    if len(y_true_nz) == 0:
        return np.nan
    return np.mean(np.abs((y_true_nz - y_pred_nz) / y_true_nz)) * 100


# --- Endpoint /predict modificado para optimización bayesiana ---
@app.post("/predict")
def predict(request: PredictRequest):
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description
    periods = request.periods

    df = get_excel_file_from_s3(folder_name, file_name)

    if "DESCRIPCION" not in df.columns:
        raise HTTPException(status_code=400, detail="La columna 'DESCRIPCION' no se encontró en los datos.")
    col_index = df.columns.get_loc("DESCRIPCION")
    df = df.iloc[:, col_index:]  # Se toman DESCRIPCION y las columnas a la derecha

    df_melted = df.melt(id_vars=["DESCRIPCION"], var_name="Fecha", value_name="Valor")
    try:
        df_melted["Fecha"] = pd.to_datetime(df_melted["Fecha"], format="%m-%Y")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error al convertir la columna Fecha. Verifica el formato.")
    df_filtered = df_melted[df_melted["DESCRIPCION"] == description].copy()
    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en los datos.")

    # Preparar la serie temporal
    df_filtered.set_index("Fecha", inplace=True)
    df_filtered = df_filtered.sort_index()
    df_filtered["Valor"] = pd.to_numeric(df_filtered["Valor"], errors='coerce')
    df_filtered.dropna(inplace=True)

    # Se requiere cantidad mínima de datos para optimización
    if len(df_filtered) < 15:
        raise HTTPException(status_code=400, detail="No hay suficientes datos para optimizar el modelo.")

    # División 80/20 para optimización
    train_size = int(len(df_filtered) * 0.8)
    train, test = df_filtered.iloc[:train_size], df_filtered.iloc[train_size:]
    if len(train) < 10 or len(test) < 5:
        raise HTTPException(status_code=400, detail="Datos insuficientes para entrenamiento y prueba.")

    # Definir la función objetivo para optimización (se retorna negativo RMSE para maximización)
    def sarimax_cv(p, d, q, P, D, Q):
        p, d, q = int(round(p)), int(round(d)), int(round(q))
        P, D, Q = int(round(P)), int(round(D)), int(round(Q))
        try:
            model = SARIMAX(train["Valor"],
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            preds = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
            rmse_val = sqrt(mean_squared_error(test["Valor"], preds))
            return -rmse_val  # Se retorna negativo para maximizar
        except Exception:
            return -1e6  # Penalización si el modelo falla

    # Límites de búsqueda para los hiperparámetros
    pbounds = {
        'p': (0, 3),
        'd': (0, 1),
        'q': (0, 3),
        'P': (0, 2),
        'D': (0, 1),
        'Q': (0, 2)
    }

    optimizer = BayesianOptimization(
        f=sarimax_cv,
        pbounds=pbounds,
        random_state=42,
        verbose=0
    )
    # Realiza la optimización
    optimizer.maximize(init_points=5, n_iter=15)
    best = optimizer.max['params']
    best_p = int(round(best['p']))
    best_d = int(round(best['d']))
    best_q = int(round(best['q']))
    best_P = int(round(best['P']))
    best_D = int(round(best['D']))
    best_Q = int(round(best['Q']))

    # Una vez obtenidos los mejores parámetros, se entrena con la serie completa
    try:
        final_model = SARIMAX(df_filtered["Valor"],
                              order=(best_p, best_d, best_q),
                              seasonal_order=(best_P, best_D, best_Q, 12),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
        final_fit = final_model.fit(disp=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ajustar el modelo final SARIMAX: {str(e)}")

    # Pronosticar los periodos solicitados
    last_date = df_filtered.index.max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
    forecast = final_fit.forecast(steps=periods)

    historical_data = df_filtered.reset_index().rename(columns={"Fecha": "ds", "Valor": "y"}).to_dict(orient="records")
    predictions = [{"ds": date, "yhat": pred} for date, pred in zip(future_dates, forecast)]

    return {
        "historical_data": historical_data,
        "predictions": predictions,
        "model": "SARIMAX",
        "best_order": (best_p, best_d, best_q),
        "best_seasonal_order": (best_P, best_D, best_Q, 12)
    }


# --- Endpoint /evaluate-model modificado para optimización bayesiana ---
@app.post("/evaluate-model")
def evaluate_model(request: PredictRequest):
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description

    # Cargar el archivo Excel desde S3
    df = get_excel_file_from_s3(folder_name, file_name)
    if "DESCRIPCION" not in df.columns:
        raise HTTPException(status_code=400, detail="La columna 'DESCRIPCION' no se encontró en los datos.")
    col_index = df.columns.get_loc("DESCRIPCION")
    df = df.iloc[:, col_index:]

    # Convertir de formato wide a long
    df_melted = df.melt(id_vars=["DESCRIPCION"], var_name="Fecha", value_name="Valor")
    try:
        df_melted["Fecha"] = pd.to_datetime(df_melted["Fecha"], format="%m-%Y")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error al convertir la columna Fecha. Verifica el formato.")

    # Filtrar por el medicamento solicitado
    df_med = df_melted[df_melted["DESCRIPCION"] == description].copy()
    if df_med.empty:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en los datos.")

    # Preparar la serie temporal
    df_med = df_med[["Fecha", "Valor"]].rename(columns={"Fecha": "ds", "Valor": "y"})
    try:
        df_med["ds"] = pd.to_datetime(df_med["ds"], format="%m-%Y")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error al convertir la columna ds. Verifica el formato.")
    df_med.set_index("ds", inplace=True)
    df_med = df_med.sort_index()
    df_med["y"] = pd.to_numeric(df_med["y"], errors='coerce')
    df_med.dropna(inplace=True)

    if len(df_med) < 15:
        raise HTTPException(status_code=400, detail="No hay suficientes datos para optimizar el modelo.")

    # División 80/20 para entrenamiento y prueba
    train_size = int(len(df_med) * 0.8)
    train = df_med.iloc[:train_size]
    test = df_med.iloc[train_size:]
    if len(train) < 10 or len(test) < 5:
        raise HTTPException(status_code=400, detail="Datos insuficientes para entrenamiento y prueba.")

    # Función objetivo para optimización bayesiana: se entrena sobre train y se evalúa en test
    def sarimax_cv(p, d, q, P, D, Q):
        p, d, q = int(round(p)), int(round(d)), int(round(q))
        P, D, Q = int(round(P)), int(round(D)), int(round(Q))
        try:
            model = SARIMAX(train["y"],
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            preds = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
            rmse_val = sqrt(mean_squared_error(test["y"], preds))
            return -rmse_val  # Retornamos negativo para maximizar (mínimo RMSE)
        except Exception:
            return -1e6

    # Límites para la optimización
    pbounds = {
        'p': (0, 3),
        'd': (0, 1),
        'q': (0, 3),
        'P': (0, 2),
        'D': (0, 1),
        'Q': (0, 2)
    }

    optimizer = BayesianOptimization(
        f=sarimax_cv,
        pbounds=pbounds,
        random_state=42,
        verbose=0
    )
    optimizer.maximize(init_points=5, n_iter=15)
    best = optimizer.max['params']
    best_p = int(round(best['p']))
    best_d = int(round(best['d']))
    best_q = int(round(best['q']))
    best_P = int(round(best['P']))
    best_D = int(round(best['D']))
    best_Q = int(round(best['Q']))

    # Entrenar el modelo final sobre el conjunto de entrenamiento y evaluar en test (estilo Colab)
    try:
        final_model = SARIMAX(train["y"],
                              order=(best_p, best_d, best_q),
                              seasonal_order=(best_P, best_D, best_Q, 12),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
        final_fit = final_model.fit(disp=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ajustar el modelo final SARIMAX: {str(e)}")

    preds_test = final_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
    rmse_test = sqrt(mean_squared_error(test["y"], preds_test))
    mae_test = mean_absolute_error(test["y"], preds_test)
    mape_test = mape_metric(test["y"], preds_test)


    try:
        full_model = SARIMAX(df_med["y"],
                             order=(best_p, best_d, best_q),
                             seasonal_order=(best_P, best_D, best_Q, 12),
                             enforce_stationarity=False,
                             enforce_invertibility=False)
        full_fit = full_model.fit(disp=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ajustar el modelo completo SARIMAX: {str(e)}")

    # Training metrics (in-sample sobre la serie completa)
    full_pred = full_fit.fittedvalues
    rmse_train = sqrt(mean_squared_error(df_med["y"], full_pred))
    mae_train = mean_absolute_error(df_med["y"], full_pred)
    mape_train = mape_metric(df_med["y"], full_pred)

    # Validación cruzada con ventana móvil en la serie completa
    def rolling_forecast_cv_sarimax(series, order, seasonal_order, initial, horizon, step=1):
        rmse_list = []
        mae_list = []
        mape_list = []
        for i in range(initial, len(series) - horizon + 1, step):
            train_fold = series.iloc[:i]
            test_fold = series.iloc[i:i + horizon]
            try:
                model_fold = SARIMAX(train_fold,
                                     order=order,
                                     seasonal_order=seasonal_order,
                                     enforce_stationarity=False,
                                     enforce_invertibility=False)
                results_fold = model_fold.fit(disp=False)
                forecast_fold = results_fold.forecast(steps=horizon)
                rmse_fold = sqrt(mean_squared_error(test_fold, forecast_fold))
                mae_fold = mean_absolute_error(test_fold, forecast_fold)
                mape_fold = mape_metric(test_fold, forecast_fold)
                rmse_list.append(rmse_fold)
                mae_list.append(mae_fold)
                mape_list.append(mape_fold)
            except Exception:
                continue
        if len(rmse_list) == 0:
            return np.nan, np.nan, np.nan
        return np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list)

    initial = int(len(df_med) * 0.7) if len(df_med) > 10 else 1
    horizon = 1
    cv_rmse, cv_mae, cv_mape = rolling_forecast_cv_sarimax(df_med["y"], (best_p, best_d, best_q),
                                                           (best_P, best_D, best_Q, 12), initial, horizon)

    # Métricas del modelo naive (predicción = valor anterior) sobre la serie completa
    naive_pred = df_med["y"].shift(1).dropna()
    actual_naive = df_med["y"].iloc[1:]
    rmse_naive = sqrt(mean_squared_error(actual_naive, naive_pred))
    mae_naive = mean_absolute_error(actual_naive, naive_pred)
    mape_naive = mape_metric(actual_naive, naive_pred)

    comparison = {
        "rmse_sarimax": rmse_train,
        "rmse_naive": rmse_naive,
        "mae_sarimax": mae_train,
        "mae_naive": mae_naive,
        "mape_sarimax": mape_train,
        "mape_naive": mape_naive
    }

    return {
        "model_name": "SARIMAX",
        "best_order": (best_p, best_d, best_q),
        "best_seasonal_order": (best_P, best_D, best_Q, 12),
        "training_metrics": {
            "rmse": rmse_train,
            "mae": mae_train,
            "mape": mape_train
        },
        "cross_validation_metrics": {
            "rmse": rmse_test,
            "mae": mae_test,
            "mape": mape_test
        },
        "naive_model_metrics": {
            "rmse": rmse_naive,
            "mae": mae_naive,
            "mape": mape_naive
        },
        "comparison_sarimax_vs_naive": comparison
    }


# El endpoint /top_correlated se mantiene sin cambios
@app.post("/top_correlated")
def top_correlated(request: CorrelationRequest):
    folder_name = request.folder_name
    description = request.description
    file_name = request.file_name
    top_n = request.top_n

    df = get_excel_file_from_s3(folder_name, file_name)
    if 'DESCRIPCION' not in df.columns:
        raise HTTPException(status_code=400, detail="La columna 'DESCRIPCION' no se encontró en los datos.")

    columns_after_description = df.columns[df.columns.get_loc('DESCRIPCION') + 1:]
    df_filtered = df[['DESCRIPCION'] + list(columns_after_description)]
    df_filtered = df_filtered.groupby('DESCRIPCION').first().reset_index()
    df_filtered.set_index('DESCRIPCION', inplace=True)
    df_filtered = clean_and_convert_columns(df_filtered)

    if description not in df_filtered.index:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en los datos.")

    df_transposed = df_filtered.transpose()
    corr_matrix = df_transposed.corr()

    if description not in corr_matrix.columns:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en la matriz de correlación.")

    # Obtener las top correlaciones (función auxiliar)
    def get_top_correlated_medications(medication_name, corr_matrix, top_n=5):
        try:
            medication_correlations = corr_matrix[medication_name]
            medication_correlations = medication_correlations.drop(labels=[medication_name])
            sorted_correlations = medication_correlations.abs().sort_values(ascending=False)
            top_medications = sorted_correlations.head(top_n)
            result = []
            for med in top_medications.index:
                result.append({
                    "medication": med,
                    "correlation": medication_correlations[med]
                })
            return result
        except KeyError:
            raise HTTPException(status_code=404,
                                detail=f"El medicamento '{medication_name}' no se encontró en la matriz de correlación.")

    top_medications = get_top_correlated_medications(description, corr_matrix, top_n)

    return {
        "description": description,
        "top_correlated_medications": top_medications
    }
