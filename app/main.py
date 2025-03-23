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

# Cargar variables de entorno
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


def rolling_forecast_cv_sarimax(series, order, seasonal_order, initial, horizon, step=1):
    """
    Realiza validación cruzada mediante forecast en ventana móvil.
    Para cada corte, entrena el modelo y pronostica 'horizon' pasos.
    """
    rmse_list = []
    mae_list = []
    mape_list = []

    for i in range(initial, len(series) - horizon + 1, step):
        train = series.iloc[:i]
        test = series.iloc[i:i + horizon]
        try:
            model = SARIMAX(train,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results = model.fit(disp=False)
        except Exception as e:
            continue  # Si falla en algún corte, se salta ese fold
        forecast = results.forecast(steps=horizon)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mae = mean_absolute_error(test, forecast)
        mape = mape_metric(test, forecast)
        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)

    if len(rmse_list) == 0:
        return np.nan, np.nan, np.nan
    return np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list)


@app.post("/predict")
def predict(request: PredictRequest):
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description
    periods = request.periods

    df = get_excel_file_from_s3(folder_name, file_name)

    # Verificar que la columna 'DESCRIPCION' exista y tomar desde esa columna en adelante
    if "DESCRIPCION" not in df.columns:
        raise HTTPException(status_code=400, detail="La columna 'DESCRIPCION' no se encontró en los datos.")
    col_index = df.columns.get_loc("DESCRIPCION")
    df = df.iloc[:, col_index:]  # Toma la columna DESCRIPCION y todas las columnas a su derecha

    df_melted = df.melt(id_vars=["DESCRIPCION"], var_name="Fecha", value_name="Valor")
    df_melted["Fecha"] = pd.to_datetime(df_melted["Fecha"], format="%m-%Y")
    df_filtered = df_melted[df_melted["DESCRIPCION"] == description].copy()
    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en los datos.")

    # Preparar datos para SARIMAX
    df_filtered.set_index("Fecha", inplace=True)
    df_filtered = df_filtered.sort_index()
    df_filtered["Valor"] = pd.to_numeric(df_filtered["Valor"], errors='coerce')
    df_filtered.dropna(inplace=True)

    # División opcional en entrenamiento y prueba
    train_size = int(len(df_filtered) * 0.8)
    train, test = df_filtered[:train_size], df_filtered[train_size:]

    try:
        model = SARIMAX(train["Valor"],
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        sarima_model = model.fit(disp=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ajustar el modelo SARIMAX: {str(e)}")

    last_date = df_filtered.index.max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
    forecast = sarima_model.forecast(steps=periods)

    historical_data = df_filtered.reset_index().rename(columns={"Fecha": "ds", "Valor": "y"}).to_dict(orient="records")
    predictions = [{"ds": date, "yhat": pred} for date, pred in zip(future_dates, forecast)]
    return {
        "historical_data": historical_data,
        "predictions": predictions,
        "model": "SARIMAX"
    }


@app.post("/evaluate-model")
def evaluate_model(request: PredictRequest):
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description

    df = get_excel_file_from_s3(folder_name, file_name)

    # Asegurarse de que se seleccione la columna "DESCRIPCION" y las que le siguen
    if "DESCRIPCION" not in df.columns:
        raise HTTPException(status_code=400, detail="La columna 'DESCRIPCION' no se encontró en los datos.")
    col_index = df.columns.get_loc("DESCRIPCION")
    df = df.iloc[:, col_index:]

    df_melted = df.melt(id_vars=["DESCRIPCION"], var_name="Fecha", value_name="Valor")
    df_melted["Fecha"] = pd.to_datetime(df_melted["Fecha"], format="%m-%Y")
    df_med = df_melted[df_melted["DESCRIPCION"] == description].copy()
    if df_med.empty:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en los datos.")

    # Preparar la serie temporal
    df_med.set_index("Fecha", inplace=True)
    df_med = df_med.sort_index()
    df_med["Valor"] = pd.to_numeric(df_med["Valor"], errors='coerce')
    df_med.dropna(inplace=True)
    series = df_med["Valor"]

    # Hiperparámetros "óptimos" para SARIMAX (se usan los mismos que en /predict)
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    # Ajustar el modelo SARIMAX sobre la serie completa
    try:
        model = SARIMAX(series,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        sarima_model = model.fit(disp=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ajustar el modelo SARIMAX: {str(e)}")

    # Predicciones in-sample (predicción uno a uno en todo el histórico)
    in_sample_pred = sarima_model.get_prediction(start=series.index[1], end=series.index[-1], dynamic=False)
    pred_mean = sarima_model.fittedvalues
    actual = series[pred_mean.index]

    rmse_train = np.sqrt(mean_squared_error(series, pred_mean))
    mae_train = mean_absolute_error(series, pred_mean)
    mape_train = mape_metric(series, pred_mean)

    # Validación cruzada con ventana móvil (forecast de 1 paso)
    initial = int(len(series) * 0.7) if len(series) > 10 else 1
    horizon = 1  # Pronosticar 1 mes adelante
    cv_rmse, cv_mae, cv_mape = rolling_forecast_cv_sarimax(series, order, seasonal_order, initial, horizon, step=1)

    # Modelo naive: la predicción es el último valor observado
    naive_pred = series.shift(1).dropna()
    actual = series.iloc[1:]
    rmse_naive = np.sqrt(mean_squared_error(actual, naive_pred))
    mae_naive = mean_absolute_error(actual, naive_pred)
    mape_naive = mape_metric(actual, naive_pred)

    return {
        "model_name": "SARIMAX",
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
            "rmse": rmse_naive,
            "mae": mae_naive,
            "mape": mape_naive
        },
        "comparison_sarimax_vs_naive": {
            "rmse_sarimax": rmse_train,
            "rmse_naive": rmse_naive,
            "mae_sarimax": mae_train,
            "mae_naive": mae_naive,
            "mape_sarimax": mape_train,
            "mape_naive": mape_naive
        }
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
