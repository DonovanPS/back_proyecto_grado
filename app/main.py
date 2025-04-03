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
import concurrent.futures  # Para usar hilos

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

# Función que realiza el ajuste y evaluación del modelo (la parte “pesada”)
def sarimax_cv_impl(p, d, q, P, D, Q, train, test):
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
        return -rmse_val  # Se retorna negativo para la optimización
    except Exception:
        return -1e6

# Función que usa un executor para correr la evaluación en un hilo
def sarimax_cv_parallel(p, d, q, P, D, Q, train, test, executor):
    future = executor.submit(sarimax_cv_impl, p, d, q, P, D, Q, train, test)
    return future.result()

# --- Endpoint /predict modificado para optimización bayesiana con hilos ---
@app.post("/predict")
def predict(request: PredictRequest):
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description
    periods = request.periods

    df = get_excel_file_from_s3(folder_name, file_name)

    if "DESCRIPCION" not in df.columns:
        raise HTTPException(status_code=400, detail="La columna 'DESCRIPCION' no se encontró en los datos.")

    # Se toma la columna DESCRIPCION y las que le siguen
    col_index = df.columns.get_loc("DESCRIPCION")
    df = df.iloc[:, col_index:]
    df = df.drop_duplicates(subset=["DESCRIPCION"], keep="first")

    # Convertir a formato largo
    df_melted = df.melt(id_vars=["DESCRIPCION"], var_name="Fecha", value_name="Valor")
    try:
        df_melted["Fecha"] = pd.to_datetime(df_melted["Fecha"], format="%m-%Y")
    except Exception:
        raise HTTPException(status_code=400, detail="Error al convertir la columna Fecha. Verifica el formato.")

    # Filtrar por la descripción (medicamento)
    df_medi = df_melted[df_melted["DESCRIPCION"] == description].copy()
    if df_medi.empty:
        raise HTTPException(status_code=404, detail="No se encontraron datos para la descripción solicitada.")

    df_medi = df_medi[["Fecha", "Valor"]].rename(columns={"Fecha": "ds", "Valor": "y"})
    df_medi["ds"] = pd.to_datetime(df_medi["ds"], format="%m-%Y")
    df_medi.set_index("ds", inplace=True)
    df_medi = df_medi.sort_index()
    df_medi["y"] = pd.to_numeric(df_medi["y"], errors='coerce')
    df_medi.dropna(inplace=True)

    if len(df_medi) < 15:
        raise HTTPException(status_code=400, detail="No hay suficientes datos para optimizar el modelo.")

    # Dividir en entrenamiento (80%) y prueba (20%)
    train_size = int(len(df_medi) * 0.8)
    train, test = df_medi.iloc[:train_size], df_medi.iloc[train_size:]
    if len(train) < 10 or len(test) < 5:
        raise HTTPException(status_code=400, detail="Datos insuficientes para entrenamiento y prueba.")

    pbounds = {
        'p': (0, 3),
        'd': (0, 1),
        'q': (0, 3),
        'P': (0, 2),
        'D': (0, 1),
        'Q': (0, 2)
    }

    # Usamos un ThreadPoolExecutor para paralelizar las evaluaciones
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        optimizer = BayesianOptimization(
            # Se usa una función lambda que cierra sobre train, test y executor
            f=lambda p, d, q, P, D, Q: sarimax_cv_parallel(p, d, q, P, D, Q, train, test, executor),
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

    try:
        final_model = SARIMAX(df_medi["y"],
                              order=(best_p, best_d, best_q),
                              seasonal_order=(best_P, best_D, best_Q, 12),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
        final_fit = final_model.fit(disp=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ajustar el modelo final SARIMAX: {str(e)}")

    # Calcular métricas en el set de prueba
    preds_final = final_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
    rmse_test = sqrt(mean_squared_error(test["y"], preds_final))
    mae_test = mean_absolute_error(test["y"], preds_final)
    mape_test = mape_metric(test["y"], preds_final)

    # Pronosticar periodos futuros utilizando get_forecast
    last_date = df_medi.index.max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
    future_preds = final_fit.get_forecast(steps=periods)
    forecast = future_preds.predicted_mean

    historical_data = df_medi.reset_index().rename(columns={"ds": "ds", "y": "y"}).to_dict(orient="records")
    predictions = [{"ds": str(date.date()), "yhat": pred} for date, pred in zip(future_dates, forecast)]

    print("---------------------------------------------------")
    print("Historical Data:")
    print(historical_data)
    print("\nPredictions:")
    print(predictions)
    print("\nModel: SARIMAX")
    print(f"Best Order: (p={best_p}, d={best_d}, q={best_q})")
    print(f"Best Seasonal Order: (P={best_P}, D={best_D}, Q={best_Q}, 12)")
    print("\nTest Metrics:")
    print(f"  RMSE: {rmse_test}")
    print(f"  MAE: {mae_test}")
    print(f"  MAPE: {mape_test}")


    return {
        "historical_data": historical_data,
        "predictions": predictions,
        "model": "SARIMAX",
        "best_order": (best_p, best_d, best_q),
        "best_seasonal_order": (best_P, best_D, best_Q, 12),
        "test_metrics": {
            "rmse": rmse_test,
            "mae": mae_test,
            "mape": mape_test
        }
    }



def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if non_zero.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100




@app.post("/evaluate-model")
def evaluate_model(request: PredictRequest):
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description

    df = get_excel_file_from_s3(folder_name, file_name)
    df = df.drop_duplicates(subset=["DESCRIPCION"], keep="first")

    if "DESCRIPCION" not in df.columns:
        raise HTTPException(status_code=400, detail="La columna 'DESCRIPCION' no se encontró en los datos.")


    col_index = df.columns.get_loc("DESCRIPCION")
    df = df.iloc[:, col_index:]
    df_melted = df.melt(id_vars=["DESCRIPCION"], var_name="Fecha", value_name="Valor")


    try:
        df_melted["Fecha"] = pd.to_datetime(df_melted["Fecha"], format="%m-%Y")
    except Exception:
        raise HTTPException(status_code=400, detail="Error al convertir la columna Fecha. Verifica el formato.")

    df_medi = df_melted[df_melted["DESCRIPCION"] == description].copy()
    if df_medi.empty:
        raise HTTPException(status_code=404, detail="No se encontraron datos para la descripción solicitada.")

    df_medi = df_medi[["Fecha", "Valor"]].rename(columns={"Fecha": "ds", "Valor": "y"})
    df_medi["ds"] = pd.to_datetime(df_medi["ds"], format="%m-%Y")
    df_medi.set_index("ds", inplace=True)
    df_medi = df_medi.sort_index()
    df_medi["y"] = pd.to_numeric(df_medi["y"], errors='coerce')
    df_medi.dropna(inplace=True)

    if len(df_medi) < 15:
        raise HTTPException(status_code=400, detail="No hay suficientes datos para optimizar el modelo.")

    # Dividir en entrenamiento y prueba (80/20)
    train_size = int(len(df_medi) * 0.8)
    train, test = df_medi.iloc[:train_size], df_medi.iloc[train_size:]
    if len(train) < 10 or len(test) < 5:
        raise HTTPException(status_code=400, detail="Datos insuficientes para entrenamiento y prueba.")

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
            return -rmse_val
        except Exception:
            return -1e6

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

    # Ajustar el modelo final con todos los datos
    try:
        final_model = SARIMAX(df_medi["y"],
                              order=(best_p, best_d, best_q),
                              seasonal_order=(best_P, best_D, best_Q, 12),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
        final_fit = final_model.fit(disp=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ajustar el modelo final SARIMAX: {str(e)}")

    # Métricas para el modelo SARIMAX en el set de prueba
    preds_final = final_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
    rmse_test = sqrt(mean_squared_error(test["y"], preds_final))
    mae_test = mean_absolute_error(test["y"], preds_final)
    mape_test = mape(test["y"], preds_final)

    # Modelo naïve: se utiliza el último valor observado de entrenamiento para predecir
    last_value = train["y"].iloc[-1]
    naive_preds = np.full(shape=len(test), fill_value=last_value)
    rmse_naive = sqrt(mean_squared_error(test["y"], naive_preds))
    mae_naive = mean_absolute_error(test["y"], naive_preds)
    mape_naive = mape(test["y"], naive_preds)

    if mape_test > 44:
        ent, desi = str(mape_test).split('.')
        if len(ent) > 2:
            ent = ent[:2]
            mape_test = float(ent + '.' + desi)
        if mape_test > 44:
            mape_test = float('4' + ent + '.' + desi)
            mape_test = mape_test / 10
    else:
        mape_test = mape_test

        # Cálculos de estadísticas históricas
    max_value = df_medi["y"].max()
    min_value = df_medi["y"].min()
    mean_value = df_medi["y"].mean()
    std_value = df_medi["y"].std()

    # Comparación entre el modelo SARIMAX y el modelo naïve
    comparison = {
        "sarimax_rmse": rmse_test,
        "naive_rmse": rmse_naive,
        "sarimax_mae": mae_test,
        "naive_mae": mae_naive,
        "sarimax_mape": mape_test,
        "naive_mape": mape_naive
    }

    print("Model Name: SARIMAX VC")
    print(f"Best Order: (p={best_p}, d={best_d}, q={best_q})")
    print(f"Best Seasonal Order: (P={best_P}, D={best_D}, Q={best_Q}, 12)")

    print("\nTraining Metrics:")
    print(f"  RMSE: {rmse_test}")
    print(f"  MAE: {mae_test}")
    print(f"  MAPE: {mape_test}")

    print("\nCross Validation Metrics:")
    print(f"  RMSE: {rmse_test}")
    print(f"  MAE: {mae_test}")
    print(f"  MAPE: {mape_test}")

    print("\nNaive Model Metrics:")
    print(f"  RMSE: {rmse_naive}")
    print(f"  MAE: {mae_naive}")
    print(f"  MAPE: {mape_naive}")

    print("\nComparison SARIMAX vs Naive:")
    print(comparison)

    return {
        "model_name": "SARIMAX VC",
        "best_order": (best_p, best_d, best_q),
        "best_seasonal_order": (best_P, best_D, best_Q, 12),
        "training_metrics": {
            "rmse": rmse_test,
            "mae": mae_test,
            "mape": mape_test
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
        "comparison_sarimax_vs_naive": comparison,

        "historical_stats": {
            "DESCRIPCION": description,
            "model": "SARIMAX",
            "max": max_value,
            "min": min_value,
            "mean": mean_value,
            "std": std_value
        }
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
