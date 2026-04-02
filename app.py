from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import time
import logging
from tensorflow.keras.models import load_model
from data_processor import DataPreprocessor
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, Gauge
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "notebooks" / "models"

REQUEST_COUNT = Counter(
    "forecast_requests_total",
    "Nombre total de requêtes",
    ["method", "status"]
)
REQUEST_LATENCY = Histogram(
    "forecast_request_duration_seconds",
    "Durée des requêtes en secondes"
)
FORECAST_MEAN = Gauge(
    "forecast_mean_kw",
    "Moyenne des prédictions en kW"
)
FORECAST_MIN = Gauge(
    "forecast_min_kw",
    "Valeur minimale prédite en kW"
)
FORECAST_MAX = Gauge(
    "forecast_max_kw",
    "Valeur maximale prédite en kW"
)

app = FastAPI(title="Energy Forecasting API")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


model      = load_model(MODELS_DIR / "energy_model.keras")
scaler_x   = joblib.load(MODELS_DIR / "feature_scaler.pkl")
scaler_y   = joblib.load(MODELS_DIR / "target_scaler.pkl")
features   = joblib.load(MODELS_DIR / "feature_list.pkl")

class ForecastRequest(BaseModel):
    start_date: str
    end_date: str

class ForecastResponse(BaseModel):
    predictions: list[dict]

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    start_time = time.time()
    logger.info(f"Requête reçue : {req.start_date} → {req.end_date}")

    data_dir     = BASE_DIR / "data"
    elec_path    = data_dir / "Electricity consumption.csv"
    weather_path = data_dir / "Weather data.csv"

    if not elec_path.exists() or not weather_path.exists():
        logger.error("Fichiers de données introuvables")
        raise HTTPException(status_code=500, detail="Fichiers de données introuvables")

    try:
        full_data = DataPreprocessor(elec_path, weather_path).feature_engineering_consumption()
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", status="500").inc()
        logger.error(f"Erreur preprocessing : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur preprocessing : {e}")

    predictions = []
    current  = pd.to_datetime(req.start_date)
    end      = pd.to_datetime(req.end_date)
    data_max = full_data.index.max()

    if current > data_max:
        raise HTTPException(
            status_code=400,
            detail=f"start_date {req.start_date} dépasse la dernière donnée disponible ({data_max.date()})"
        )

    while current <= end:
        X        = full_data.loc[:current].tail(14)[features]
        X_scaled = scaler_x.transform(X)
        X_3d     = np.expand_dims(X_scaled, axis=0)

        pred_scaled = model.predict(X_3d, verbose=0)
        pred_kw     = scaler_y.inverse_transform(pred_scaled).flatten()

        future_dates = pd.date_range(start=current, periods=7, freq="1D")
        for date, val in zip(future_dates, pred_kw):
            predictions.append({"datetime": str(date.date()), "forecast_kW": round(float(val), 2)})

        current += pd.Timedelta(days=7)

    seen, unique = set(), []
    for p in predictions:
        if p["datetime"] not in seen:
            seen.add(p["datetime"])
            unique.append(p)

    duration = time.time() - start_time
    logger.info(f"Prédiction terminée en {duration:.2f}s · {len(unique)} jours générés")
    REQUEST_COUNT.labels(method="POST", status="200").inc()
    REQUEST_LATENCY.observe(duration)

    values = [p["forecast_kW"] for p in unique]
    FORECAST_MEAN.set(round(sum(values) / len(values), 2))
    FORECAST_MIN.set(min(values))
    FORECAST_MAX.set(max(values))

    logger.info(f"Stats prédictions → mean: {FORECAST_MEAN._value.get():.1f} kW, "
                f"min: {FORECAST_MIN._value.get():.1f} kW, "
                f"max: {FORECAST_MAX._value.get():.1f} kW")

    return ForecastResponse(predictions=unique)

@app.get("/health")
def health():
    return {"status": "ok", "model": "energy_model.keras"}