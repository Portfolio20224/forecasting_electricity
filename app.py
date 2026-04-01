from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from data_processor import DataPreprocessor

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "notebooks" / "models"

app = FastAPI(title="Energy Forecasting API")

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
    data_dir = BASE_DIR / "data"
    elec_path    = data_dir / "Electricity consumption.csv"
    weather_path = data_dir / "Weather data.csv"

    if not elec_path.exists() or not weather_path.exists():
        raise HTTPException(status_code=500, detail="Fichiers de données introuvables")

    try:
        full_data = DataPreprocessor(elec_path, weather_path).feature_engineering_consumption()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur preprocessing : {e}")

    predictions = []
    current = pd.to_datetime(req.start_date)
    end     = pd.to_datetime(req.end_date)
    data_max = full_data.index.max()

    if current > data_max:
        raise HTTPException(
            status_code=400,
            detail=f"start_date {req.start_date} dépasse la dernière donnée disponible ({data_max.date()})"
        )

    while current <= end:
        X = full_data.loc[:current].tail(14)[features]
        X_scaled = scaler_x.transform(X)
        X_3d     = np.expand_dims(X_scaled, axis=0)

        pred_scaled = model.predict(X_3d, verbose=0)
        pred_kw     = scaler_y.inverse_transform(pred_scaled).flatten()

        future_dates = pd.date_range(start=current, periods=7, freq="1D")
        for date, val in zip(future_dates, pred_kw):
            predictions.append({"datetime": str(date.date()), "forecast_kW": round(float(val), 2)})

        current += pd.Timedelta(days=7)

    seen = set()
    unique = []
    for p in predictions:
        if p["datetime"] not in seen:
            seen.add(p["datetime"])
            unique.append(p)

    return ForecastResponse(predictions=unique)

@app.get("/health")
def health():
    return {"status": "ok", "model": "energy_model.keras"}