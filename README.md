# Forecasting Electric

Electricity consumption forecasting project with a production-ready MLOps pipeline.

## 💡 Business Context

### Why this project?

Electricity is a non-storable resource. Grid operators, energy suppliers, and large industrial consumers must anticipate demand in advance to avoid two costly situations: over-production (wasted energy, financial losses) and under-production (grid instability, penalties, supply cuts).

Traditional forecasting methods (SARIMA, expert rules) struggle to capture complex patterns such as weather effects, public holidays, or long-term consumption trends. This project addresses that gap with a deep learning approach combined with a production-ready deployment pipeline.

### For whom?

| Persona | Use case |
|---------|----------|
| **Grid operators** | Plan production capacity 7 days ahead |
| **Energy suppliers** | Optimize procurement on electricity markets |
| **Industrial consumers** | Schedule high-consumption processes during off-peak hours |
| **Smart building managers** | Automate HVAC and equipment based on predicted demand |

### Business impact

- **Cost reduction** — a 1% improvement in forecast accuracy can represent hundreds of thousands of euros in avoided over/under-production costs at scale
- **Operational efficiency** — automated 7-day forecasts replace manual processes and free up analyst time
- **Scalability** — the REST API allows any internal tool or third-party system to consume forecasts programmatically, without data science expertise

## 📋 Table of Contents

- [Business Context](#business-context)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API](#api)
- [Deployment](#deployment)
- [CI/CD](#cicd)
- [Monitoring](#monitoring)
- [Code Structure](#code-structure)
- [Models](#models)

## 🔍 Overview

This project provides tools for:

- Daily electricity consumption forecasting (kWh) for 7 days ahead
- 15-minute electricity demand forecasting (kW) for 2 days ahead
- Training and inference pipelines for SARIMA and Deep Learning models
- A production REST API deployed on Google Cloud Run
- Automated CI/CD pipeline via GitHub Actions
- Application and model monitoring with Prometheus

## 📁 Project Structure

```
FORECASTING_ELECTRIC/
│
├── data/                           # Raw data
│   ├── Electricity consumption.csv
│   └── Weather data.csv
│
├── data_processor/                 # Data processing
│   ├── __init__.py
│   ├── builder.py                  # Feature engineering
│   └── processor.py               # Preprocessing
│
├── utils/                          # Shared utilities
│
├── forecasting/                    # Prediction module
│   ├── __init__.py
│   └── forecaster.py              # EnergyForecaster class
│
├── train/                          # Training scripts
│   ├── __init__.py
│   ├── demand_model_trainer.py    # 15min model training
│   ├── model_trainer.py           # Daily model training
│   └── pipe_line.py               # Complete pipeline
│
├── notebooks/                      # Experimentation notebooks
│   ├── models/                    # Saved models
│   │   ├── energy_model.keras
│   │   ├── feature_scaler.pkl
│   │   ├── target_scaler.pkl
│   │   └── feature_list.pkl
│   ├── 15min_model.ipynb
│   ├── baseline_sarima_consumption.ipynb
│   ├── demand_model.ipynb
│   └── question_answering.ipynb
│
├── reports/                        # Analysis reports
│   └── Energy_Forecasting_Report.pdf
│
├── .github/
│   └── workflows/
│       └── deploy.yml             # GitHub Actions CI/CD pipeline
│
├── app.py                         # FastAPI application
├── main.py                        # CLI entry point
├── Dockerfile
├── .dockerignore
├── poetry.lock
├── pyproject.toml
└── README.md
```

## 🚀 Installation

```bash
# Clone the repository
git clone git@github.com:Portfolio20224/forecasting_electricity.git
cd forecasting_electricity

# Install with Poetry
poetry install
```

## 💻 Usage

### Command Line Interface

```bash
poetry run python main.py --start 2024-01-01 --end 2024-01-07 --output forecasts.csv
```

### Run the API locally

```bash
poetry run uvicorn app:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker build -t energy-api .
docker run -p 8000:8000 energy-api
```

## 🌐 API

The REST API is built with FastAPI and exposes the following endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/forecast` | Generate forecasts for a date range |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Interactive API documentation (Swagger) |

### Example request

```bash
curl -X POST https://energy-api-828996169795.europe-west1.run.app/forecast \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2019-01-01", "end_date": "2019-01-14"}'
```

### Example response

```json
{
  "predictions": [
    {"datetime": "2019-01-01", "forecast_kW": 8728.31},
    {"datetime": "2019-01-02", "forecast_kW": 8941.81},
    ...
  ]
}
```

## ☁️ Deployment

The API is deployed on **Google Cloud Run** (Europe West 1 — Belgium):

🔗 **Production URL**: https://energy-api-828996169795.europe-west1.run.app

### Infrastructure

- **Container registry**: Google Artifact Registry (`europe-west1-docker.pkg.dev`)
- **Runtime**: Cloud Run (managed, serverless)
- **Memory**: 2 GiB per instance
- **Region**: `europe-west1`

### Manual deployment

```bash
# Build and tag
docker build -t energy-api .
docker tag energy-api europe-west1-docker.pkg.dev/energy-forecasting-492012/energy-repo/energy-api:v1

# Push to registry
docker push europe-west1-docker.pkg.dev/energy-forecasting-492012/energy-repo/energy-api:v1

# Deploy to Cloud Run
gcloud run deploy energy-api \
  --image europe-west1-docker.pkg.dev/energy-forecasting-492012/energy-repo/energy-api:v1 \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi
```

## 🔄 CI/CD

Every push to the `main` branch automatically triggers the GitHub Actions pipeline:

```
git push → Build Docker image → Push to Artifact Registry → Deploy to Cloud Run
```

The pipeline is defined in `.github/workflows/deploy.yml`. Each deployment is tagged with the Git commit SHA for full traceability and easy rollback.

**Required GitHub secrets:**

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GCP_REGION` | Deployment region |
| `GCP_SA_KEY` | Service account JSON key |

## 📊 Monitoring

The API exposes Prometheus metrics at `/metrics`:

### Application metrics

| Metric | Type | Description |
|--------|------|-------------|
| `forecast_requests_total` | Counter | Total requests by method and status |
| `forecast_request_duration_seconds` | Histogram | Request latency distribution |

### Model metrics

| Metric | Type | Description |
|--------|------|-------------|
| `forecast_mean_kw` | Gauge | Mean predicted consumption (kW) |
| `forecast_min_kw` | Gauge | Minimum predicted value (kW) |
| `forecast_max_kw` | Gauge | Maximum predicted value (kW) |

All requests are also logged in **GCP Cloud Logging** with structured logs including latency and prediction statistics.

## 🏗️ Code Structure

- **`app.py`** : FastAPI application — inference, monitoring, API endpoints
- **`data_processor/`** : Preprocessing and feature engineering
- **`forecasting/`** : Inference logic and `EnergyForecaster` class
- **`train/`** : Model training scripts
- **`notebooks/`** : Exploratory analysis and prototypes
- **`main.py`** : CLI for predictions

## 🤖 Models

| Model | Resolution | Horizon | Architecture |
|-------|------------|---------|--------------|
| Daily | 1 day | 7 days | LSTM |