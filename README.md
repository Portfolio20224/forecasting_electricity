# Forecasting Electric

Electricity consumption forecasting project at different time scales (daily and 15-minute intervals).

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Models](#models)

## 🔍 Overview

This project provides tools for:
- Daily electricity consumption forecasting (kWh) for 7 days ahead
- 15-minute electricity demand forecasting (kW) for 2 days ahead
- Training and inference pipelines for SARIMA and Deep Learning models

## 📁 Project Structure

```
FORECASTING_ELECTRIC/
│
├── .venv/                          # Virtual environment
├── data/                           # Raw and processed data
│
├── data_processor/                  # Data processing
│   ├── __init__.py
│   ├── builder.py                   # Feature engineering
│   └── processor.py                  # Preprocessing
│
├── forecasting/                      # Prediction module
│   ├── __init__.py
│   └── forecaster.py                  # Main forecasting class
│
├── notebooks/                         # Experimentation notebooks
│   ├── models/                         # Saved models
│   ├── 15min_model.ipynb
│   ├── baseline_sarima_consumption.ipynb
│   ├── demand_model.ipynb
│   └── question_answering.ipynb
│
├── train/                              # Training scripts
│   ├── __init__.py
│   ├── demand_model_trainer.py          # 15min model training
│   ├── model_trainer.py                  # Daily model training
│   └── pipe_line.py                       # Complete pipeline
│
├── utils/                               # Utilities
├── main.py                              # CLI entry point
├── poetry.lock
├── pyproject.toml                        # Poetry dependencies
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
## 🏗️ Code Structure

- **data_processor/** : Preprocessing and feature engineering
- **forecasting/** : Inference logic and `EnergyForecaster` class
- **train/** : Model training scripts
- **notebooks/** : Exploratory analysis and prototypes
- **main.py** : CLI  for predictions

## 🤖 Models

| Model | Resolution | Horizon | Architecture |
|--------|------------|---------|--------------|
| Daily | 1 day | 7 days |   LSTM |

