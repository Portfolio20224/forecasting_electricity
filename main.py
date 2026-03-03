import argparse
from forecasting import EnergyForecaster
from pathlib import Path

def main():
    base_dir = Path(__file__).parent 
    parser = argparse.ArgumentParser(description="CLI de Prévision de energie_kwh (1-D)")
    parser.add_argument("--start", type=str, required=True, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="forecast_results.csv", help="Nom du fichier CSV de sortie")
    
    args = parser.parse_args()
    data_dir = base_dir / "data"      # data/ dans le même dossier
    models_dir = base_dir / "notebooks" / "models"  # notebooks/models
    forecaster = EnergyForecaster(models_dir/"energy_model.keras", models_dir/"feature_scaler.pkl", models_dir/"target_scaler.pkl", models_dir/"feature_list.pkl")

    elec_consum_path = data_dir/"Electricity consumption.csv"
    weather_data_path = data_dir/"Weather data.csv"

    print(f"Génération des prévisions du {args.start} au {args.end}...")
    results = forecaster.predict_range(args.start, args.end, elec_consum_path, weather_data_path)

    results.to_csv(args.output, index=False)
    print(f"Succès ! Fichier enregistré sous : {args.output}")

if __name__ == "__main__":
    main()