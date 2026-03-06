import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from data_processor import DataPreprocessor

class EnergyForecaster:
    """Main class for energy consumption forecasting."""
    def __init__(self, model_path, scaler_x_path, scaler_y_path, feature_columns_path):
        self.model = load_model(model_path)
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        self.feature_columns   = joblib.load(feature_columns_path)

        self.horizon = 7

    def _prepare_features_new_data(self, elec_consum_path, weather_data_path):
        """
        Prepare features from new data.
        
        Parameters:
        elec_consum_path: path to electricity consumption data
        weather_data_path: path to weather data
        
        Returns:
        DataFrame with engineered features
        """

        df_processed = DataPreprocessor(elec_consum_path, weather_data_path).feature_engineering_consumption()

        return df_processed

    def predict_range(self, start_date, end_date, elec_consum_path, weather_data_path):
        """
        Generate forecasts for a date range.
        
        Parameters:
        start_date: start date for predictions (YYYY-MM-DD)
        end_date: end date for predictions (YYYY-MM-DD)
        elec_consum_path: path to electricity consumption data
        weather_data_path: path to weather data
        
        Returns:
        DataFrame with predictions for all dates in range
        """

        predictions = []
        current_date = pd.to_datetime(start_date)
        final_date = pd.to_datetime(end_date)
        full_data = self._prepare_features_new_data( elec_consum_path, weather_data_path)

        data_max_date = full_data.index.max()

        if current_date > data_max_date:
            raise ValueError(
                f"Start date {current_date.date()} is after last available data "
                f"({data_max_date.date()})"
            )
        FEATURES = self.feature_columns

        while current_date <= final_date:
            input_data = (
                            full_data.loc[:current_date]
                            .tail(14)[FEATURES]
                        )
            X_3d = self.scaler_x.transform(input_data)

            X_3d = np.expand_dims(X_3d, axis=0)

            
            pred_scaled = self.model.predict(X_3d, verbose=0)
            
            pred_kw = self.scaler_y.inverse_transform(pred_scaled)
            
            future_dates = pd.date_range(start=current_date, periods=self.horizon, freq='1D')
            batch_df = pd.DataFrame({'datetime': future_dates, 'forecast_kW': pred_kw.flatten()})
            predictions.append(batch_df)
            
            current_date += pd.Timedelta(days=7)

        return pd.concat(predictions).drop_duplicates(subset='datetime')