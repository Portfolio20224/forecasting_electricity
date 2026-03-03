import numpy as np
from utils import process_electricity_data, process_weather_data


class DataPreprocessor:
    """Handles data preprocessing and feature engineering for forecasting."""
    def __init__(self, elec_consum_path, weather_data_path):
        """
        Initialize the data preprocessor.
        
        Parameters:
        elec_consum_path: path to electricity consumption data
        weather_data_path: path to weather data
        """
        self.file_elec_consum = elec_consum_path
        self.file_weather = weather_data_path
        self.df_consumption = None
        self.df_demand = None
        self._load_dataset()

    def _load_dataset(self):
        """
        Merge load and weather data.
        """

        load_corrected, df_daily_load = process_electricity_data(
            filepath=self.file_elec_consum,
        )

        weather_corrected, weather_15min = process_weather_data(
            filepath=self.file_weather,
        )
        self.df_consumption = df_daily_load.join(weather_corrected, how="inner")
        self.df_demand = load_corrected.join(weather_15min, how="inner")
        return self

    
    def feature_engineering_consumption(self):
        """
        Perform feature engineering on consumption data.
        
        Returns:
        Processed DataFrame with engineered features
        """

        df_consumption  = self.df_consumption

        df_consumption['temp_rolling_3'] = df_consumption['Temperature (F)'].shift(1).rolling(window=3).mean()
        df_consumption['humid_rolling_3'] = df_consumption['Humidity (%)'].shift(1).rolling(window=3).mean()
        df_consumption['Temp_Squared'] = df_consumption['Temperature (F)'] ** 2


        df_consumption["dayofweek"] = df_consumption.index.dayofweek
        df_consumption["month"] = df_consumption.index.month

        df_consumption["is_weekend"] = df_consumption["dayofweek"].isin([5,6]).astype(int)
        df_consumption['is_friday'] = (df_consumption.index.weekday == 4).astype(int)
        df_consumption['is_monday'] = (df_consumption.index.weekday == 0).astype(int)

        df_consumption["is_summer"] = df_consumption["month"].isin([12,1,2]).astype(int)
        df_consumption["is_winter"] = df_consumption["month"].isin([6,7,8]).astype(int)

        df_consumption["month_sin"] = np.sin(2 * np.pi * df_consumption["month"] / 12)
        df_consumption["month_cos"] = np.cos(2 * np.pi * df_consumption["month"] / 12)


        df_consumption["lag_1"] = df_consumption["energy_interval_kWh"].shift(1)
        df_consumption["lag_7"] = df_consumption["energy_interval_kWh"].shift(7)

        df_consumption['energy_interval_kWh_rolling_3'] = df_consumption['energy_interval_kWh'].shift(1).rolling(window=3).mean()

        df_consumption["day_sin"] = np.sin(2 * np.pi * df_consumption.index.dayofweek / 7)
        df_consumption["day_cos"] = np.cos(2 * np.pi * df_consumption.index.dayofweek / 7)

        base_temp_c = 45
        base_temp_f = 65

        df_consumption['DJC'] = df_consumption['Temperature (F)'].apply(lambda x: max(0, base_temp_c - x))
        df_consumption['DJF'] = df_consumption['Temperature (F)'].apply(lambda x: max(0, x - base_temp_f))

        return df_consumption.dropna()

    
    def feature_engineering_demand(self):
        """
        Perform feature engineering on demand data.
        
        Returns:
        Processed DataFrame with engineered features
        """

        df  = self.df_demand

        df['energy_kW_rolling_4'] = df['Electricity consumption (kW)'].shift(1).rolling(window=4).mean()


        df['temp_rolling_12'] = df['Temperature (F)'].shift(1).rolling(window=12).mean()
        df['humid_rolling_12'] = df['Humidity (%)'].shift(1).rolling(window=12).mean()
        df['Temp_Squared'] = df['Temperature (F)'] ** 2

        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["month"] = df.index.month

        df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
        df['is_friday'] = (df.index.weekday == 4).astype(int)
        df['is_monday'] = (df.index.weekday == 0).astype(int)

        df["lag_1"] = df["Electricity consumption (kW)"].shift(1)
        df["lag_2"] = df["Electricity consumption (kW)"].shift(2)
        df["lag_96"] = df["Electricity consumption (kW)"].shift(96)

        df["is_summer"] = df["month"].isin([12,1,2]).astype(int)
        df["is_winter"] = df["month"].isin([6,7,8]).astype(int)

        df["is_occuped"] = df["hour"].between(8, 19).astype(int)

        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)


        df["HDD"] = np.where(df["Temperature (F)"] < 55, 
                            55 - df["Temperature (F)"], 0)
        df["HDD_occupied"] = df["HDD"] * df["is_occuped"]
        df["CDD"] = np.where(df["Temperature (F)"] > 65, 
                            df["Temperature (F)"] - 65, 0)
        return df.dropna()

