import numpy as np
from scipy import stats
import pandas as pd
from .load_data import load_and_prepare_data




def fill_outliers_consumption_values(weather, z_threshold=4):
    """
    Replace values below threshold with last value.
    """

    weather[['Temperature (F)', 'Humidity (%)']] = weather[['Temperature (F)', 'Humidity (%)']].replace(0, np.nan)
    z_scores = np.abs(stats.zscore(weather['Temperature (F)'], nan_policy='omit'))

    weather.loc[z_scores > z_threshold, 'Temperature (F)'] = np.nan

    weather['Temperature (F)'] = weather['Temperature (F)'].ffill()
    weather['Humidity (%)'] = weather['Humidity (%)'].ffill()
    weather_15min = weather.resample("15min").ffill()

    return weather, weather_15min
    

def process_weather_data(filepath, z_threshold=4):
    """
    Main function to process weather data.
    """
    print("Loading and preparing data...")
    weather_timed = load_and_prepare_data(filepath)
    
    print(f"Filling values below {z_threshold} with historical ...")
    weather_corrected, weather_15min = fill_outliers_consumption_values(weather_timed, z_threshold)
    
    print("Processing complete!")
    
    return weather_corrected, weather_15min