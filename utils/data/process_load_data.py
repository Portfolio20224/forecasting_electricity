import numpy as np
import pandas as pd
from .load_data import load_and_prepare_data


def create_reference_profile(df_valid, threshold_bas=205):
    """
    Create reference profile from valid data above threshold.
    """
    df_clean_ref = df_valid[df_valid['Electricity consumption (kW)'] > threshold_bas]
    reference_profile = df_clean_ref.groupby([
        df_clean_ref.index.dayofweek,
        df_clean_ref.index.hour,
        df_clean_ref.index.minute
    ])['Electricity consumption (kW)'].mean()
    
    return reference_profile

def fill_low_consumption_values(load_corrected, threshold_bas=205):
    """
    Replace values below threshold with historical means.
    """
    df_valid = load_corrected.copy()
    reference_profile = create_reference_profile(df_valid, threshold_bas)
    
    def get_historical_mean(timestamp):
        return reference_profile.loc[(timestamp.dayofweek, timestamp.hour, timestamp.minute)]
    
    load_corrected.loc[load_corrected['Electricity consumption (kW)'] < threshold_bas, 'Electricity consumption (kW)'] = np.nan
    
    nan_mask = load_corrected['Electricity consumption (kW)'].isna()
    load_corrected.loc[nan_mask, 'Electricity consumption (kW)'] = load_corrected.index[nan_mask].map(get_historical_mean)
    
    return load_corrected

def handle_weekend_outliers(load_corrected, weekend_threshold=350):
    """
    Cap weekend consumption values.
    """
    is_weekend_mask = load_corrected.index.dayofweek >= 5
    load_corrected.loc[is_weekend_mask, 'Electricity consumption (kW)'] = load_corrected.loc[
        is_weekend_mask, 'Electricity consumption (kW)'
    ].clip(upper=weekend_threshold)
    
    return load_corrected

def handle_weekday_outliers(load_corrected):
    """
    Apply Winsorizing (capping) for each weekday separately.
    """
    for i in range(5):
        mask = load_corrected.index.dayofweek.isin([i])
        
        threshold = load_corrected.loc[mask, 'Electricity consumption (kW)'].quantile(0.99)
        
        load_corrected.loc[mask, 'Electricity consumption (kW)'] = \
            load_corrected.loc[mask, 'Electricity consumption (kW)'].clip(upper=threshold)
        
        print(f"Threshold applied for day {i}: {threshold:.2f} kW")
    
    return load_corrected

def calculate_daily_energy(load_corrected):
    """
    Calculate energy per interval and resample to daily total.
    """
    load_corrected["energy_interval_kWh"] = load_corrected["Electricity consumption (kW)"] * 0.25
    
    df_daily_load = load_corrected["energy_interval_kWh"].resample('D').sum().to_frame()
    
    return df_daily_load, load_corrected

def process_electricity_data(filepath, threshold_bas=205, weekend_threshold=350):
    """
    Main function to process electricity consumption data.
    """
    print("Loading and preparing data...")
    load_timed = load_and_prepare_data(filepath)
    
    print(f"Filling values below {threshold_bas} kW with historical means...")
    load_corrected = fill_low_consumption_values(load_timed, threshold_bas)
    
    print("Handling weekend outliers...")
    load_corrected = handle_weekend_outliers(load_corrected, weekend_threshold)
    
    print("Handling weekday outliers...")
    load_corrected = handle_weekday_outliers(load_corrected)
    
    print("Calculating daily energy consumption...")
    df_daily_load, load_corrected = calculate_daily_energy(load_corrected)
    
    print("Processing complete!")
    
    return load_corrected, df_daily_load