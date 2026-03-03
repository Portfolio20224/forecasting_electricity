class DataPreparator:
    """Handles data preparation and daily profile creation."""
    
    def __init__(self, resolution='h'):
        """
        Initialize the data preparator.
        
        Parameters:
        resolution: 'h' for hourly, '15min' for quarter-hour
        """
        self.resolution = resolution
    
    def prepare_daily_profiles(self, df):
        """
        Prepares daily profiles from raw data.
        
        Parameters:
        df: DataFrame with columns 'date_time' and 'Electricity consumption (kW)'
        
        Returns:
        daily_profiles: DataFrame with index = dates, columns = hours/periods
        """
        df = df.copy()
        
        if self.resolution == 'h':
            df_hourly = df.resample('h').sum()
            df_hourly['hour'] = df_hourly.index.hour
            df_hourly['date'] = df_hourly.index.date
            
            daily_profiles = df_hourly.pivot_table(
                index='date',
                columns='hour',
                values='Electricity consumption (kW)'
            )
        else:  
            df['time'] = df.index.time
            df['date'] = df.index.date
            
            daily_profiles = df.pivot_table(
                index='date',
                columns='time',
                values='Electricity consumption (kW)'
            )
        
        daily_profiles = daily_profiles.dropna(thresh=len(daily_profiles.columns)*0.8)
        
        return daily_profiles