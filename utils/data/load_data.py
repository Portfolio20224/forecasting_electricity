import pandas as pd

def load_and_prepare_data(filepath):
    """
    Load and prepare the time data.
    """
    df = pd.read_csv(filepath)    
    df["date_time"] = pd.to_datetime(df["date_time"])
    df_timed = df.set_index("date_time").sort_index()
    
    return df_timed