from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import joblib
import os


class FeatureSelector:
    """Handles feature selection for LSTM models."""
    
    def __init__(self, feature_list=None):
        """
        Initialize the feature selector.
        
        Parameters:
        feature_list: list of feature names (if None, uses default)
        """
        self.default_features =  [
            "energy_interval_kWh", "energy_interval_kWh_rolling_3","Temperature (F)", "Humidity (%)", "is_weekend", "is_friday",
            "is_monday", "is_winter", "DJC", "DJF", "temp_rolling_3", "humid_rolling_3", "Temp_Squared",
            "lag_1", "lag_7","month_cos", "month_sin", "day_cos", "day_sin"]
        self.feature_list = feature_list if feature_list is not None else self.default_features
        self.selected_features = None
    
    def select(self, df, target_column = "energy_interval_kWh"):
        """
        Select features from DataFrame.
        
        Parameters:
        df: DataFrame containing the data
        
        Returns:
        X: feature array
        y: target array
        """
        self.selected_features = self.feature_list
        X = df[self.selected_features].values
        y = df[target_column].values
        
        return X, y
    
    def get_feature_names(self):
        """Return the list of selected feature names."""
        return self.selected_features


class SequenceCreator:
    """Creates sequences for LSTM models."""
    
    def __init__(self, window=14, horizon=7):
        """
        Initialize the sequence creator.
        
        Parameters:
        window: number of time steps to look back (default: 14 for 2 weeks)
        horizon: number of steps to predict ahead (default: 7)
        """
        self.window = window
        self.horizon = horizon
        self.X_sequences = None
        self.y_sequences = None
    
    def create(self, X, y):
        """
        Create sequences from feature and target arrays.
        
        Parameters:
        X: feature array
        y: target array
        
        Returns:
        X_sequences, y_sequences
        """
        X_sequences, y_sequences = [], []
        
        for i in range(len(X) - self.window - self.horizon):
            X_sequences.append(X[i: i + self.window])
            y_sequences.append(y[i + self.window: i + self.window + self.horizon])
        
        self.X_sequences = np.array(X_sequences)
        self.y_sequences = np.array(y_sequences)
        
        return self.X_sequences, self.y_sequences
    
    def get_parameters(self):
        """Return window and horizon parameters."""
        return {'window': self.window, 'horizon': self.horizon}


class DataSplitter:
    """Splits data into train, validation, and test sets."""
    
    def __init__(self, train_ratio=0.7, val_ratio=0.15):
        """
        Initialize the data splitter.
        
        Parameters:
        train_ratio: proportion for training
        val_ratio: proportion for validation
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        
        self.splits = {}
    
    def split(self, X, y):
        """
        Split sequences into train, validation, and test sets.
        
        Parameters:
        X: feature sequences
        y: target sequences
        
        Returns:
        Dictionary containing all splits
        """
        train_size = int(len(X) * self.train_ratio)
        val_size = int(len(X) * self.val_ratio)
        
        self.splits = {
            'X_train_raw': X[:train_size],
            'X_val_raw': X[train_size:train_size+val_size],
            'X_test_raw': X[train_size+val_size:],
            'y_train': y[:train_size],
            'y_val': y[train_size:train_size+val_size],
            'y_test': y[train_size+val_size:]
        }
        
        return self.splits
    
    def get_splits(self):
        """Return the splits dictionary."""
        return self.splits
    
    def get_ratios(self):
        """Return train/val/test ratios."""
        return {
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio
        }


class FeatureScaler:
    """Scales features for LSTM models (handles 3D -> 2D -> 3D reshaping)."""
    
    def __init__(self, scaler_type='minmax'):
        """
        Initialize the feature scaler.
        
        Parameters:
        scaler_type: 'minmax' or 'standard'
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.scaler = self._create_scaler()
        self.shapes = {}
    
    def _create_scaler(self):
        """Create the appropriate scaler."""
        if self.scaler_type == 'minmax':
            return MinMaxScaler()
        else:
            return StandardScaler()
    
    def fit_transform(self, X_train_raw, X_val_raw, X_test_raw):
        """
        Fit scaler on training data and transform all sets.
        
        Parameters:
        X_train_raw, X_val_raw, X_test_raw: raw feature sequences (3D)
        
        Returns:
        Scaled datasets
        """
        n_train, w_train, f_train = X_train_raw.shape
        n_val, w_val, f_val = X_val_raw.shape
        n_test, w_test, f_test = X_test_raw.shape
        
        self.shapes = {
            'train': (n_train, w_train, f_train),
            'val': (n_val, w_val, f_val),
            'test': (n_test, w_test, f_test)
        }
        
        X_train_scaled = self.scaler.fit_transform(X_train_raw.reshape(-1, f_train))
        X_train_scaled = X_train_scaled.reshape(n_train, w_train, f_train)
        
        X_val_scaled, X_test_scaled = X_val_raw, X_test_raw

        if X_val_raw is not None and X_val_raw.size > 0:
            n_val, w_val, f_val = X_val_raw.shape
            X_val_scaled = self.scaler.transform(X_val_raw.reshape(-1, f_val))
            X_val_scaled = X_val_scaled.reshape(n_val, w_val, f_val)

        if X_test_raw is not None and X_test_raw.size > 0:
            n_test, w_test, f_test = X_test_raw.shape
            X_test_scaled = self.scaler.transform(X_test_raw.reshape(-1, f_test))
            X_test_scaled = X_test_scaled.reshape(n_test, w_test, f_test)
            
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def transform(self, X_raw):
        """
        Transform new data using fitted scaler.
        
        Parameters:
        X_raw: raw feature sequences (3D)
        
        Returns:
        Scaled sequences
        """
        n_samples, w, f = X_raw.shape
        X_scaled = self.scaler.transform(X_raw.reshape(-1, f))
        return X_scaled.reshape(n_samples, w, f)
    
    def inverse_transform(self, X_scaled):
        """
        Inverse transform scaled features.
        
        Parameters:
        X_scaled: scaled feature sequences (3D)
        
        Returns:
        Original scale sequences
        """
        n_samples, w, f = X_scaled.shape
        X_raw = self.scaler.inverse_transform(X_scaled.reshape(-1, f))
        return X_raw.reshape(n_samples, w, f)
    
    def get_scaler(self):
        """Return the fitted scaler."""
        return self.scaler


class TargetScaler:
    """Scales target sequences for LSTM models."""
    
    def __init__(self, scaler_type='minmax'):
        """
        Initialize the target scaler.
        
        Parameters:
        scaler_type: 'minmax' or 'standard'
        """
        self.scaler_type = scaler_type
        self.scaler = self._create_scaler()
    
    def _create_scaler(self):
        """Create the appropriate scaler."""
        if self.scaler_type == 'minmax':
            return MinMaxScaler()
        else:
            return StandardScaler()
    
    def fit_transform(self, y_train, y_val, y_test):
        """
        Fit scaler on training targets and transform all sets.
        
        Parameters:
        y_train, y_val, y_test: target sequences
        
        Returns:
        Scaled targets
        """
        y_train_scaled = self.scaler.fit_transform(y_train)
        y_val_scaled, y_test_scaled = y_val, y_test
        if y_val.size>0:
          y_val_scaled = self.scaler.transform(y_val)
        if y_test.size>0:
          y_test_scaled = self.scaler.transform(y_test)
        
        return y_train_scaled, y_val_scaled, y_test_scaled
    
    def transform(self, y):
        """Transform new targets using fitted scaler."""
        return self.scaler.transform(y)
    
    def inverse_transform(self, y_scaled):
        """Inverse transform scaled targets."""
        return self.scaler.inverse_transform(y_scaled)
    
    def get_scaler(self):
        """Return the fitted scaler."""
        return self.scaler

class TargetScalerSeq2Seq:
    """Scales target sequences for LSTM models."""
    
    def __init__(self, horizon, scaler_type='minmax'):
        """
        Initialize the target scaler.
        
        Parameters:
        horizon : step of prediction
        scaler_type: 'minmax' or 'standard'
        """
        self.horizon = horizon
        self.scaler_type = scaler_type
        self.scaler = self._create_scaler()
    
    def _create_scaler(self):
        """Create the appropriate scaler."""
        if self.scaler_type == 'minmax':
            return MinMaxScaler()
        else:
            return StandardScaler()
    
    def fit_transform(self, y_train, y_val, y_test):
        """
        Fit scaler on training targets and transform all sets.
        
        Parameters:
        y_train, y_val, y_test: target sequences
        
        Returns:
        Scaled targets
        """
        y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1, self.horizon, 1)
        y_val_scaled = self.scaler.transform(y_val.reshape(-1, 1)).reshape(-1, self.horizon, 1)
        y_test_scaled = self.scaler.transform(y_test.reshape(-1, 1)).reshape(-1, self.horizon, 1)
        
        return y_train_scaled, y_val_scaled, y_test_scaled
    
    def transform(self, y):
        """Transform new targets using fitted scaler."""
        return self.scaler.transform(y)
    
    def inverse_transform(self, y_scaled):
        """Inverse transform scaled targets."""
        return self.scaler.inverse_transform(y_scaled.reshape(-1, 1))
    
    def get_scaler(self):
        """Return the fitted scaler."""
        return self.scaler


class LSTMDataPreparator:
    """Main class orchestrating LSTM data preparation."""
    
    def __init__(self, 
                 feature_list=None,
                 window=14, 
                 horizon=7, 
                 train_ratio=0.7, 
                 val_ratio=0.15,
                 feature_scaler_type='minmax',
                 target_scaler_type='minmax',
                 verbose=True,
                 target_column = "energy_interval_kWh"):
        """
        Initialize the LSTM data preparator.
        
        Parameters:
        feature_list: list of feature names
        window: number of time steps to look back
        horizon: number of steps to predict ahead
        train_ratio: proportion for training
        val_ratio: proportion for validation
        feature_scaler_type: scaler type for features
        target_scaler_type: scaler type for targets
        verbose: whether to print information
        """
        self.verbose = verbose
        
        self.feature_selector = FeatureSelector(feature_list)
        self.sequence_creator = SequenceCreator(window, horizon)
        self.data_splitter = DataSplitter(train_ratio, val_ratio)
        self.feature_scaler = FeatureScaler(feature_scaler_type)
        self.target_scaler = TargetScaler(target_scaler_type)
        self.target_column = target_column
        
        self.X_raw = None
        self.y_raw = None
        self.X_sequences = None
        self.y_sequences = None
        self.splits = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.y_train_raw = None
        self.y_val_raw = None
        self.y_test_raw = None
    
    def prepare(self, df):
        """
        Complete data preparation pipeline for LSTM.
        
        Parameters:
        df: DataFrame containing the data
        
        Returns:
        self (for method chaining)
        """
        
        if self.verbose:
            print("1. Selecting features...")
        self.X_raw, self.y_raw = self.feature_selector.select(df, self.target_column)
        
        if self.verbose:
            print("2. Creating LSTM sequences...")
        self.X_sequences, self.y_sequences = self.sequence_creator.create(self.X_raw, self.y_raw)
        
        if self.verbose:
            print("3. Splitting data...")
        self.splits = self.data_splitter.split(self.X_sequences, self.y_sequences)
        
        self.y_train_raw = self.splits['y_train']
        self.y_val_raw = self.splits['y_val']
        self.y_test_raw = self.splits['y_test']
        
        if self.verbose:
            print("4. Scaling features...")
        self.X_train, self.X_val, self.X_test = self.feature_scaler.fit_transform(
            self.splits['X_train_raw'],
            self.splits['X_val_raw'],
            self.splits['X_test_raw']
        )
        
        if self.verbose:
            print("5. Scaling targets...")
        self.y_train, self.y_val, self.y_test = self.target_scaler.fit_transform(
            self.splits['y_train'],
            self.splits['y_val'],
            self.splits['y_test']
        )
        
        return self
    
    def get_training_data(self):
        """Return training data."""
        return self.X_train, self.y_train
    
    def get_validation_data(self):
        """Return validation data."""
        return self.X_val, self.y_val
    
    def get_test_data(self):
        """Return test data."""
        return self.X_test, self.y_test
    
    def get_raw_data(self):
        """Return raw (unscaled) data."""
        return {
            'X_train_raw': self.splits['X_train_raw'],
            'X_val_raw': self.splits['X_val_raw'],
            'X_test_raw': self.splits['X_test_raw'],
            'y_train_raw': self.y_train_raw,
            'y_val_raw': self.y_val_raw,
            'y_test_raw': self.y_test_raw
        }
    
    def get_all_data(self):
        """Return all prepared data as a dictionary."""
        return {
            'X_train': self.X_train,
            'X_val': self.X_val,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_val': self.y_val,
            'y_test': self.y_test,
            'y_train_raw': self.y_train_raw,
            'y_val_raw': self.y_val_raw,
            'y_test_raw': self.y_test_raw,
            'feature_scaler': self.feature_scaler.get_scaler(),
            'target_scaler': self.target_scaler.get_scaler(),
            'features_used': self.feature_selector.get_feature_names(),
            'window': self.sequence_creator.window,
            'horizon': self.sequence_creator.horizon
        }
    
    def inverse_transform_targets(self, y_scaled):
        """
        Inverse transform scaled targets back to original scale.
        
        Parameters:
        y_scaled: scaled target values
        
        Returns:
        Original scale targets
        """
        return self.target_scaler.inverse_transform(y_scaled)
    
    def inverse_transform_features(self, X_scaled):
        """
        Inverse transform scaled features back to original scale.
        
        Parameters:
        X_scaled: scaled feature sequences (3D)
        
        Returns:
        Original scale feature sequences
        """
        return self.feature_scaler.inverse_transform(X_scaled)


    def save_preparator_artifacts(self, folder_path="models"):
        """
        Save scalers and features'list.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        joblib.dump(self.feature_scaler.get_scaler(), f"{folder_path}/feature_scaler.pkl")
        
        joblib.dump(self.target_scaler.get_scaler(), f"{folder_path}/target_scaler.pkl")
        
        joblib.dump(self.feature_selector.get_feature_names(), f"{folder_path}/feature_list.pkl")
        
        print(f"Artefacts saved at : {folder_path}")