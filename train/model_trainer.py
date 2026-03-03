import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

class ConsumptionModelTrainer:
    """Handles GRU model architecture construction and evaluation."""    
    def __init__(self):
        self.model = None
        self.history = None
    
    
    def build_model(self, input_shape, horizon):
        """
        Build the GRU model.
        
        Parameters:
        input_shape: (timesteps,features)
        horizon: number of steps to predict
        
        Returns:
        Built model
        """
        self.model = Sequential([
            Input(shape=input_shape),
            GRU(128, return_sequences=True),
            Dropout(0.3),
            GRU(64),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(horizon)
        ])
        
        return self
    
    def compile_model(self):
        """Compile the model."""

        if self.model is None:
            raise ValueError("Modèle non construit. Appelez build_model() d'abord.")
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mae','mse']
            )
        return self
    
    def show_summary(self):
        """Affiche le résumé du modèle"""
        if self.model is None:
            raise ValueError("Modèle non construit.")
        self.model.summary()
        return self
    
    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
    ):
        """Training model"""
        if self.model is None:
            raise ValueError("Modèle non compilé. Appelez compile_model() d'abord.")
        
        es = EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            verbose=1, 
            patience=15, 
            restore_best_weights=True
            )
        if X_val is None:
          self.history = self.model.fit(
              X_train, y_train,
              epochs=100,
              batch_size=32,
              callbacks=[es],
              verbose=1
          )
          return self.history
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[es],
            verbose=1
        )
        
        return self.history
    
    def get_model(self):
        """Return the built model."""        
        return self.model
    
    def get_history(self):
        """Get training"""
        return self.history
    

    def visualize_training(self):
        """Plot training history."""

        plt.plot(self.history.history['loss'], label='Train')
        plt.plot(self.history.history['val_loss'], label='Validation')
        plt.legend()
        plt.title('GRU Learning Curve')
        plt.show()


    def save_model(self, file_path="models/energy_model.keras"):
        """
        Save trained model.
        """
        if self.model is None:
            raise ValueError("No model founed. call build_model() and train() first.")
        
        folder = os.path.dirname(file_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            
        self.model.save(file_path)
        print(f"Successfully saved : {file_path}")
        return self