import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

class DemandModelTrainer:
    """Handles Seq2Seq model architecture construction and evaluation."""    
    def __init__(self):
        self.model = None
        self.history = None
    
    
    def build_model(self, input_shape, horizon):
        """
        Build the Seq2seq model.
        
        Parameters:
        input_shape: (timesteps,features)
        horizon: number of steps to predict
        
        Returns:
        Built model
        """
        self.model = Sequential([

            GRU(128, activation='tanh', input_shape=input_shape, return_sequences=True),

            Dropout(0.2),

            GRU(64, activation='tanh', return_sequences=False),
            
            RepeatVector(horizon),
            
            GRU(64, activation='tanh', return_sequences=True),

            Dropout(0.2),
            
            TimeDistributed(Dense(1))
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


    def save_model(self, file_path="models/demand_model.keras"):
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