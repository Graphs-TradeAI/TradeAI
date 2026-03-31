import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from MLmodels.Forex.data_layer.client import TwelveDataClient
from MLmodels.Forex.Data.processing import build_forex_feature_set
from MLmodels.Forex.models.registry import ModelRegistry
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib


class ForexTrainer:
    def __init__(self, base_path="/home/job/Desktop/projects/TradeAI/MLmodels/Forex"):
        load_dotenv()
        self.api_key = os.getenv("TWELVE_DATA_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.base_path = base_path
        self.client = TwelveDataClient(self.api_key)
        self.scaler = MinMaxScaler()
        self.registry = ModelRegistry()

	# ...existing code...

    def prepare_sequences(self, df, seq_length=60, feature_cols=None):
        if feature_cols is None:
            # Exclude timestamp and targets
            feature_cols = [c for c in df.columns if c not in ["timestamp", "target_price"]]
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_cols])
        X, y = [], []
        for i in range(len(df) - seq_length):
            X.append(scaled_data[i:i+seq_length])
            y.append(df["target_price"].iloc[i+seq_length])
        
        return np.array(X), np.array(y)

    def walk_forward_split(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """
        Chronological splitting for time-series data.
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def build_sequential_model(self, input_shape):
        """
        Builds a Sequential LSTM model for price regression.
        """
        model = Sequential([
            Input(shape=input_shape),
            LSTM(256, return_sequences=True),
            Dropout(0.3),
            LSTM(128, return_sequences=True),
            Dropout(0.25),
            LSTM(64),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="linear")
        ], name="regression_lstm")
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )
        return model

    def train_model_for_pair(self, symbol="EUR/USD", timeframe="1h", epochs=50, batch_size=32):
        """
        Train a specialized LSTM model for a specific currency pair and timeframe.
        """
        print(f"Training specialized model for {symbol} {timeframe}...")
        # Fetch data for the specific pair
        df = self.client.get_forex_history(symbol=symbol, interval=timeframe, output_size=5000)
        df_features = build_forex_feature_set(df, symbol=symbol, timeframe=timeframe)
        # Prepare sequences
        X, y = self.prepare_sequences(df_features)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.walk_forward_split(X, y)
        # Ensure y is shape (n, 1) for binary classification
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        # Build and train model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_sequential_model(input_shape)
        model.summary()
        
        # Save model and scaler using ModelRegistry
        model_path = self.registry._pair_model_path(symbol, timeframe)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=7, 
            restore_best_weights=True,
            verbose=1
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stop]
        )
        scaler_path = self.registry._pair_scaler_path(symbol, timeframe)
        joblib.dump(self.scaler, scaler_path)
        # Evaluate
        loss, mae = model.evaluate(X_test, y_test)
        print(f"Test MAE for {symbol} {timeframe}: {mae:.4f}")
        return model, history

if __name__ == "__main__":
    # Example usage for training
    trainer = ForexTrainer()
    # Only train specialized models for specific pairs
    pair = "EUR/USD"
    timeframe = "1h"
    try:
        trainer.train_model_for_pair(symbol=pair, timeframe=timeframe, epochs=30)
    except Exception as e:
        print(f"Failed to train {pair} {timeframe}: {e}")
