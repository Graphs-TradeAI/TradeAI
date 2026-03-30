import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from MLmodels.Forex.data_layer.client import TwelveDataClient
from Data.processing import build_forex_feature_set
from forex_models.architectures import GlobalLSTMModel
from MLmodels.Forex.models.registry import ModelRegistry
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib

class ForexTrainer:
    def __init__(self, api_key=None, base_path="/home/job/Desktop/projects/TradeAI/MLmodels/Forex"):
        self.api_key = api_key
        self.base_path = base_path
        self.client = TwelveDataClient(api_key)
        self.scaler = MinMaxScaler()
        self.registry = ModelRegistry()
    def __init__(self, api_key=None, base_path="/home/job/Desktop/projects/TradeAI/MLmodels/Forex"):
        self.api_key = api_key
        self.base_path = base_path
        self.client = TwelveDataClient(api_key)
        self.scaler = MinMaxScaler()

    def load_multi_pair_data(self, pairs=["EUR/USD", "GBP/USD", "USD/JPY"], timeframes=["15min", "30min", "1h"]):
        """
        Global Dataset Builder: Combines multiple pairs and timeframes using Twelve Data API.
        """
        all_data = []
        for pair in pairs:
            for tf_str in timeframes:
                print(f"Fetching {pair} at {tf_str}...")
                # Twelve Data uses format like "EUR/USD"
                try:
                    df = self.client.get_forex_history(symbol=pair, interval=tf_str, output_size=5000)
                    df_features = build_forex_feature_set(df, symbol=pair, timeframe=tf_str)
                    all_data.append(df_features)
                except Exception as e:
                    print(f"Error fetching {pair} {tf_str}: {e}")
        
        return pd.concat(all_data, ignore_index=True).sort_values("timestamp")

    def prepare_sequences(self, df, seq_length=60, feature_cols=None):
        if feature_cols is None:
            # Exclude timestamp and targets
            feature_cols = [c for c in df.columns if c not in ["timestamp", "target_direction", "target_return_5"]]
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_cols])
        X, y = [], []
        for i in range(len(df) - seq_length):
            X.append(scaled_data[i:i+seq_length])
            y.append(df["target_direction"].iloc[i+seq_length])
        
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

    def train_global_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = GlobalLSTMModel.build(input_shape)
        
        # Callbacks
        save_dir = os.path.join(self.base_path, "global_checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            os.path.join(save_dir, "global_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stop]
        )
        
        # Save absolute final model and scaler using ModelRegistry
        global_model_path = self.registry._pair_model_path("global", "base")
        os.makedirs(os.path.dirname(global_model_path), exist_ok=True)
        model.save(global_model_path)
        global_scaler_path = self.registry._pair_scaler_path("global", "base")
        joblib.dump(self.scaler, global_scaler_path)
        
        return model, history

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
        
        # Build and train model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = GlobalLSTMModel.build(input_shape)
        
        # Save model and scaler using ModelRegistry
        model_path = self.registry._pair_model_path(symbol, timeframe)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
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
        loss, acc = model.evaluate(X_test, y_test)
        print(f"Test Accuracy for {symbol} {timeframe}: {acc:.4f}")
        
        return model, history

if __name__ == "__main__":
    # Example usage for training
    trainer = ForexTrainer(api_key="YOUR_TWELVE_DATA_API_KEY") 
    
    # Option 1: Train a global model across multiple pairs
    # df_global = trainer.load_multi_pair_data(
    #     pairs=["EUR/USD", "GBP/USD", "USD/JPY"], 
    #     timeframes=["15min", "30min", "1h"]
    # )
    # X, y = trainer.prepare_sequences(df_global)
    # (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.walk_forward_split(X, y)
    # model, history = trainer.train_global_model(X_train, y_train, X_val, y_val)
    # loss, acc = model.evaluate(X_test, y_test)
    # print(f"Test Accuracy: {acc:.4f}")
    
    # Option 2: Train specialized models for specific pairs
    pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "USD/CAD", "AUD/USD", "NZD/USD"]
    timeframes = ["1min", "5min", "15min", "30min", "1h", "4h", "1day"]
    
    for pair in pairs:
        for tf in timeframes:
            try:
                trainer.train_model_for_pair(symbol=pair, timeframe=tf, epochs=30)
            except Exception as e:
                print(f"Failed to train {pair} {tf}: {e}")
