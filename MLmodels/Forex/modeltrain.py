import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from Data.alphavantage import AlphaVantageClient
from Data.processing import build_forex_feature_set
from forex_models.architectures import GlobalLSTMModel
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib

class ForexTrainer:
    def __init__(self, api_key=None, base_path="/home/job/Desktop/projects/TradeAI/MLmodels/Forex"):
        self.api_key = api_key
        self.base_path = base_path
        self.client = AlphaVantageClient(api_key)
        self.scaler = MinMaxScaler()

    def load_multi_pair_data(self, pairs=["EUR/USD", "GBP/USD", "USD/JPY"], timeframes=["15min", "30min", "1h"]):
        """
        Global Dataset Builder: Combines multiple pairs and timeframes.
        """
        all_data = []
        for pair in pairs:
            for tf_str in timeframes:
                print(f"Fetching {pair} at {tf_str}...")
                from_sym, to_sym = pair.split("/")
                try:
                    df = self.client.get_forex_history(from_sym, to_sym, interval=tf_str)
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
        
        # Save absolute final model and scaler
        model.save(os.path.join(self.base_path, "forex_models/global_base.keras"))
        joblib.dump(self.scaler, os.path.join(self.base_path, "forex_models/global_scaler.pkl"))
        
        return model, history

if __name__ == "__main__":
    # Example usage for Colab / Local
    trainer = ForexTrainer(api_key="YOUR_API_KEY") 
    
    # Building massive dataset
    df_global = trainer.load_multi_pair_data(
        pairs=["EUR/USD", "GBP/USD", "USD/JPY"], 
        timeframes=["15min", "30min", "1h"]
    )
    
    X, y = trainer.prepare_sequences(df_global)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.walk_forward_split(X, y)
    
    # Train
    model, history = trainer.train_global_model(X_train, y_train, X_val, y_val)
    
    # Simple Evaluation
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")