from Data.twelvedata import TwelveDataClient
from Data.processing import build_forex_feature_set
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import argparse
import os

def fine_tune_model(symbol="AUD/USD", interval="15min", epochs=20, batch_size=64, learning_rate=0.0001):
    api_key = "fb941e0ebad44b4caa431760fcc5bef3"
    client = TwelveDataClient(api_key)
    
    print(f"Fetching latest data for {symbol} ({interval})...")
    df = client.get_forex_history(symbol=symbol, interval=interval, output_size=5000)
    df_features = build_forex_feature_set(df)
    
    # Path construction
    # Mapping symbol "AUD/USD" -> "AUDUSD" for directory structure
    symbol_dir = symbol.replace("/", "")
    model_dir = f"/home/job/Desktop/projects/TradeAI/MLmodels/Forex/forex_models/{symbol_dir}/{interval}"
    model_path = os.path.join(model_dir, "model.keras")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train it first using modeltrain.py.")
        return

    print(f"Loading existing model from {model_path}...")
    model = models.load_model(model_path)
    
    # Preparation for LSTM (same logic as modeltrain.py)
    feature_cols = [c for c in df_features.columns if c not in ["timestamp"]]
    
    def prepare_lstm_data(df, feature_cols, target_col="future_close", seq_length=50):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[feature_cols])
        
        X, y = [], []
        for i in range(len(df) - seq_length):
            X.append(scaled_features[i:i+seq_length])
            y.append(df[target_col].iloc[i+seq_length])
            
        X = np.array(X)
        y = np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test, scaler

    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df_features, feature_cols)
    
    # Re-compile with a lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae", "mse"]
    )
    
    earlystop = EarlyStopping(
        monitor="val_mse",
        patience=5,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    
    modelcheckpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_mse',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode='min'
    )
    
    print(f"Starting fine-tuning for {epochs} epochs...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[earlystop, modelcheckpoint]
    )
    print(f"Fine-tuning complete. Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Forex LSTM models.")
    parser.add_argument("--symbol", type=str, default="AUD/USD", help="Forex pair (e.g., AUD/USD)")
    parser.add_argument("--interval", type=str, default="15min", help="Timeframe (e.g., 15min, 1h, 1d)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for fine-tuning")
    
    args = parser.parse_args()
    
    # Handle cases where user might input AUDUSD instead of AUD/USD
    symbol = args.symbol
    if "/" not in symbol and len(symbol) == 6:
        symbol = f"{symbol[:3]}/{symbol[3:]}"
        
    fine_tune_model(symbol=symbol, interval=args.interval, epochs=args.epochs, learning_rate=args.lr)
