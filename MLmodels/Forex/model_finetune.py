import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from MLmodels.Forex.data_layer.client import TwelveDataClient
from Data.processing import build_forex_feature_set
import joblib

def fine_tune_for_pair(symbol="EUR/USD", timeframe="15min", epochs=30, lr=0.0001, api_key=None):
    """
    Fine-tune the global LSTM model for a specific currency pair and timeframe.
    Uses Twelve Data API for price data.
    """
    registry = ModelRegistry()
    global_model_path = registry._pair_model_path("global", "base")
    if not os.path.exists(global_model_path):
        print("Global base model not found. Please train global model first.")
        return

    # 1. Fetch data for specific pair using Twelve Data
    client = TwelveDataClient(api_key)
    print(f"Fetching data for {symbol} {timeframe}...")
    df = client.get_forex_history(symbol=symbol, interval=timeframe, output_size=5000)
    df_features = build_forex_feature_set(df, symbol=symbol, timeframe=timeframe)
    
    # 2. Prepare Sequences
    # Load global scaler for consistency
    scaler_path = registry._pair_scaler_path("global", "base")
    scaler = joblib.load(scaler_path)
    
    feature_cols = [c for c in df_features.columns if c not in ["timestamp", "target_direction", "target_return_5"]]
    scaled_data = scaler.transform(df_features[feature_cols])
    
    seq_length = 60
    X, y = [], []
    for i in range(len(df_features) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(df_features["target_direction"].iloc[i+seq_length])
    
    X, y = np.array(X), np.array(y)
    
    # Simple chronological split for fine-tuning
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # 3. Load and Modify Model
    print("Loading global LSTM model...")
    model = models.load_model(global_model_path)
    
    # Optional: Freeze first 2 LSTM layers to preserve global patterns
    # for layer in model.layers[:3]: 
    #     layer.trainable = False
        
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 4. Save Path
    model_path = registry._pair_model_path(symbol, timeframe)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    print(f"Starting fine-tuning LSTM for {symbol}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[checkpoint, early_stop]
    )
    model.save(model_path)
    print(f"Fine-tuning complete. Saved to {model_path}")
    
    # Print final metrics
    val_acc = history.history['val_accuracy'][-1]
    print(f"Final validation accuracy: {val_acc:.4f}")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LSTM model for a specific forex pair")
    parser.add_argument("--symbol", type=str, default="EUR/USD", help="Forex pair symbol (e.g., EUR/USD)")
    parser.add_argument("--timeframe", type=str, default="15min", help="Timeframe (e.g., 1min, 5min, 15min, 30min, 1h, 4h, 1day)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--api_key", type=str, default=None, help="Twelve Data API key")
    args = parser.parse_args()
    
    fine_tune_for_pair(
        symbol=args.symbol, 
        timeframe=args.timeframe, 
        epochs=args.epochs,
        lr=args.lr,
        api_key=args.api_key
    )
