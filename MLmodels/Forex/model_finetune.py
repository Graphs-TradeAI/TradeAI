import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Data.alphavantage import AlphaVantageClient
from Data.processing import build_forex_feature_set
import joblib

def fine_tune_for_pair(symbol="EUR/USD", timeframe="15min", epochs=30, lr=0.0001):
    base_path = "/home/job/Desktop/projects/TradeAI/MLmodels/Forex"
    global_model_path = os.path.join(base_path, "forex_models/global_base.keras")
    
    if not os.path.exists(global_model_path):
        print("Global base model not found. Please train global model first.")
        return

    # 1. Fetch data for specific pair
    client = AlphaVantageClient()
    from_sym, to_sym = symbol.split("/")
    print(f"Fetching data for {symbol} {timeframe}...")
    df = client.get_forex_history(from_sym, to_sym, interval=timeframe)
    df_features = build_forex_feature_set(df, symbol=symbol, timeframe=timeframe)
    
    # 2. Prepare Sequences
    # Load global scaler or use a new one? Usually better to use global scaler for consistency
    scaler_path = os.path.join(base_path, "forex_models/global_scaler.pkl")
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
    print("Loading global model...")
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
    pair_tag = symbol.replace("/", "")
    save_dir = os.path.join(base_path, f"forex_models/{pair_tag}/{timeframe}")
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        os.path.join(save_dir, "finetuned_model.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 5. Fine-tune
    print(f"Starting fine-tuning for {symbol}...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[checkpoint, early_stop]
    )
    
    print(f"Fine-tuning complete. Saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="EUR/USD")
    parser.add_argument("--timeframe", type=str, default="15min")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    fine_tune_for_pair(symbol=args.symbol, timeframe=args.timeframe, epochs=args.epochs)
