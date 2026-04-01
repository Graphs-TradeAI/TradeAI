import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping

from client import TwelveDataClient
from indicators import build_features, get_feature_columns
from registry import ModelRegistry


class ForexTrainer:
    def __init__(self, base_path="/home/job/Desktop/projects/TradeAI/Forex/Models"):
        load_dotenv()
        self.api_key = os.getenv("TWELVE_DATA_API_KEY")
        self.base_path = base_path
        self.client = TwelveDataClient(self.api_key)
        self.registry = ModelRegistry()

    # ----------------------------
    # Sequence builder (FIXED)
    # ----------------------------
    def prepare_sequences(self, df, seq_length=60, feature_cols=None, target_col="target_return"):
        df = df.copy()

        if feature_cols is None:
            feature_cols = get_feature_columns(df)

        X, y = [], []

        values = df[feature_cols].values
        targets = df[target_col].values

        for i in range(len(df) - seq_length):
            X.append(values[i:i + seq_length])
            y.append(targets[i + seq_length])

        return np.array(X), np.array(y)

    # ----------------------------
    # Chronological split
    # ----------------------------
    def walk_forward_split(self, X, y, train_ratio=0.7, val_ratio=0.15):
        n = len(X)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return (
            (X[:train_end], y[:train_end]),
            (X[train_end:val_end], y[train_end:val_end]),
            (X[val_end:], y[val_end:])
        )

    # ----------------------------
    # NO SCALING (IMPORTANT FIX)
    # ----------------------------
    def scale_data(self, X_train, X_val, X_test):
        """
        Removed scaling entirely because:
        - features already normalized in feature pipeline
        - LSTM performs better on stable ratio-based features
        """

        return X_train, X_val, X_test, None

    # ----------------------------
    # Model
    # ----------------------------
    def build_sequential_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True),
            Dropout(0.25),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.15),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            # regression output (return prediction)
            Dense(1, activation="linear")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.Huber(),  # better than MSE for finance
            metrics=["mae"]
        )

        return model

    # ----------------------------
    # Training pipeline
    # ----------------------------
    def train_model_for_pair(
        self,
        symbol="AUD/USD",
        timeframe="1day",
        epochs=100,
        batch_size=64
    ):

        print(f"\nTraining model for {symbol} {timeframe}\n")

        # 1. Fetch data
        df = self.client.get_forex_history(
            symbol=symbol,
            interval=timeframe,
            output_size=5000
        )

        # 2. Feature engineering
        df_features = build_features(
            df,
            symbol=symbol,
            timeframe=timeframe
        )

        # 3. Drop leakage columns if present
        df_features = df_features.dropna().reset_index(drop=True)

        # 4. Build sequences (IMPORTANT FIX: target_return)
        X, y = self.prepare_sequences(
            df_features,
            target_col="target_return"
        )

        # 5. Chronological split BEFORE training
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.walk_forward_split(X, y)

        # 6. No scaling (clean pipeline)
        X_train, X_val, X_test, _ = self.scale_data(X_train, X_val, X_test)

        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # 7. Model
        model = self.build_sequential_model(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )

        model.summary()

        # 8. Paths
        model_dir = self.registry._pair_dir(symbol, timeframe)
        os.makedirs(model_dir, exist_ok=True)
        model_path = self.registry._pair_model_path(symbol, timeframe)

        # 9. Callbacks
        checkpoint = ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            # No mode=min here, it should be mode='min' or auto
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.registry.cfg.get("training", {}).get("patience", 4),
            restore_best_weights=True,
            verbose=1
        )

        # 10. Train
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stop],
            shuffle=False  # CRITICAL for time series
        )

        # 11. Evaluate
        loss, mae = model.evaluate(X_test, y_test)
        print(f"\nTest MAE ({symbol} {timeframe}): {mae:.6f}\n")

        return model, history


if __name__ == "__main__":
    trainer = ForexTrainer()

    try:
        trainer.train_model_for_pair(
            symbol="AUD/USD",
            timeframe="1day",
            epochs=100
        )
    except Exception as e:
        print(f"Training failed: {e}")