from Data.twelvedata import TwelveDataClient
from Data.processing import build_crypto_feature_set

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler
import numpy as np

from keras import models, layers,Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

api_key = "fb941e0ebad44b4caa431760fcc5bef3"

client = TwelveDataClient(api_key)

df = client.get_crypto_history(
    symbol="BTC/USD",
    exchange="binance",
    interval="30min",
    output_size=5000
)
df_features = build_crypto_feature_set(df)

def clean_data(df):
    df = df.copy()
    df = df.dropna()
    df = df.drop_duplicates()
    return df

df_features = clean_data(df_features)

def prepare_lstm_data(
    df,
    feature_cols,
    target_col="future_close",
    seq_length=50
):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])

    X, y = [], []
    for i in range(len(df) - seq_length):
        X.append(scaled_features[i:i+seq_length])
        y.append(df[target_col].iloc[i+seq_length])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    return X_train, X_test, y_train, y_test, scaler
feature_cols = [
    c for c in df_features.columns
    if c not in ["timestamp"]
]

X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(
    df_features,
    feature_cols,
    seq_length=50
)
model = models.Sequential([
    layers.LSTM(256, return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.Dropout(0.2),
    layers.LSTM(256),
    layers.Dense(1)
])

earlystop = EarlyStopping(
    monitor="val_mse",
    patience=2,
    restore_best_weights=True,
    mode="min",
    verbose=1
)

modelcheckpoint = ModelCheckpoint(
    filepath="/home/job/Desktop/projects/TradeAI/MLmodels/Crypto/crypto_models/BTCUSDT/30min/model.keras",
    monitor="val_mse",
    save_best_only=True,
    verbose=1,
    mode="min"
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae", "mse"]
)

model.summary()

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=64,
    callbacks=[earlystop, modelcheckpoint]
)
