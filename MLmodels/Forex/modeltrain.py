from Data.twelvedata import TwelveDataClient
from Data.processing import build_forex_feature_set
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import models,layers
from keras.callbacks import ModelCheckpoint,EarlyStopping
import pandas as pd

api_key = "fb941e0ebad44b4caa431760fcc5bef3"

client = TwelveDataClient(api_key)

df = client.get_forex_history(
    symbol="AUD/USD",
    interval="30min",
    output_size=5000
)
df_features = build_forex_feature_set(df)

#data cleaning
def clean_data(dataframe):
    dataframe.dropna(inplace=True)
    dataframe.drop_duplicates()
    for col in dataframe.columns:
        q1=dataframe[col].quantile(0.25)
        q3=dataframe[col].quantile(0.75)
        iqr=q3-q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        df_cleaned=dataframe[(dataframe[col]<=lower_bound) & (dataframe[col]>=upper_bound)]
        return df_cleaned
    
clean_data(df_features)

def prepare_lstm_data(df, feature_cols, target_col="future_close", seq_length=50):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    

    X, y = [], []
    for i in range(len(df) - seq_length):
        X.append(scaled_features[i:i+seq_length])
        y.append(df[target_col].iloc[i+seq_length])

    X = np.array(X)
    y = np.array(y)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train,X_test,y_train,y_test,scaler


feature_cols = [c for c in df_features.columns if c not in ["timestamp","future_close"]]
X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df_features, feature_cols, seq_length=50)


model=models.Sequential([
    layers.LSTM(256,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])),
    layers.Dropout(0.2),
    layers.LSTM(256),
    layers.Dense(1)
])


earlystop=EarlyStopping(
    monitor="val_mse",
    patience=4,
    restore_best_weights=True,
    mode='min',
    verbose=1
)
modelcheckpoint=ModelCheckpoint(
    filepath="/home/job/Desktop/projects/TradeAI/MLmodels/Forex/forex_models/AUDUSD/30min/model.keras",
    monitor='val_mse',
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
    mode='min'
)

# Select feature columns (all except timestamp, future_close, target_up)

# Prepare sequences

# Build model


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae","mse"]
)
model.summary()
# Train 
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    callbacks=[earlystop,modelcheckpoint]
)

