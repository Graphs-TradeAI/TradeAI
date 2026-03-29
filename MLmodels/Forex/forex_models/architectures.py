import tensorflow as tf
from tensorflow.keras import layers, models
import xgboost as xgb
import os
import joblib

class GlobalLSTMModel:
    """
    A multi-layer LSTM model for global forex structure learning.
    Includes timeframe and pair conditioning as features.
    """
    @staticmethod
    def build(input_shape, lr=0.001):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(256, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid') # For direction probability
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

class XGBoostHybrid:
    """
    XGBoost model for hybrid market regime prediction.
    Typically used for high-volatility regimes.
    """
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'eval_metric': 'logloss'
        }
        self.model = xgb.XGBClassifier(**self.params)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None:
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train)

    def save(self, path):
        self.model.save_model(path)
    
    def load(self, path):
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)

class ModelFactory:
    """
    Handles model creation, loading, and conversion for the selection agent.
    """
    @staticmethod
    def save_scaler(scaler, path):
        joblib.dump(scaler, path)

    @staticmethod
    def load_scaler(path):
        return joblib.load(path)
