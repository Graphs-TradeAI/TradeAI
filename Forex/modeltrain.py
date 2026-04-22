import os
import hashlib
from dotenv import load_dotenv
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping

from Forex.client import TwelveDataClient
from Forex.indicators import build_features, get_feature_columns
from Forex.registry import ModelRegistry


class ForexTrainer:
    def __init__(self, base_path=None):
        load_dotenv()
        self.api_key = os.getenv("TWELVE_DATA_API_KEY")
        if base_path is None:
            self.base_path = os.path.join(os.path.dirname(__file__), "Models")
        else:
            self.base_path = base_path
        self.client = TwelveDataClient(self.api_key)
        self.registry = ModelRegistry(models_root=self.base_path)
        self.train_cfg = self.registry.cfg.get("training", {})

        # Deterministic defaults for reproducible training runs.
        seed = int(self.train_cfg.get("random_seed", 42))
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # ----------------------------
    # Sequence builder (FIXED)
    # ----------------------------
    def prepare_sequences(
        self,
        df,
        seq_length=60,
        feature_cols=None,
        reg_target_col="target_return",
        cls_target_col="target_direction",
    ):
        df = df.copy()

        if feature_cols is None:
            feature_cols = get_feature_columns(df)

        X, y_reg, y_cls = [], [], []

        values = df[feature_cols].values
        reg_targets = df[reg_target_col].values
        cls_targets = df[cls_target_col].values

        for i in range(len(df) - seq_length):
            X.append(values[i:i + seq_length])
            y_reg.append(reg_targets[i + seq_length])
            y_cls.append(cls_targets[i + seq_length])

        return np.array(X), np.array(y_reg), np.array(y_cls)

    # ----------------------------
    # Chronological split
    # ----------------------------
    def walk_forward_split(self, X, y, train_ratio=None, val_ratio=None):
        n = len(X)
        if train_ratio is None:
            train_ratio = float(self.train_cfg.get("train_ratio", 0.70))
        if val_ratio is None:
            val_ratio = float(self.train_cfg.get("val_ratio", 0.15))

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
        learning_rate = float(self.train_cfg.get("learning_rate", 0.001))
        reg_loss_weight = float(self.train_cfg.get("regression_loss_weight", 0.7))
        cls_loss_weight = float(self.train_cfg.get("classification_loss_weight", 0.3))

        inputs = Input(shape=input_shape, name="sequence_input")
        x = LSTM(128, return_sequences=True, name="lstm_1")(inputs)
        x = Dropout(0.25, name="dropout_1")(x)
        x = LSTM(64, return_sequences=True, name="lstm_2")(x)
        x = Dropout(0.2, name="dropout_2")(x)
        x = LSTM(32, name="lstm_3")(x)
        x = Dropout(0.15, name="dropout_3")(x)
        shared = Dense(32, activation="relu", name="shared_dense")(x)

        return_branch = Dense(16, activation="relu", name="return_dense")(shared)
        return_head = Dense(1, activation="linear", name="return_head")(return_branch)

        direction_branch = Dense(16, activation="relu", name="direction_dense")(shared)
        direction_head = Dense(1, activation="sigmoid", name="direction_head")(direction_branch)

        model = Model(inputs=inputs, outputs=[return_head, direction_head], name="forex_multitask_lstm")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                "return_head": tf.keras.losses.Huber(),  # regression target_return
                "direction_head": tf.keras.losses.BinaryCrossentropy(),
            },
            loss_weights={
                "return_head": reg_loss_weight,
                "direction_head": cls_loss_weight,
            },
            metrics={
                "return_head": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
                "direction_head": [
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                    tf.keras.metrics.AUC(name="auc"),
                ],
            },
        )

        return model

    # ----------------------------
    # Training pipeline
    # ----------------------------
    def train_model_for_pair(
        self,
        symbol: str,
        timeframe: str,
        epochs=None,
        batch_size=None,
        output_size=None,
    ):

        print(f"\nTraining model for {symbol} {timeframe}\n")
        epochs = int(epochs if epochs is not None else self.train_cfg.get("epochs", 100))
        batch_size = int(batch_size if batch_size is not None else self.train_cfg.get("batch_size", 64))
        output_size = int(output_size if output_size is not None else self.train_cfg.get("output_size", 5000))
        seq_length = int(self.train_cfg.get("sequence_length", 60))

        # 1. Fetch data
        df = self.client.get_forex_history(
            symbol=symbol,
            interval=timeframe,
            output_size=output_size,
        )

        # 2. Feature engineering
        df_features = build_features(
            df,
            symbol=symbol,
            timeframe=timeframe
        )

        # 3. Drop leakage columns if present
        df_features = df_features.dropna().reset_index(drop=True)

        # 4. Build sequences (multitask: regression + classification)
        X, y_reg, y_cls = self.prepare_sequences(
            df_features,
            seq_length=seq_length,
            reg_target_col="target_return",
            cls_target_col="target_direction",
        )
        min_sequences = int(self.train_cfg.get("min_sequences", 300))
        if len(X) < min_sequences:
            raise ValueError(
                f"Insufficient sequences for training ({len(X)} < {min_sequences}) "
                f"for {symbol} {timeframe}"
            )

        # 5. Chronological split BEFORE training
        (X_train, y_reg_train), (X_val, y_reg_val), (X_test, y_reg_test) = self.walk_forward_split(X, y_reg)
        (_, y_cls_train), (_, y_cls_val), (_, y_cls_test) = self.walk_forward_split(X, y_cls)
        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            raise ValueError(
                f"Invalid split sizes for {symbol} {timeframe}: "
                f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
            )

        # 6. No scaling (clean pipeline)
        X_train, X_val, X_test, _ = self.scale_data(X_train, X_val, X_test)

        y_reg_train = y_reg_train.reshape(-1, 1)
        y_reg_val = y_reg_val.reshape(-1, 1)
        y_reg_test = y_reg_test.reshape(-1, 1)

        y_cls_train = y_cls_train.reshape(-1, 1).astype(np.float32)
        y_cls_val = y_cls_val.reshape(-1, 1).astype(np.float32)
        y_cls_test = y_cls_test.reshape(-1, 1).astype(np.float32)

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
            {"return_head": y_reg_train, "direction_head": y_cls_train},
            validation_data=(X_val, {"return_head": y_reg_val, "direction_head": y_cls_val}),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stop],
            shuffle=False  # CRITICAL for time series
        )

        # 11. Evaluate
        eval_metrics = model.evaluate(
            X_test,
            {"return_head": y_reg_test, "direction_head": y_cls_test},
            return_dict=True,
            verbose=0,
        )
        test_mae = float(eval_metrics.get("return_head_mae", 0.0))
        test_acc = float(eval_metrics.get("direction_head_accuracy", 0.0))
        test_auc = float(eval_metrics.get("direction_head_auc", 0.0))
        print(
            f"\nTest MAE ({symbol} {timeframe}): {test_mae:.6f} | "
            f"Direction Acc: {test_acc:.4f} | AUC: {test_auc:.4f}\n"
        )

        # 12. Persist summary metrics for registry-backed APIs
        feature_cols = get_feature_columns(df_features)
        feature_signature = "|".join(feature_cols)
        feature_hash = hashlib.sha256(feature_signature.encode("utf-8")).hexdigest()

        metrics = {
            "symbol": symbol,
            "timeframe": timeframe,
            "task_type": "multitask_regression_classification",
            "loss": float(eval_metrics.get("loss", 0.0)),
            "mae": test_mae,
            "directional_accuracy": test_acc,
            "direction_auc": test_auc,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "sequence_length": int(X_train.shape[1]),
            "n_features": int(X_train.shape[2]),
            "feature_columns": feature_cols,
            "feature_columns_hash": feature_hash,
            "train_ratio": float(self.train_cfg.get("train_ratio", 0.70)),
            "val_ratio": float(self.train_cfg.get("val_ratio", 0.15)),
            "output_size": output_size,
            "model_outputs": ["return_head", "direction_head"],
        }
        self.registry.save_metrics(symbol, timeframe, metrics)

        return model, history

    def _resolve_training_grid(
        self,
        pairs=None,
        timeframes=None,
    ):
        cfg_pairs = self.registry.cfg.get("pairs", [])
        cfg_timeframes = self.registry.cfg.get("timeframes", [])

        resolved_pairs = list(pairs) if pairs is not None else list(cfg_pairs)
        resolved_timeframes = list(timeframes) if timeframes is not None else list(cfg_timeframes)

        if not resolved_pairs:
            raise ValueError("No training pairs found. Populate `pairs` in config.yaml.")
        if not resolved_timeframes:
            raise ValueError("No training timeframes found. Populate `timeframes` in config.yaml.")

        return resolved_pairs, resolved_timeframes

    def train_from_config(
        self,
        pairs=None,
        timeframes=None,
        epochs=None,
        batch_size=None,
        output_size=None,
        stop_on_error=False,
    ):
        """
        Train all pair/timeframe combinations from config (or provided overrides).

        Returns
        -------
        dict summary with totals and per-run status.
        """
        resolved_pairs, resolved_timeframes = self._resolve_training_grid(pairs, timeframes)
        total_jobs = len(resolved_pairs) * len(resolved_timeframes)
        results = []

        print(
            f"\nStarting config-driven training loop "
            f"({len(resolved_pairs)} pairs × {len(resolved_timeframes)} timeframes = {total_jobs} jobs)\n"
        )

        job_idx = 0
        for symbol in resolved_pairs:
            for timeframe in resolved_timeframes:
                job_idx += 1
                print(f"[{job_idx}/{total_jobs}] {symbol} {timeframe}")
                try:
                    self.train_model_for_pair(
                        symbol=symbol,
                        timeframe=timeframe,
                        epochs=epochs,
                        batch_size=batch_size,
                        output_size=output_size,
                    )
                    results.append(
                        {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "status": "ok",
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    print(f"Failed {symbol} {timeframe}: {exc}")
                    if stop_on_error:
                        raise

        ok_count = sum(1 for item in results if item["status"] == "ok")
        failed_count = total_jobs - ok_count
        summary = {
            "total_jobs": total_jobs,
            "success_count": ok_count,
            "failed_count": failed_count,
            "results": results,
        }

        print(
            f"\nTraining loop complete: "
            f"{ok_count}/{total_jobs} succeeded, {failed_count} failed.\n"
        )
        return summary


if __name__ == "__main__":
    trainer = ForexTrainer()

    try:
        trainer.train_from_config()
    except Exception as e:
        print(f"Training failed: {e}")
