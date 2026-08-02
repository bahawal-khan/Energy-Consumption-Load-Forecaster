"""
model.py
========
Phase 2 of the Energy Consumption Forecasting system: a *pure* Artificial
Neural Network (feed-forward MLP) for regression -- no CNN/RNN/LSTM
components, per the spec.

Architecture:
    Input(6 features)
      -> Dense(128, relu) -> Dropout(0.2)
      -> Dense(64,  relu) -> Dropout(0.2)
      -> Dense(32,  relu) -> Dropout(0.1)
      -> Dense(1,   linear)   # continuous regression output (scaled 0-1)

Overfitting controls: Dropout + EarlyStopping + ModelCheckpoint (best val_loss).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_ann(input_dim: int, dropout_rate: float = 0.2, learning_rate: float = 1e-3) -> tf.keras.Model:
    """Builds and compiles the pure MLP regression model."""
    model = models.Sequential(name="energy_load_forecaster_mlp")
    model.add(layers.Input(shape=(input_dim,), name="features"))

    model.add(layers.Dense(128, activation="relu", name="dense_1"))
    model.add(layers.Dropout(dropout_rate, name="dropout_1"))

    model.add(layers.Dense(64, activation="relu", name="dense_2"))
    model.add(layers.Dropout(dropout_rate, name="dropout_2"))

    model.add(layers.Dense(32, activation="relu", name="dense_3"))
    model.add(layers.Dropout(dropout_rate * 0.5, name="dropout_3"))

    model.add(layers.Dense(1, activation="linear", name="prediction"))  # continuous output

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_callbacks(checkpoint_path: str, patience: int = 15) -> list:
    """EarlyStopping + ModelCheckpoint, as specified in Phase 2."""
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    checkpoint_path: str = "artifacts/model.h5",
    epochs: int = 200,
    batch_size: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Full training loop with overfitting controls."""
    model = build_ann(input_dim=X_train.shape[1], dropout_rate=dropout_rate, learning_rate=learning_rate)
    cbs = get_callbacks(checkpoint_path)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=2,
    )
    return model, history


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, target_scaler) -> dict:
    """Evaluate on the held-out chronological test set, in real kW units (inverse-scaled)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_true = target_scaler.inverse_transform(y_test)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100)

    metrics = {"mae_kw": float(mae), "rmse_kw": float(rmse), "r2": float(r2), "mape_pct": mape}
    logger.info("Test metrics: %s", metrics)
    return metrics


def load_trained_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)
