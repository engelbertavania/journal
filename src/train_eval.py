import json
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .lstm_model import get_callbacks
from .config import CFG

def _metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "R2":  float(r2_score(y_true, y_pred))
    }

def _random_walk_baseline(y_true):
    # Predict 0 change (ΔETH = 0)
    return np.zeros_like(y_true)

def train_and_eval(model, data, scaler_y, cfg):
    X_tr, y_tr, X_va, y_va, X_te, y_te = data
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                     epochs=cfg["EPOCHS"], batch_size=cfg["BATCH_SIZE"],
                     callbacks=get_callbacks(cfg), verbose=0)

    # Predictions (scaled) → invert scale for reporting on ΔETH scale
    yhat_scaled = model.predict(X_te, verbose=0).ravel()
    yhat = scaler_y.inverse_transform(yhat_scaled.reshape(-1,1)).ravel()
    y_true = scaler_y.inverse_transform(y_te.reshape(-1,1)).ravel()

    # Baselines
    rw = _random_walk_baseline(y_true)
    metrics = {
        "ecm_lstm": _metrics(y_true, yhat),
        "naive_rw": _metrics(y_true, rw),
    }

    return hist, yhat, metrics

def save_metrics(metrics, cfg, suffix="ecm_lstm"):
    Path(cfg["OUT_DIR"]).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg["OUT_DIR"]) / f"metrics_{suffix}.json", "w") as f:
        json.dump(metrics, f, indent=2)
