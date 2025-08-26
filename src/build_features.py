import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def split_indices(n, train_frac, val_frac):
    n_train = int(n * train_frac)
    n_val   = int(n_train * val_frac)
    return slice(0, n_train - n_val), slice(n_train - n_val, n_train), slice(n_train, n)

def make_supervised_features(df, ecm_fit, cfg):
    # Base frame with ΔETH, ΔBTC and ECM residual lag (error-correction)
    d = df.diff().dropna().copy()
    d["ecm_resid_lag1"] = ecm_fit["residuals"].shift(1).reindex(d.index)

    # Add lags up to cfg["LAGS"]
    L = cfg["LAGS"]
    for k in range(1, L+1):
        d[f"dETH_lag{k}"] = d["ETH_Open"].shift(k)
        d[f"dBTC_lag{k}"] = d["BTC_Open"].shift(k)

    data = d.dropna().copy()
    y = data["ETH_Open"].values  # predict ΔETH (difference of open)
    X = data.drop(columns=["ETH_Open"]).values

    # standardize X, but not y (we'll scale y for the network’s training stability then invert)
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

    # Train/val/test split by time
    train_slice, val_slice, test_slice = split_indices(len(X), cfg["TRAIN_SPLIT"], cfg["VAL_SPLIT"])

    X_train, y_train = X[train_slice], y_scaled[train_slice]
    X_val,   y_val   = X[val_slice],   y_scaled[val_slice]
    X_test,  y_test  = X[test_slice],  y_scaled[test_slice]

    # LSTM expects [batch, timesteps, features]; we’ll use a single-step window
    X_train = X_train.reshape((-1, 1, X.shape[1]))
    X_val   = X_val.reshape((-1, 1, X.shape[1]))
    X_test  = X_test.reshape((-1, 1, X.shape[1]))

    return (X_train, y_train, X_val, y_val, X_test, y_test, scaler_y,
            {"test_index": data.index[test_slice], "feature_names": list(data.drop(columns=["ETH_Open"]).columns)})
