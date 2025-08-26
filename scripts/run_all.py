from pathlib import Path
from src.config import CFG
from src.data_download import load_prices
from src.preprocess import make_stationary, engle_granger
from src.cointegration_ecm import fit_ecm
from src.build_features import make_supervised_features
from src.lstm_model import build_lstm
from src.train_eval import train_and_eval, save_metrics
from src.backtest import run_backtest_and_save
from src.plots import plot_predictions

Path(CFG["OUT_DIR"]).mkdir(parents=True, exist_ok=True)

# 1) Data
df = load_prices(CFG)

# 2) Stationarity + Engle–Granger cointegration (ETH ~ BTC)
stationarity = make_stationary(df, CFG)
eg = engle_granger(df, CFG)

# 3) ECM fit on (ΔETH, ΔBTC) with error-correction term
ecm_fit = fit_ecm(df, eg, CFG)

# 4) Build supervised set for LSTM (ECM residuals + selected lags)
X_train, y_train, X_val, y_val, X_test, y_test, scaler_y, feat_info = make_supervised_features(df, ecm_fit, CFG)

# 5) LSTM (ECM-informed) + training
model = build_lstm(input_dim=X_train.shape[-1], cfg=CFG)
hist, preds_test, metrics = train_and_eval(model, (X_train, y_train, X_val, y_val, X_test, y_test), scaler_y, CFG)

# 6) Save metrics and plots
save_metrics(metrics, CFG, suffix="ecm_lstm")
plot_predictions(df, preds_test, feat_info["test_index"], CFG, suffix="ecm_lstm")

# 7) Baselines: standalone LSTM, random-walk handled inside train_and_eval when flags set
# (already computed; metrics include baselines)

# 8) Backtest: long–short on predicted ΔETH → implied next open
run_backtest_and_save(df, preds_test, feat_info["test_index"], CFG)

print("Pipeline done. Outputs saved under:", CFG["OUT_DIR"])
