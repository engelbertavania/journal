# ECM–LSTM for Ethereum Opening Price Prediction

This repo replicates the workflow from the manuscript “Hybrid Error Correction Model and Long Short-Term Memory Approach for Predicting Ethereum Opening Prices”: data (Jan 2018–Jun 2024), Engle–Granger cointegration, ECM residual→LSTM features, 80/20 split, EarlyStopping, metrics (MAE, MSE, R²), and a simple long–short backtest with 0.1% per-trade cost and Sharpe ratio. :contentReference[oaicite:1]{index=1}

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
