import numpy as np
import pandas as pd
from pathlib import Path

def _sharpe(returns, risk_free=0.0, periods_per_year=252):
    if len(returns) < 2:
        return np.nan
    excess = returns - risk_free/periods_per_year
    return np.sqrt(periods_per_year) * (excess.mean() / (excess.std(ddof=1) + 1e-12))

def run_backtest_and_save(df, yhat_dETH_test, test_index, cfg):
    """
    Strategy:
      - Predict next-day ΔETH_Open (from ECM–LSTM).
      - Position: sign(pred) ∈ {-1, +1}.
      - Realized return ~ position * realized ΔETH/ETH_Open_prev.
      - Apply 0.1% cost at position *changes* (entry/flip).
    """
    outdir = Path(cfg["OUT_DIR"])
    outdir.mkdir(parents=True, exist_ok=True)

    # Build a frame over test dates
    sub = df.loc[test_index.union(test_index.union([test_index[-1]])), ["ETH_Open"]].copy()
    sub["ETH_Open_next"] = sub["ETH_Open"].shift(-1)
    sub = sub.loc[test_index]  # align to test window only

    # model predictions on ΔETH (Open-to-Open difference)
    sub["pred_dETH"] = yhat_dETH_test
    sub["signal"] = np.sign(sub["pred_dETH"]).replace(0, 0)  # 0 means flat

    # realized ΔETH
    sub["real_dETH"] = sub["ETH_Open_next"] - sub["ETH_Open"]
    sub["ret"] = sub["signal"] * (sub["real_dETH"] / sub["ETH_Open"].shift(1))  # simple approx

    # Trading cost: apply when the signal changes (enter/flip/exit)
    sub["signal_prev"] = sub["signal"].shift(1).fillna(0)
    trades = (sub["signal"] != sub["signal_prev"]).astype(int)
    sub["cost"] = trades * cfg["COST_PER_TRADE"]
    sub["ret_net"] = sub["ret"] - sub["cost"]

    res = {
        "gross_return_%": float(100 * (sub["ret"].add(1).prod() - 1)),
        "net_return_%":   float(100 * (sub["ret_net"].add(1).prod() - 1)),
        "sharpe_gross":   float(_sharpe(sub["ret"].fillna(0))),
        "sharpe_net":     float(_sharpe(sub["ret_net"].fillna(0))),
        "trades":         int(trades.sum()),
    }

    # Save
    sub.to_csv(outdir / "backtest_ecm_lstm.csv")
    with open(outdir / "backtest_summary.json", "w") as f:
        import json; json.dump(res, f, indent=2)
    print("Backtest:", res)
