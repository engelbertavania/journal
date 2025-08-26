from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions(df, yhat_dETH_test, test_index, cfg, suffix="ecm_lstm"):
    # Reconstruct predicted ETH_Open (next day) path for visual comparison (approx)
    eth = df["ETH_Open"]
    # For display, align a series for predictions: pred next open = curr open + pred ΔETH
    pred_next = eth.loc[test_index] + yhat_dETH_test
    actual_next = eth.shift(-1).loc[test_index]

    outdir = Path(cfg["OUT_DIR"])
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10,5))
    plt.plot(actual_next.index, actual_next.values, label="Actual ETH Open (t+1)")
    plt.plot(pred_next.index,   pred_next.values,   label="Predicted ETH Open (t+1)")
    plt.title("Actual vs Predicted ETH Open (ECM–LSTM)")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"pred_vs_actual_{suffix}.png", dpi=160)
    plt.close()
