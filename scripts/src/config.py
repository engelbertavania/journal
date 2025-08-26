CFG = {
    "TICKERS": {"ETH": "ETH-USD", "BTC": "BTC-USD"},
    "DATE_START": "2018-01-01",
    "DATE_END":   "2024-06-30",
    "TARGET_COL": "ETH_Open",
    "FEATURES":   ["BTC_Open"],
    "TRAIN_SPLIT": 0.8,    # 80/20 split
    "VAL_SPLIT":   0.1,    # from train portion
    "LAGS": 10,            # create up to 10 lags for ΔETH and ΔBTC
    "BATCH_SIZE": 64,
    "EPOCHS": 200,
    "PATIENCE": 20,
    "LR": 1e-3,
    "DROPOUT": 0.2,
    "HIDDEN": 64,
    "COST_PER_TRADE": 0.001,  # 0.1%
    "OUT_DIR": "outputs",
    "SEED": 42
}
