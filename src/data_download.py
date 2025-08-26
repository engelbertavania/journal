import yfinance as yf
import pandas as pd

def load_prices(cfg):
    eth = yf.download(cfg["TICKERS"]["ETH"], start=cfg["DATE_START"], end=cfg["DATE_END"])
    btc = yf.download(cfg["TICKERS"]["BTC"], start=cfg["DATE_START"], end=cfg["DATE_END"])
    df = pd.DataFrame({
        "ETH_Open": eth["Open"],
        "BTC_Open": btc["Open"],
    }).dropna().copy()
    df.index.name = "Date"
    return df
