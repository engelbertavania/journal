import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

def adf_pvalue(series):
    return adfuller(series, autolag="AIC")[1]

def make_stationary(df, cfg):
    out = {}
    out["ADF_ETH_Open"] = adf_pvalue(df["ETH_Open"])
    out["ADF_BTC_Open"] = adf_pvalue(df["BTC_Open"])
    # first differences
    d = df.diff().dropna()
    out["ADF_dETH"] = adf_pvalue(d["ETH_Open"])
    out["ADF_dBTC"] = adf_pvalue(d["BTC_Open"])
    return out

def engle_granger(df, cfg):
    # Engle–Granger: test cointegration ETH_Open ~ BTC_Open
    score, pvalue, _ = coint(df["ETH_Open"], df["BTC_Open"])
    return {"coint_stat": float(score), "pvalue": float(pvalue)}
