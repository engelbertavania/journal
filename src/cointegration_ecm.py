import pandas as pd
import statsmodels.api as sm

def fit_ecm(df, eg, cfg):
    # 1) OLS long-run ETH_Open ~ BTC_Open
    X = sm.add_constant(df["BTC_Open"])
    ols = sm.OLS(df["ETH_Open"], X, missing='drop').fit()
    resid = ols.resid.dropna()

    # 2) Build ECM on differences with error-correction (lagged residual)
    d = df.diff().dropna()
    d["resid_lag1"] = resid.shift(1).reindex(d.index)

    # Include K lags of ΔETH and ΔBTC (small K keeps it simple & stable)
    K = min(10, cfg["LAGS"])
    for k in range(1, K+1):
        d[f"dETH_lag{k}"] = d["ETH_Open"].shift(k)
        d[f"dBTC_lag{k}"] = d["BTC_Open"].shift(k)
    d = d.dropna()

    y = d["ETH_Open"]
    X = d.drop(columns=["ETH_Open"])
    X = sm.add_constant(X)
    ecm = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    return {
        "ols": ols,
        "ecm": ecm,
        "residuals": resid,
        "ecm_data": d
    }
