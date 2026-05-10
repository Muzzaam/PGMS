"""
features.py
-----------
Weekly feature construction for the Bitcoin HMM regime project.

Takes a merged daily DataFrame (from load_data.py) and returns a weekly
feature DataFrame ready for modelling. Separated from load_data.py so that
feature definitions can be modified without re-downloading raw data.
"""

import numpy as np
import pandas as pd


def build_weekly_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a merged daily DataFrame to weekly frequency and compute
    all model features.

    Parameters
    ----------
    daily : pd.DataFrame
        Merged daily data with columns:
        ['Date', 'btc_close', 'btc_volume', 'dxy_close',
         'fear_greed_value', 'dgs10']

    Returns
    -------
    pd.DataFrame
        Weekly feature DataFrame with one row per week (Sunday),
        after dropping any rows with NaN values.
    """
    df = daily.copy().set_index("Date")

    # Resample to weekly (week ending Sunday)
    weekly = pd.DataFrame({
        "btc_close":        df["btc_close"].resample("W-SUN").last(),
        "btc_volume":       df["btc_volume"].resample("W-SUN").sum(),
        "dxy_close":        df["dxy_close"].resample("W-SUN").last(),
        "fear_greed_value": df["fear_greed_value"].resample("W-SUN").mean(),
        "dgs10":            df["dgs10"].resample("W-SUN").last(),
    }).dropna().reset_index()

    # --- Core price features ---
    weekly["btc_log_return"]       = np.log(weekly["btc_close"]).diff()
    weekly["btc_volume_log_change"] = np.log(weekly["btc_volume"]).diff()
    weekly["btc_4w_vol"]           = weekly["btc_log_return"].rolling(4).std()

    # --- Macro features ---
    weekly["dxy_log_return"] = np.log(weekly["dxy_close"]).diff()
    weekly["dgs10_change"]   = weekly["dgs10"].diff()

    # --- Psychology / sentiment features ---
    weekly["btc_momentum_4w"]  = np.log(
        weekly["btc_close"] / weekly["btc_close"].shift(4)
    )
    rolling_max_12w = weekly["btc_close"].rolling(12).max()
    weekly["btc_drawdown_12w"] = (weekly["btc_close"] / rolling_max_12w) - 1.0

    # --- Trader / technical-analysis features ---
    weekly["btc_sma_4"]          = weekly["btc_close"].rolling(4).mean()
    weekly["btc_sma_12"]         = weekly["btc_close"].rolling(12).mean()
    weekly["btc_ma_spread_4_12"] = (weekly["btc_sma_4"] / weekly["btc_sma_12"]) - 1.0
    weekly["btc_price_vs_sma12"] = (weekly["btc_close"] / weekly["btc_sma_12"]) - 1.0

    # --- Prediction target ---
    # 1 if next week's log return is positive, 0 otherwise
    weekly["next_week_up"] = (weekly["btc_log_return"].shift(-1) > 0).astype(int)

    weekly = weekly.dropna().reset_index(drop=True)
    return weekly