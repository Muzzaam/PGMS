"""
load_data.py
------------
Raw data download and weekly feature construction for the Bitcoin HMM
regime project.

Downloads four data sources:
  - BTC-USD daily OHLCV from Yahoo Finance
  - DXY (US Dollar Index) daily from Yahoo Finance
  - Crypto Fear & Greed Index daily from alternative.me API
  - 10-year US Treasury yield (DGS10) from FRED

Merges them at daily frequency, then delegates weekly feature engineering
to features.py. Run this script once to populate data/raw/ and
data/processed/ before running any modelling scripts.
"""

from pathlib import Path

import numpy as np
import requests
import pandas as pd
import yfinance as yf

from features import build_weekly_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2018-02-01"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a MultiIndex column structure returned by yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def download_yahoo_series(ticker: str, out_name: str) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance and save to raw/.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol, e.g. 'BTC-USD'.
    out_name : str
        Filename to save in data/raw/.

    Returns
    -------
    pd.DataFrame
    """
    df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=False)
    df = flatten_columns(df).reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df.to_csv(RAW_DIR / out_name, index=False)
    return df


def download_fear_greed() -> pd.DataFrame:
    """
    Download the full Crypto Fear & Greed Index history from alternative.me.

    The index is a composite of market momentum, social media sentiment,
    surveys, and volatility. Values range from 0 (Extreme Fear) to 100
    (Extreme Greed). Note that the index methodology has changed over time,
    introducing potential non-stationarity in the series.

    Returns
    -------
    pd.DataFrame with columns ['Date', 'fear_greed_value', 'value_classification']
    """
    url = "https://api.alternative.me/fng/?limit=0"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()

    df = pd.DataFrame(payload["data"])
    df["Date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
    df["fear_greed_value"] = pd.to_numeric(df["value"], errors="coerce")
    df = (
        df[["Date", "fear_greed_value", "value_classification"]]
          .sort_values("Date")
          .reset_index(drop=True)
    )
    df.to_csv(RAW_DIR / "fear_greed_daily.csv", index=False)
    return df


def download_fred_series(
    series_id: str,
    value_name: str,
    out_name: str,
) -> pd.DataFrame:
    """
    Download a daily time series from the FRED public CSV API.

    Parameters
    ----------
    series_id : str
        FRED series identifier, e.g. 'DGS10'.
    value_name : str
        Column name for the downloaded values.
    out_name : str
        Filename to save in data/raw/.

    Returns
    -------
    pd.DataFrame with columns ['Date', value_name]
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["Date", value_name]
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df.to_csv(RAW_DIR / out_name, index=False)
    return df


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def build_daily_merged(
    btc: pd.DataFrame,
    dxy: pd.DataFrame,
    fear_greed: pd.DataFrame,
    dgs10: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all daily series into a single DataFrame aligned on trading dates.

    Fear & Greed and DGS10 are forward-filled to handle weekends and
    public holidays where those series have no observation.

    Parameters
    ----------
    btc : pd.DataFrame
        BTC-USD daily OHLCV.
    dxy : pd.DataFrame
        DXY daily OHLCV.
    fear_greed : pd.DataFrame
        Daily Fear & Greed values.
    dgs10 : pd.DataFrame
        Daily 10-year Treasury yield.

    Returns
    -------
    pd.DataFrame
        Merged daily DataFrame saved to data/processed/daily_merged.csv.
    """
    btc = btc.rename(columns={"Close": "btc_close", "Volume": "btc_volume"})[
        ["Date", "btc_close", "btc_volume"]
    ]
    dxy = dxy.rename(columns={"Close": "dxy_close"})[["Date", "dxy_close"]]

    daily = (
        btc.merge(dxy, on="Date", how="inner")
           .merge(fear_greed[["Date", "fear_greed_value"]], on="Date", how="left")
           .merge(dgs10, on="Date", how="left")
           .sort_values("Date")
           .reset_index(drop=True)
    )

    # Forward-fill sentiment and yield over weekends / holidays
    daily["fear_greed_value"] = daily["fear_greed_value"].ffill()
    daily["dgs10"]            = daily["dgs10"].ffill()

    daily.to_csv(PROCESSED_DIR / "daily_merged.csv", index=False)
    return daily


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Download all raw data, merge daily, and build weekly features."""
    print("Downloading BTC-USD...")
    btc = download_yahoo_series("BTC-USD", "btc_daily.csv")

    print("Downloading DXY...")
    dxy = download_yahoo_series("DX-Y.NYB", "dxy_daily.csv")

    print("Downloading Fear & Greed Index...")
    fear_greed = download_fear_greed()

    print("Downloading 10Y Treasury yield (DGS10)...")
    dgs10 = download_fred_series("DGS10", "dgs10", "dgs10_daily.csv")

    print("Merging daily data...")
    daily = build_daily_merged(btc, dxy, fear_greed, dgs10)

    print("Building weekly features...")
    weekly = build_weekly_features(daily)
    weekly.to_csv(PROCESSED_DIR / "weekly_features.csv", index=False)

    print("\nDone.")
    print(f"Daily rows:  {len(daily)}")
    print(f"Weekly rows: {len(weekly)}")
    print(f"Saved: {PROCESSED_DIR / 'daily_merged.csv'}")
    print(f"Saved: {PROCESSED_DIR / 'weekly_features.csv'}")


if __name__ == "__main__":
    main()