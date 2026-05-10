from pathlib import Path
import requests
import numpy as np
import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2018-02-01"


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def download_yahoo_series(ticker: str, out_name: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=START_DATE,
        progress=False,
        auto_adjust=False,
    )
    df = flatten_columns(df).reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df.to_csv(RAW_DIR / out_name, index=False)
    return df


def download_fear_greed() -> pd.DataFrame:
    url = "https://api.alternative.me/fng/?limit=0"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()

    df = pd.DataFrame(payload["data"])
    df["Date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
    df["fear_greed_value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["Date", "fear_greed_value", "value_classification"]].sort_values("Date")
    df.to_csv(RAW_DIR / "fear_greed_daily.csv", index=False)
    return df


def download_fred_series(series_id: str, value_name: str, out_name: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["Date", value_name]
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df.to_csv(RAW_DIR / out_name, index=False)
    return df


def build_daily_merged(
    btc: pd.DataFrame,
    dxy: pd.DataFrame,
    fear_greed: pd.DataFrame,
    dgs10: pd.DataFrame,
) -> pd.DataFrame:
    btc = btc.rename(
        columns={
            "Close": "btc_close",
            "Volume": "btc_volume",
        }
    )[["Date", "btc_close", "btc_volume"]]

    dxy = dxy.rename(columns={"Close": "dxy_close"})[["Date", "dxy_close"]]

    daily = (
        btc.merge(dxy, on="Date", how="inner")
           .merge(fear_greed[["Date", "fear_greed_value"]], on="Date", how="left")
           .merge(dgs10, on="Date", how="left")
           .sort_values("Date")
           .reset_index(drop=True)
    )

    daily["fear_greed_value"] = daily["fear_greed_value"].ffill()
    daily["dgs10"] = daily["dgs10"].ffill()

    daily.to_csv(PROCESSED_DIR / "daily_merged.csv", index=False)
    return daily


def build_weekly_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy().set_index("Date")

    weekly = pd.DataFrame({
        "btc_close": df["btc_close"].resample("W-SUN").last(),
        "btc_volume": df["btc_volume"].resample("W-SUN").sum(),
        "dxy_close": df["dxy_close"].resample("W-SUN").last(),
        "fear_greed_value": df["fear_greed_value"].resample("W-SUN").mean(),
        "dgs10": df["dgs10"].resample("W-SUN").last(),
    }).dropna().reset_index()

    # Core market features
    weekly["btc_log_return"] = np.log(weekly["btc_close"]).diff()
    weekly["btc_volume_log_change"] = np.log(weekly["btc_volume"]).diff()
    weekly["btc_4w_vol"] = weekly["btc_log_return"].rolling(4).std()

    # Macro features
    weekly["dxy_log_return"] = np.log(weekly["dxy_close"]).diff()
    weekly["dgs10_change"] = weekly["dgs10"].diff()

    # Sentiment / psychology features
    weekly["btc_momentum_4w"] = np.log(weekly["btc_close"] / weekly["btc_close"].shift(4))
    rolling_max_12w = weekly["btc_close"].rolling(12).max()
    weekly["btc_drawdown_12w"] = (weekly["btc_close"] / rolling_max_12w) - 1.0

    # Trader / technical-analysis features
    weekly["btc_sma_4"] = weekly["btc_close"].rolling(4).mean()
    weekly["btc_sma_12"] = weekly["btc_close"].rolling(12).mean()
    weekly["btc_ma_spread_4_12"] = (weekly["btc_sma_4"] / weekly["btc_sma_12"]) - 1.0
    weekly["btc_price_vs_sma12"] = (weekly["btc_close"] / weekly["btc_sma_12"]) - 1.0

    # Target for next-week direction
    weekly["next_week_up"] = (weekly["btc_log_return"].shift(-1) > 0).astype(int)

    weekly = weekly.dropna().reset_index(drop=True)
    weekly.to_csv(PROCESSED_DIR / "weekly_features.csv", index=False)
    return weekly


def main() -> None:
    print("Downloading BTC...")
    btc = download_yahoo_series("BTC-USD", "btc_daily.csv")

    print("Downloading DXY...")
    dxy = download_yahoo_series("DX-Y.NYB", "dxy_daily.csv")

    print("Downloading Fear & Greed...")
    fear_greed = download_fear_greed()

    print("Downloading 10Y Treasury yield...")
    dgs10 = download_fred_series("DGS10", "dgs10", "dgs10_daily.csv")

    print("Merging daily data...")
    daily = build_daily_merged(btc, dxy, fear_greed, dgs10)

    print("Building weekly features...")
    weekly = build_weekly_features(daily)

    print("\nDone.")
    print(f"Daily rows:  {len(daily)}")
    print(f"Weekly rows: {len(weekly)}")
    print(f"Saved: {PROCESSED_DIR / 'daily_merged.csv'}")
    print(f"Saved: {PROCESSED_DIR / 'weekly_features.csv'}")


if __name__ == "__main__":
    main()