"""Fetch daily BTC-USD market data and store it as raw CSV."""

import argparse
import os
import sys

import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "data")
RAW_FILE = "btc_raw.csv"


def fetch_btc(period="max", interval="1d"):
    """Download BTC-USD daily OHLCV data via yfinance."""
    import yfinance as yf

    df = yf.download(
        "BTC-USD",
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data for BTC-USD")

    # Newer yfinance versions return MultiIndex columns; flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    keep = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.dropna().sort_values("date").reset_index(drop=True)


def save_raw(df, data_dir=DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, RAW_FILE)
    df.to_csv(path, index=False)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="max")
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args(argv)

    df = fetch_btc(period=args.period)
    path = save_raw(df, args.data_dir)
    print(f"Saved {len(df)} rows ({df['date'].min().date()} to "
          f"{df['date'].max().date()}) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())