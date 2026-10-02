import os
import pandas as pd
import yfinance as yf

# FB was renamed to META in 2022; yfinance keeps the full price history under META
YF_SYMBOLS = {'GME': 'GME', 'AMC': 'AMC', 'TSLA': 'TSLA', 'FB': 'META'}

# Posts span 2021-01-28 to 2021-08-16. Start early so trailing windows (y_vol) have history,
# end late so the last post day still has a next trading day.
START, END = "2020-12-01", "2021-09-01"

def fetch_prices(ticker: str) -> pd.DataFrame:
    """Downloads daily OHLCV for one ticker (split-adjusted, so open-to-open ratios are consistent)."""
    raw = yf.download(YF_SYMBOLS[ticker], start=START, end=END,
                      auto_adjust=False, progress=False, multi_level_index=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    df.index.name = 'trade_date'
    return df.reset_index()

def main():
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    for ticker in YF_SYMBOLS:
        df = fetch_prices(ticker)
        out_path = os.path.join("data", "processed", f"prices_{ticker}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"Exported: {out_path} ({len(df)} trading days, {df.trade_date.min()} to {df.trade_date.max()})")

if __name__ == "__main__":
    main()
