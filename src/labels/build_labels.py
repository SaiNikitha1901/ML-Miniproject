import os
import pandas as pd

TICKERS = ['GME', 'AMC', 'TSLA', 'FB']

# Trailing window for the adaptive thresholds of y_sig and y_vol (only past days are used).
WINDOW = 20

def build_labels(prices: pd.DataFrame, sig_threshold: float = None) -> pd.DataFrame:
    """
    One row per trading day k, labelling what happens at the NEXT trading day k+1:
      ret   = Open[k+1] / Open[k] - 1
      y_dir = 1 if Open[k+1] > Open[k]                          (paper's label v1)
      y_sig = 1 if |ret| > threshold                            (paper's label v2, threshold ours)
      y_vol = 1 if day k+1's range (High-Low)/Open is above the
              median range of the WINDOW days up to and including k      (volatility extra, E2)

    The paper never states its "significant shift" threshold. A fixed one is very ticker-dependent
    (3% marks 71% of AMC days but 8% of FB days), so by default the threshold is the median |ret|
    of the previous WINDOW known returns, which keeps classes near-balanced without using future
    data. Pass sig_threshold (e.g. 0.02) for a fixed threshold instead.
    """
    df = prices.sort_values('trade_date').reset_index(drop=True)
    df['next_open'] = df['open'].shift(-1)
    df['ret'] = df['next_open'] / df['open'] - 1

    # ret on day j is only known at Open[j+1], so on day k the latest known return is row k-1
    if sig_threshold is None:
        sig_ref = df['ret'].abs().shift(1).rolling(WINDOW, min_periods=5).median()
    else:
        sig_ref = pd.Series(sig_threshold, index=df.index)

    day_range = (df['high'] - df['low']) / df['open']
    vol_ref = day_range.rolling(WINDOW, min_periods=5).median()
    next_range = day_range.shift(-1)

    df['y_dir'] = (df['ret'] > 0).astype(int)
    df['y_sig'] = (df['ret'].abs() > sig_ref).astype(int)
    df['y_vol'] = (next_range > vol_ref).astype(int)

    # The last day has no k+1, and the first few have no trailing reference
    valid = df['next_open'].notna() & sig_ref.notna() & vol_ref.notna()
    df = df[valid]
    return df[['trade_date', 'open', 'next_open', 'ret', 'y_dir', 'y_sig', 'y_vol']].reset_index(drop=True)

def main():
    for ticker in TICKERS:
        prices_path = os.path.join("data", "processed", f"prices_{ticker}.parquet")
        if not os.path.exists(prices_path):
            raise FileNotFoundError(f"Run fetch_prices.py first. Missing {prices_path}")
        labels = build_labels(pd.read_parquet(prices_path))
        out_path = os.path.join("data", "processed", f"labels_{ticker}.parquet")
        labels.to_parquet(out_path, index=False)
        print(f"Exported: {out_path} ({len(labels)} days) | "
              f"up={labels.y_dir.mean():.1%} sig={labels.y_sig.mean():.1%} vol={labels.y_vol.mean():.1%}")

if __name__ == "__main__":
    main()
