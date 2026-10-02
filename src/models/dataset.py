import os
import pandas as pd

TICKERS = ['GME', 'AMC', 'TSLA', 'FB']

# The paper's 6 data features; the 7th (bias column of 1s) is added inside the Newton LogReg.
FEATURES = ['post_score', 'total_comms_num', 'len_of_post', 'sent_pos', 'sent_neu', 'sent_neg']
TARGETS = ['y_dir', 'y_sig', 'y_vol']

PROCESSED = os.path.join("data", "processed")

def load_dataset(ticker: str) -> pd.DataFrame:
    """
    Joins Member A's per-post features with Member B's per-day labels.

    A post's trade_date is its calendar date, which can be a weekend or holiday. Each post is mapped
    to the first trading day on or after that date (a Saturday post counts toward Monday), then
    takes that day's labels, i.e. what happens at the following open.
    """
    features = pd.read_parquet(os.path.join(PROCESSED, f"features_{ticker}.parquet"))
    labels = pd.read_parquet(os.path.join(PROCESSED, f"labels_{ticker}.parquet"))

    features = features.rename(columns={'trade_date': 'post_date'})
    features['post_date'] = pd.to_datetime(features['post_date'])
    labels = labels.copy()
    labels['label_date'] = pd.to_datetime(labels['trade_date'])

    merged = pd.merge_asof(
        features.sort_values('post_date'),
        labels.sort_values('label_date'),
        left_on='post_date', right_on='label_date', direction='forward',
    )
    # Drop posts that fall outside the labelled range (e.g. a stray 2020 TSLA post before the
    # trailing-window warm-up, or posts after the last labelled day)
    merged = merged.dropna(subset=['label_date'])
    merged = merged[(merged['label_date'] - merged['post_date']).dt.days <= 5]
    for t in TARGETS:
        merged[t] = merged[t].astype(int)
    return merged.drop(columns=['label_date']).reset_index(drop=True)
