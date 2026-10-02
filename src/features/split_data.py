#updated chronological split code
import os
import pandas as pd

def temporal_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Splits financial post data chronologically by sorting timestamps,
    preventing future data leakage while maintaining an exact train/test volume ratio.
    """
    # Sort strictly by timestamp/created_utc
    df_sorted = df.sort_values(by="created_utc").reset_index(drop=True)
    
    split_idx = int(len(df_sorted) * train_ratio)
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    cutoff_timestamp = df_sorted.loc[split_idx, 'created_utc']
    cutoff_trade_date = df_sorted.loc[split_idx, 'trade_date']
    
    return train_df, test_df, cutoff_trade_date

if __name__ == "__main__":
    sample_path = os.path.join("data", "processed", "features_GME.parquet")
    if os.path.exists(sample_path):
        df = pd.read_parquet(sample_path)
        train, test, cutoff = temporal_train_test_split(df)
        print("GME Temporal Split verification (Chronological by Post):")
        print(f"  Cutoff Trade Date: {cutoff}")
        print(f"  Train set posts: {len(train):,} ({len(train)/len(df):.1%})")
        print(f"  Test set posts:  {len(test):,} ({len(test)/len(df):.1%})")

# import os
# import pandas as pd

# def temporal_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8):
#     """
#     Splits financial post data chronologically by trade_date to prevent future leakage.
#     Returns train_df, test_df, and cutoff_date.
#     """
#     sorted_dates = sorted(df['trade_date'].dropna().unique())
#     split_idx = int(len(sorted_dates) * train_ratio)
#     cutoff_date = sorted_dates[split_idx]

#     train_dates = set(sorted_dates[:split_idx])
#     test_dates = set(sorted_dates[split_idx:])

#     train_df = df[df['trade_date'].isin(train_dates)].copy()
#     test_df = df[df['trade_date'].isin(test_dates)].copy()

#     return train_df, test_df, cutoff_date

# if __name__ == "__main__":
#     sample_path = os.path.join("data", "processed", "features_GME.parquet")
#     if os.path.exists(sample_path):
#         df = pd.read_parquet(sample_path)
#         train, test, cutoff = temporal_train_test_split(df)
#         print(f"GME Temporal Split verification:")
#         print(f"  Cutoff Trade Date: {cutoff}")
#         print(f"  Train set posts: {len(train):,} ({len(train)/len(df):.1%})")
#         print(f"  Test set posts:  {len(test):,} ({len(test)/len(df):.1%})")