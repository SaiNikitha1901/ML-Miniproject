import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def build_ticker_matrix(ticker: str, df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer):
    # Filter posts mentioning this ticker
    stock_df = df[df['tickers'].apply(lambda ts: ticker in ts)].copy()
    print(f"Building features for {ticker} ({len(stock_df)} posts found)...")

    # Paper features: score, comments, length, sentiment
    stock_df['post_score'] = stock_df['score'].astype(float)
    stock_df['total_comms_num'] = stock_df['comms_num'].astype(float)
    stock_df['len_of_post'] = (stock_df['title'].str.len() + stock_df['body'].str.len()).astype(float)

    def get_vader(text):
        scores = analyzer.polarity_scores(text)
        return scores['pos'], scores['neu'], scores['neg']

    scores = stock_df['clean_text'].apply(get_vader)
    stock_df['sent_pos'] = [s[0] for s in scores]
    stock_df['sent_neu'] = [s[1] for s in scores]
    stock_df['sent_neg'] = [s[2] for s in scores]

    stock_df['trade_date'] = stock_df['timestamp'].dt.strftime('%Y-%m-%d')
    stock_df['created_utc'] = stock_df['timestamp'].astype('int64') // 10**9
    stock_df['post_id'] = stock_df['id'].astype(str)

    # Required interface columns for Member B
    cols = [
        'post_id', 'created_utc', 'trade_date',
        'post_score', 'total_comms_num', 'len_of_post',
        'sent_pos', 'sent_neu', 'sent_neg'
    ]

    out_path = os.path.join("data", "processed", f"features_{ticker}.parquet")
    stock_df[cols].to_parquet(out_path, index=False)
    print(f"Exported: {out_path}")

def main():
    pkl_path = os.path.join("data", "processed", "cleaned_wsb.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Run make_dataset.py first. Missing {pkl_path}")

    print("Loading cleaned dataset...")
    df = pd.read_pickle(pkl_path)
    analyzer = SentimentIntensityAnalyzer()

    for ticker in ['GME', 'AMC', 'TSLA', 'FB']:
        build_ticker_matrix(ticker, df, analyzer)

    print("\nAll 4 ticker feature tables generated successfully.")

if __name__ == "__main__":
    main()