import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def clean_text(text: str) -> str:
    """Cleans post body and title following the paper's exact steps."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)          # URLs
    text = re.sub(r'(@\w+|u/\w+)', '', text)                      # Handles
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)                      # Special chars
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)                     # Single chars
    text = re.sub(r'\s+', ' ', text).strip()                      # Extra whitespace
    return text

def extract_tickers(title: str, body: str) -> list:
    """Detects target stock tickers using word boundaries."""
    combined = f"{str(title)} {str(body)}"
    patterns = {
        'GME': r'(\$GME\b|\bGME\b|GAMESTOP)',
        'AMC': r'(\$AMC\b|\bAMC\b)',
        'TSLA': r'(\$TSLA\b|\bTSLA\b|TESLA)',
        'FB': r'(\$FB\b|\bFB\b|FACEBOOK|\$META\b|\bMETA\b)',
        'AAPL': r'(\$AAPL\b|\bAAPL\b|APPLE)',
        'MSFT': r'(\$MSFT\b|\bMSFT\b|MICROSOFT)',
        'GOOGL': r'(\$GOOGL\b|\$GOOG\b|\bGOOGL\b|ALPHABET)'
    }
    found = []
    for ticker, pat in patterns.items():
        if re.search(pat, combined, flags=re.IGNORECASE):
            found.append(ticker)
    return found

def main():
    raw_path = "data/raw/reddit_wsb.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}. Place reddit_wsb.csv in data/raw/")

    print("Loading raw CSV...")
    df = pd.read_csv(raw_path)
    df['title'] = df['title'].fillna('')
    df['body'] = df['body'].fillna('')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    print("Cleaning text and extracting tickers...")
    df['clean_text'] = (df['title'] + " " + df['body']).apply(clean_text)
    df['tickers'] = df.apply(lambda r: extract_tickers(r['title'], r['body']), axis=1)

    # Plot Figure 2: Mentions Histogram
    all_mentions = [t for sublist in df['tickers'] for t in sublist]
    mention_counts = pd.Series(all_mentions).value_counts()
    
    os.makedirs("reports/figures", exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=mention_counts.index, y=mention_counts.values, color="mediumturquoise")
    plt.title("Mentions of various stocks in the dataset (Replication Fig 2)")
    plt.xlabel("Stock")
    plt.ylabel("Mentions")
    plt.tight_layout()
    plt.savefig("reports/figures/figure_2_mentions_histogram.png", dpi=300)
    print("Saved figure to reports/figures/figure_2_mentions_histogram.png")

    os.makedirs("data/processed", exist_ok=True)
    df.to_pickle("data/processed/cleaned_wsb.pkl")
    print("Saved intermediate dataset to data/processed/cleaned_wsb.pkl")

if __name__ == "__main__":
    main()