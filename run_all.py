"""
Runs the full pipeline from the repo root:  python run_all.py
Needs data/raw/reddit_wsb.csv (Kaggle: gpreda/reddit-wallstreetsbets-posts) and internet for yfinance.
Pass --skip-data to reuse existing features_*.parquet and only rerun Member B's stages.
"""
import subprocess
import sys

DATA_STAGES = [
    ["src/data/make_dataset.py"],            # Member A: clean text, tickers, Fig. 2
    ["src/features/build_features.py"],      # Member A: VADER + per-post features
]
MODEL_STAGES = [
    ["src/data/fetch_prices.py"],            # Member B: yfinance OHLC
    ["src/labels/build_labels.py"],          # Member B: y_dir / y_sig / y_vol
    ["-m", "src.models.train_replication"],  # Member B: Table 1 + Fig. 3
    ["-m", "src.eval.evaluate_chronological"],  # Member B: leakage-free evaluation
]

def main():
    stages = MODEL_STAGES if "--skip-data" in sys.argv else DATA_STAGES + MODEL_STAGES
    for args in stages:
        print(f"\n=== python {' '.join(args)} ===", flush=True)
        subprocess.run([sys.executable, *args], check=True)

if __name__ == "__main__":
    main()
