# Member B: labels, models and evaluation

## How to run

```bash
python run_all.py              # full pipeline (needs data/raw/reddit_wsb.csv)
python run_all.py --skip-data  # reuse features_*.parquet, rerun only the stages below
```

| Stage | Script | Output |
|---|---|---|
| Prices (O3) | `src/data/fetch_prices.py` | `data/processed/prices_<T>.parquet` (FB is fetched as META) |
| Labels (O3) | `src/labels/build_labels.py` | `data/processed/labels_<T>.parquet` |
| Table 1 replication (O4–O6) | `python -m src.models.train_replication` | `reports/table1_replication.csv`, `figures/figure_3_newton_convergence.png` |
| Chronological evaluation (O7) | `python -m src.eval.evaluate_chronological` | `reports/results_chronological.csv`, `figures/leakage_random_vs_chronological.png` |

## Label definitions (`labels_<T>.parquet`)

One row per trading day k. Posts are mapped to the first trading day on or after their calendar date,
so weekend posts count toward Monday.

- `ret = Open[k+1] / Open[k] - 1`
- `y_dir = 1` if `Open[k+1] > Open[k]` (the paper's label)
- `y_sig = 1` if `|ret|` is above the median `|ret|` of the previous 20 known days. The paper never
  gives its "significant shift" threshold, and a fixed one is very ticker-dependent: 3% marks 71% of
  AMC days but 8% of FB days.
- `y_vol = 1` if day k+1's range `(High-Low)/Open` is above the trailing 20-day median range
  (volatility extra, E2). Only past data is used in both thresholds.

## Models

- `NewtonLogisticRegression`: from scratch in NumPy, Newton–Raphson with a bias column and
  tolerance `||Δθ|| < 1e-6`. Agrees with unregularised sklearn on 99.6–100% of test predictions.
- `build_paper_nn`: 3×256 ReLU layers with a sigmoid output, Adam lr 1e-3, 30 epochs, batch 16
  (64 for GME/TSLA). It uses sklearn's `MLPClassifier`, which is the same architecture without
  adding a TensorFlow/PyTorch dependency.
- `daily_majority_vote`: the paper's bot rule, turning post predictions into one call per day.

## Results so far

**Table 1 replication** (paper's random 80/10/10 post split, `y_dir`, mean of 5 seeds):

| Stock | NN (ours / paper) | LogReg (ours / paper) | Random (ours / paper) | Majority class |
|---|---|---|---|---|
| GME | 0.767 / 0.844 | 0.767 / 0.719 | 0.506 / 0.527 | 0.768 |
| AMC | 0.753 / 0.819 | 0.624 / N/A | 0.495 / 0.420 | 0.611 |
| TSLA | 0.557 / 0.640 | 0.613 / 0.740 | 0.515 / 0.532 | 0.616 |
| FB | 0.547 / 0.693 | 0.580 / 0.680 | 0.540 / 0.420 | 0.633 |

**Findings for the write-up and Q&A**

1. **The paper's accuracies are inflated by class imbalance.** Posts cluster on a few days: about
   5,000 GME posts fall on 29 Jan 2021 alone, and they all share one label. So at post level the
   majority class is already 61–77% accurate. On GME, our NN's 0.767 is exactly "always predict
   down". The paper only compared against a 50% coin flip.
2. **Leakage.** On AMC the NN beats the majority class on the random split (0.75 vs 0.61) because
   posts from the same days sit in train and test. With the chronological day split (train on
   28 Jan–28 Jun, test on the last ~25 trading days), every model is close to the majority
   baseline and ROC-AUC is about 0.5 for direction. See `results_chronological.csv` and
   `figures/leakage_random_vs_chronological.png`.
3. **Collinear sentiment features.** VADER's pos + neu + neg = 1 for every post, so with the bias
   column the design matrix is near-singular (cond(XᵀX) ≈ 5×10⁵). The weights are not unique,
   which is a likely reason the paper's Newton LogReg did not converge on AMC. Ours converges in
   5–9 iterations, while the paper's Fig. 3 shows 5–7.
4. **Small chronological test sets.** Post volume drops sharply after February (GME has only 116
   test posts over 25 days; FB has 36), so the day-level metrics are noisy. Report them with that
   caveat.
