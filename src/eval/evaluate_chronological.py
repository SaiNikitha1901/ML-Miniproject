"""
Objective O7: re-evaluate the same models without the paper's leakage.

The paper splits POSTS at random, but every post from one trading day shares that day's label, so
the same days appear in train and test. Here the first 80% of trading days train and the last 20%
test, so no day (and no future information) crosses the boundary. Unlike a post-level cutoff,
splitting on whole days also keeps the boundary day from straddling both sides.

Reported per ticker x target x model, at post level and at day level (the paper's majority vote),
against random and majority-class baselines. Also runs the paper's random split for the same
target so the leakage gap is visible side by side.

Run from the repo root:  python -m src.eval.evaluate_chronological
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.eval.metrics import classification_metrics, majority_baseline, random_baseline
from src.eval.plot_style import INK, SERIES, apply_style
from src.models.dataset import FEATURES, TARGETS, TICKERS, load_dataset
from src.models.logreg_newton import NewtonLogisticRegression
from src.models.neural_net import build_paper_nn, daily_majority_vote, fit_quietly
from src.models.train_replication import BATCH_SIZE

TRAIN_RATIO = 0.8
SEED = 0

def chronological_split_by_day(df, train_ratio=TRAIN_RATIO):
    """First train_ratio of trading days -> train, the rest -> test. Returns train, test, cutoff."""
    days = np.sort(df['trade_date'].unique())
    cutoff = days[int(len(days) * train_ratio)]
    return df[df['trade_date'] < cutoff], df[df['trade_date'] >= cutoff], cutoff

def random_post_split(df, seed=SEED):
    """The paper's protocol (80/20 over posts, ignoring time) for the side-by-side comparison."""
    return train_test_split(df, test_size=0.2, random_state=seed)

def day_level(test, preds, scores, target):
    """Collapse post predictions to one per day (majority vote) and score against the day label."""
    day_pred = daily_majority_vote(test['trade_date'], preds)
    day_score = pd.Series(scores, index=test.index).groupby(test['trade_date']).mean()
    day_true = test.groupby('trade_date')[target].first()
    return classification_metrics(day_true.loc[day_pred.index], day_pred, day_score.loc[day_pred.index])

def evaluate(train, test, ticker, target):
    scaler = StandardScaler().fit(train[FEATURES])
    X_tr, X_te = scaler.transform(train[FEATURES]), scaler.transform(test[FEATURES])
    y_tr, y_te = train[target].values, test[target].values

    lr = NewtonLogisticRegression().fit(X_tr, y_tr)
    nn = fit_quietly(build_paper_nn(BATCH_SIZE[ticker], SEED), X_tr, y_tr)

    models = {
        'random': (random_baseline(len(y_te), SEED), np.full(len(y_te), 0.5)),
        'majority': (majority_baseline(y_tr, len(y_te)), np.full(len(y_te), 0.5)),
        'logreg_newton': (lr.predict(X_te), lr.predict_proba(X_te)),
        'nn_3x256': (nn.predict(X_te), nn.predict_proba(X_te)[:, 1]),
    }
    rows = []
    for name, (pred, score) in models.items():
        # Constant scores carry no ranking information, so baselines get no ROC-AUC
        has_score = name not in ('random', 'majority')
        post = classification_metrics(y_te, pred, score if has_score else None)
        day = day_level(test, pred, score, target)
        if not has_score:
            day['roc_auc'] = np.nan
        for level, m in (('post', post), ('day', day)):
            rows.append({'model': name, 'level': level, **m})
    return rows

def plot_leakage_gap(results, path):
    """Post-level NN accuracy on y_dir: paper's random split vs chronological split, each with its
    own majority-class baseline, so imbalance and leakage are both visible."""
    apply_style()
    sel = results[(results.target == 'y_dir') & (results.level == 'post')]
    acc = sel.pivot_table(index='ticker', columns=['split', 'model'], values='accuracy').loc[TICKERS]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x, w = np.arange(len(TICKERS)), 0.36
    for offset, split, colour, label in ((-w / 2 - 0.01, 'random_posts', SERIES[0], "Paper's random post split"),
                                         (w / 2 + 0.01, 'chronological', SERIES[1], 'Chronological day split (ours)')):
        bars = ax.bar(x + offset, acc[(split, 'nn_3x256')], w, color=colour, label=label)
        ax.bar_label(bars, fmt='%.2f', label_type='center', color='white', fontsize=9, fontweight='bold')
        ax.scatter(x + offset, acc[(split, 'majority')], marker='_', s=550, color=INK, linewidths=2, zorder=3,
                   label='Majority-class baseline' if split == 'chronological' else None)
    ax.set_xticks(x, TICKERS)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Test accuracy (post level)')
    ax.set_title('3x256 NN on next-open direction: random vs chronological split')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8.5)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def main():
    all_rows = []
    for ticker in TICKERS:
        df = load_dataset(ticker)
        train, test, cutoff = chronological_split_by_day(df)
        print(f"{ticker}: train {train.trade_date.nunique()} days / {len(train):,} posts, "
              f"test {test.trade_date.nunique()} days / {len(test):,} posts (cutoff {cutoff})")
        r_train, r_test = random_post_split(df)
        for target in TARGETS:
            for split, (tr, te) in (('chronological', (train, test)), ('random_posts', (r_train, r_test))):
                if tr[target].nunique() < 2:
                    continue
                for row in evaluate(tr, te, ticker, target):
                    all_rows.append({'ticker': ticker, 'target': target, 'split': split, **row})

    results = pd.DataFrame(all_rows)
    os.makedirs(os.path.join("reports", "figures"), exist_ok=True)
    out_csv = os.path.join("reports", "results_chronological.csv")
    results.round(4).to_csv(out_csv, index=False)
    plot_leakage_gap(results, os.path.join("reports", "figures", "leakage_random_vs_chronological.png"))

    view = results[(results.split == 'chronological')]
    for level in ('post', 'day'):
        print(f"\nChronological split, {level} level (accuracy / macro-F1 / ROC-AUC):")
        t = view[view.level == level].copy()
        t['cell'] = t.apply(lambda r: f"{r.accuracy:.2f} / {r.macro_f1:.2f} / "
                                      + ('  - ' if np.isnan(r.roc_auc) else f"{r.roc_auc:.2f}"), axis=1)
        print(t.pivot_table(index=['target', 'ticker'], columns='model', values='cell', aggfunc='first')
               [['random', 'majority', 'logreg_newton', 'nn_3x256']].to_string())
    print(f"\nSaved {out_csv} and reports/figures/leakage_random_vs_chronological.png")

if __name__ == "__main__":
    main()
