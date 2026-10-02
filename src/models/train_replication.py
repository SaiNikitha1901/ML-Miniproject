"""
Objective O6: reproduce the paper's Table 1 under the paper's own protocol.

Per ticker: per-post feature vectors, a RANDOM 80/10/10 split of posts, standard scaling fitted on
train, Newton LogReg, the 3x256 NN and a random baseline. Repeated over 5 seeds (mean +- std).
Also produces Fig. 3 (Newton convergence) and checks the Newton solution against sklearn.

Run from the repo root:  python -m src.models.train_replication
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.eval.metrics import classification_metrics, random_baseline
from src.eval.plot_style import SERIES, apply_style
from src.models.dataset import FEATURES, TICKERS, load_dataset
from src.models.logreg_newton import NewtonLogisticRegression
from src.models.neural_net import build_paper_nn, fit_quietly

SEEDS = [0, 1, 2, 3, 4]
BATCH_SIZE = {'GME': 64, 'TSLA': 64, 'AMC': 16, 'FB': 16}  # paper Sec. 5.2

# Paper Table 1: (NN, LogReg, Baseline); AMC LogReg never converged
PAPER_TABLE1 = {
    'TSLA': (0.64, 0.740, 0.532),
    'AMC': (0.819, np.nan, 0.42),
    'GME': (0.844, 0.719, 0.527),
    'FB': (0.693, 0.68, 0.42),
}

def random_split(df, seed):
    """Paper's 80/10/10 split over posts, ignoring time. Posts from one day land on both sides."""
    train, rest = train_test_split(df, test_size=0.2, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, random_state=seed)
    return train, val, test

def run_ticker(ticker, target='y_dir'):
    df = load_dataset(ticker)
    rows, convergence = [], None
    for seed in SEEDS:
        train, val, test = random_split(df, seed)
        scaler = StandardScaler().fit(train[FEATURES])
        X_tr, X_va, X_te = (scaler.transform(d[FEATURES]) for d in (train, val, test))
        y_tr, y_va, y_te = train[target].values, val[target].values, test[target].values

        lr = NewtonLogisticRegression().fit(X_tr, y_tr)
        if seed == SEEDS[0]:
            convergence = lr.delta_norms
            # Cross-check against unregularised sklearn LogReg. VADER's pos+neu+neg sum to 1 for
            # every post, so those columns are collinear with the bias: the weights are not unique
            # (and the Hessian is near-singular, a likely cause of the paper's AMC failure), but
            # the predictions must agree.
            sk = LogisticRegression(C=np.inf, max_iter=5000).fit(X_tr, y_tr)
            agree = float(np.mean(sk.predict(X_te) == lr.predict(X_te)))
            design = np.hstack([np.ones((len(X_tr), 1)), X_tr])
            print(f"  [{ticker}] Newton converged={lr.converged} in {len(lr.delta_norms)} iters; "
                  f"agrees with sklearn on {agree:.1%} of test predictions; "
                  f"cond(X^T X) = {np.linalg.cond(design.T @ design):.1e}")

        nn = fit_quietly(build_paper_nn(BATCH_SIZE[ticker], seed), X_tr, y_tr)

        lr_m = classification_metrics(y_te, lr.predict(X_te), lr.predict_proba(X_te))
        nn_m = classification_metrics(y_te, nn.predict(X_te), nn.predict_proba(X_te)[:, 1])
        rb_m = classification_metrics(y_te, random_baseline(len(y_te), seed))
        rows.append({
            'seed': seed,
            'nn_val_acc': classification_metrics(y_va, nn.predict(X_va))['accuracy'],
            'nn_test_acc': nn_m['accuracy'], 'nn_test_auc': nn_m['roc_auc'],
            'lr_test_acc': lr_m['accuracy'] if lr.converged else np.nan,
            'lr_converged': lr.converged,
            'random_test_acc': rb_m['accuracy'],
            'majority_test_acc': max(y_te.mean(), 1 - y_te.mean()),
        })
    return pd.DataFrame(rows), convergence, len(df)

def plot_convergence(curves, path):
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (ticker, deltas) in enumerate(curves.items()):
        ax.semilogy(range(1, len(deltas) + 1), deltas, marker='o', color=SERIES[i], label=ticker)
    ax.axhline(1e-6, color='#52514e', linewidth=1, linestyle='--')
    ax.text(0.98, 1.6e-6, 'tolerance 1e-6', transform=ax.get_yaxis_transform(),
            ha='right', color='#52514e', fontsize=9)
    ax.set_title("Convergence of Newton's method (replication of Fig. 3)")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Norm of change in weight vector')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def main():
    os.makedirs(os.path.join("reports", "figures"), exist_ok=True)
    summary, curves = [], {}
    for ticker in TICKERS:
        print(f"Replicating Table 1 for {ticker}...")
        res, curves[ticker], n_posts = run_ticker(ticker)
        paper_nn, paper_lr, paper_rb = PAPER_TABLE1[ticker]
        summary.append({
            'stock': ticker, 'posts': n_posts,
            'nn_acc': res.nn_test_acc.mean(), 'nn_acc_std': res.nn_test_acc.std(),
            'lr_acc': res.lr_test_acc.mean(), 'lr_acc_std': res.lr_test_acc.std(),
            'lr_converged_runs': f"{int(res.lr_converged.sum())}/{len(res)}",
            'random_acc': res.random_test_acc.mean(),
            'majority_acc': res.majority_test_acc.mean(),
            'nn_auc': res.nn_test_auc.mean(),
            'paper_nn_acc': paper_nn, 'paper_lr_acc': paper_lr, 'paper_random_acc': paper_rb,
        })

    table = pd.DataFrame(summary)
    out_csv = os.path.join("reports", "table1_replication.csv")
    table.round(4).to_csv(out_csv, index=False)
    plot_convergence(curves, os.path.join("reports", "figures", "figure_3_newton_convergence.png"))

    print("\nTable 1 replication (random post-level split, label y_dir, mean of 5 seeds):")
    print(table[['stock', 'posts', 'nn_acc', 'paper_nn_acc', 'lr_acc', 'paper_lr_acc',
                 'random_acc', 'paper_random_acc', 'majority_acc']].round(3).to_string(index=False))
    print(f"\nSaved {out_csv} and reports/figures/figure_3_newton_convergence.png")

if __name__ == "__main__":
    main()
