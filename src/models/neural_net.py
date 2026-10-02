import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

def build_paper_nn(batch_size: int = 16, seed: int = 42) -> MLPClassifier:
    """
    The paper's network (Sec. 4 and 5.2): 3 linear layers of 256 units with ReLU and a sigmoid head,
    Adam with lr 0.001, 30 epochs, batch 16 (64 for GME and TSLA).

    sklearn's MLPClassifier builds exactly this for binary targets (its output unit is a logistic
    sigmoid trained on log-loss), so we avoid adding a TensorFlow/PyTorch dependency.
    """
    return MLPClassifier(
        hidden_layer_sizes=(256, 256, 256),
        activation='relu',
        solver='adam',
        learning_rate_init=1e-3,
        batch_size=batch_size,
        max_iter=30,           # epochs
        shuffle=True,
        random_state=seed,
    )

def fit_quietly(model, X, y):
    # A fixed 30-epoch budget always ends "unconverged" in sklearn's eyes; that is intended
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(X, y)

def daily_majority_vote(trade_dates, post_preds) -> pd.Series:
    """
    The paper's bot rule: a security goes up on day k if most of that day's posts say so.
    Ties count as 'up'. Returns one 0/1 prediction per trade_date.
    """
    votes = pd.DataFrame({'trade_date': np.asarray(trade_dates), 'pred': np.asarray(post_preds)})
    return votes.groupby('trade_date')['pred'].mean().ge(0.5).astype(int)
