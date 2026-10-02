import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

def random_baseline(n: int, seed: int = 0) -> np.ndarray:
    """The paper's floor: a fair coin flip per example."""
    return np.random.default_rng(seed).integers(0, 2, size=n)

def majority_baseline(y_train, n: int) -> np.ndarray:
    """A stronger floor the paper omits: always predict the training set's majority class."""
    majority = int(np.mean(y_train) >= 0.5)
    return np.full(n, majority)

def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        'n': len(y_true),
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'roc_auc': np.nan,
    }
    # ROC-AUC needs scores and both classes present in the test set
    if y_score is not None and len(np.unique(y_true)) == 2:
        out['roc_auc'] = roc_auc_score(y_true, y_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
    return out
