import numpy as np

class NewtonLogisticRegression:
    """
    Logistic regression fitted with Newton's method, as in the paper (Sec. 4).

    theta <- theta - H^-1 grad, with
      grad = X^T (h - y) / m
      H    = X^T S X / m,  S = diag(h (1 - h))
    A bias column of 1s is prepended (the paper's 7th feature). Inputs should already be scaled.
    """

    def __init__(self, max_iter: int = 50, tol: float = 1e-6, l2: float = 0.0):
        self.max_iter = max_iter
        self.tol = tol
        self.l2 = l2  # 0 reproduces the paper; a small value stabilises near-separable data
        self.theta = None
        self.delta_norms = []  # ||theta_t - theta_{t-1}|| per iteration, for Fig. 3
        self.converged = False

    @staticmethod
    def _add_bias(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X = self._add_bias(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        m, n = X.shape
        self.theta = np.zeros(n)
        self.delta_norms = []
        self.converged = False
        reg = self.l2 * np.eye(n)
        reg[0, 0] = 0.0  # never penalise the bias

        for _ in range(self.max_iter):
            h = self._sigmoid(X @ self.theta)
            grad = X.T @ (h - y) / m + reg @ self.theta
            hessian = (X.T * (h * (1 - h))) @ X / m + reg
            try:
                step = np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                # Singular Hessian: the failure mode the paper hit for AMC
                break
            self.theta = self.theta - step
            delta = float(np.linalg.norm(step))
            self.delta_norms.append(delta)
            if not np.isfinite(delta):
                break
            if delta < self.tol:
                self.converged = True
                break
        return self

    def predict_proba(self, X):
        return self._sigmoid(self._add_bias(np.asarray(X, dtype=float)) @ self.theta)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
