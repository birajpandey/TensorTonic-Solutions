import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        logits = X @ w + b 
        preds = _sigmoid(logits)
        resid = preds - y
        grad_w = (X.T @ resid) / N
        grad_b = resid.sum() / N 

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b