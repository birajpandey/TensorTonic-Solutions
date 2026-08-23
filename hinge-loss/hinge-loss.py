import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """Return binary hinge loss with the selected reduction."""
    # Write code here
    y_true = np.array(y_true, float)
    y_score = np.array(y_score, float)
    resid = margin - (y_score * y_true)
    loss = np.where(resid > 0, resid, 0)
    if reduction == "mean":
        return float(np.mean(loss))

    if reduction == "sum":
        return float(np.sum(loss))