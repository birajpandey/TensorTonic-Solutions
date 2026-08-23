import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Return the mean multiclass cross-entropy loss.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.array(y_pred, float)
    N = len(y_pred)
    p = y_pred[np.arange(N), y_true]
    return -np.log(p).sum() / N