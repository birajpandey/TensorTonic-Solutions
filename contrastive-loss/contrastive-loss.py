import numpy as np

def contrastive_loss(a: list, b: list, y: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """Return contrastive loss with the selected reduction."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    y = np.asarray(y, float)
    differences = a - b
    if differences.ndim == 1:
        differences = differences[None, :]

    d = np.linalg.norm(differences, axis=1)
    loss = y * d ** 2 + (1 - y) * np.maximum(0, margin - d) ** 2
    if reduction == "mean":
        return float(np.mean(loss))
    if reduction == "sum":
        return float(np.sum(loss))
