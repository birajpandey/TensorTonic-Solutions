import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)
    
    if method == 'relu':
        return torch.clamp(x, 0)

    if method == 'sigmoid':
        return torch.pow(1 + torch.exp(-x), -1)

    if method == "tanh":
        num = torch.exp(x) - torch.exp(-x)
        den = torch.exp(x) + torch.exp(-x)
        return num / den

    if method == "leaky_relu":
        return torch.where(x > 0, x, 0.01 * x)
    pass