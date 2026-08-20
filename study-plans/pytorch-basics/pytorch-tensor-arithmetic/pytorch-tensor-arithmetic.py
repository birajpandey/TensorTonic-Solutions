import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x = torch.tensor(x)
    y = torch.tensor(y)
    if op == "add":
        return torch.add(x, y).tolist()

    if op == "multiply":
        return torch.mul(x, y).tolist()

    if op == "matmul":
        return torch.matmul(x, y).tolist()

    if op == "power":
        return torch.pow(x, y).tolist()

    if op == "max":
        return torch.max(x, y).tolist()
    pass