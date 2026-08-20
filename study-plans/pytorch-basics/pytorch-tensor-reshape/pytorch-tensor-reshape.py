import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.Tensor(x)
    
    if op == "flatten":
        return torch.flatten(x).tolist()

    if op == "squeeze":
        return torch.squeeze(x).tolist()

    if op == "transpose":
        return torch.transpose(x, 0, 1).tolist()
    pass
