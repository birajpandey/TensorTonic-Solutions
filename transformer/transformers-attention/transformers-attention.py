import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    d_k = Q.shape[-1]
    h = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    alpha = F.softmax(h, dim=-1)
    output = torch.matmul(alpha, V)
    return output
