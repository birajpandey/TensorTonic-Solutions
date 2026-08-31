import numpy as np
import torch

def get_alpha_bar(betas):
    """
    Compute cumulative product of (1 - beta).
    Returns list of floats rounded to 6 decimals.
    """
    return np.cumprod(1 - np.asarray(betas), dtype=float)

def forward_diffusion(x_0, t, betas, epsilon):
    """
    Returns: tuple of (np.ndarray x_t, np.ndarray epsilon) with same shape as x_0
    """
    x_0 = np.asarray(x_0, dtype=float)
    epsilon = np.asarray(epsilon, dtype=float)
    alphas = get_alpha_bar(betas)
    alpha_bar = alphas[t - 1]
    x_t = x_0 * np.sqrt(alpha_bar) + np.sqrt(1 - alpha_bar) * epsilon
    return x_t