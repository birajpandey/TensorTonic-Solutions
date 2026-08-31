import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    """
    Returns: np.ndarray x_{t-1} after one reverse diffusion step
    """
    x_t = np.asarray(x_t, dtype=float)
    epsilon_pred = np.asarray(epsilon_pred, dtype=float)
    betas = np.asarray(betas, dtype=float)
    
    if z is None:
        z = np.zeros_like(x_t)
    else:
        z = np.asarray(z)
    beta_t = betas[t-1]
    alpha_t = 1 - beta_t 
    alpha_bar_t = np.cumprod(1 - betas, dtype=float)[t - 1]
    sigma_t = np.sqrt(beta_t)
    resid = x_t -  epsilon_pred * (1 - alpha_t) / np.sqrt(1 - alpha_bar_t)
    coeff = 1 / np.sqrt(alpha_t) 
    return coeff * resid + sigma_t * z