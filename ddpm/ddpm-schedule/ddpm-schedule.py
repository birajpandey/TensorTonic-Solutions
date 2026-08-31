import numpy as np

def linear_beta_schedule(T, beta_1=0.0001, beta_T=0.02):
    """
    Linear noise schedule from beta_1 to beta_T.
    Returns list of floats rounded to 6 decimals.
    """
    return np.linspace(beta_1, beta_T, T)

def cosine_alpha_bar_schedule(T, s=0.008):
    """
    Cosine schedule for alpha_bar (cumulative signal retention).
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """
    steps = np.arange(T + 1) / T
    f_t = np.cos((steps + s)/ (1+ s) * np.pi/ 2) ** 2 
    alpha_bars = np.clip(f_t[1:] / f_t[0], 0.0001, 0.9999)
    return [round(float(v), 6) for v in alpha_bars]
    

def alpha_bar_to_betas(alpha_bars):
    """
    Convert alpha_bar schedule to beta schedule.
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """
    ab = np.array(alpha_bars, dtype=float)
    ab_prev = np.concatenate([[1.0], ab[:-1]])
    betas = 1 - ab / ab_prev
    betas = np.clip(betas, 0.0001, 0.9999)
    return [round(float(v), 6) for v in betas]