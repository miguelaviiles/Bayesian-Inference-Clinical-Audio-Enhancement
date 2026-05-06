"""SIR Particle Filter"""

import numpy as np
from scipy.special import gammaln
from tqdm import tqdm
from kalman import build_companion


def systematic_resample(weights: np.ndarray) -> np.ndarray:
    N = len(weights)
    cdf = np.cumsum(weights)
    u0 = np.random.uniform(0, 1.0 / N)
    u = u0 + np.arange(N) / N
    indices = np.searchsorted(cdf, u)
    return np.clip(indices, 0, N - 1)


def _t_logpdf_const(df: float, scale: float) -> tuple[float, float, float]:
    """Pre-compute constants for Student-t log-pdf."""
    c = (gammaln(0.5 * (df + 1)) - gammaln(0.5 * df) - 0.5 * np.log(df * np.pi) - np.log(scale))
    half_dfp1 = 0.5 * (df + 1)
    inv_df_scale2 = 1.0 / (df * scale * scale)
    return float(c), float(half_dfp1), float(inv_df_scale2)


def estimate_t_params(noise_signal: np.ndarray) -> tuple[float, float]:
    """Fit a Student-t to the observation noise"""
    med = np.median(noise_signal)
    mad = np.median(np.abs(noise_signal - med))
    scale = mad * 1.4826
    if scale < 1e-12:
        return 30.0, max(float(np.std(noise_signal)), 1e-8)
    kurt = float(np.mean(((noise_signal - med) / scale) ** 4))
    if kurt <= 3.0:
        df = 30.0
    else:
        df = max(4.0, 6.0 / (kurt - 3.0) + 4.0)
    return df, float(scale)


def particle_filter_signal(noisy: np.ndarray, ar_frames: list[dict], df: float, obs_scale: float, order: int = 10, n_particles: int = 200, frame_len: int = 320, hop: int = 160) -> np.ndarray:
    """SIR particle filter with AR transition proposal and Student-t likelihood."""
    N = len(noisy)
    p = order
    M = n_particles
    n_ar = len(ar_frames)

    x_hat = np.zeros(N, dtype=np.float64)
    particles = np.zeros((M, p))
    weights = np.ones(M) / M

    H = np.zeros(p)
    H[0] = 1.0

    c, half_dfp1, inv_df_s2 = _t_logpdf_const(df, obs_scale)

    for n in tqdm(range(N), desc="Particle", leave=False, miniters=50000):
        frame_idx = min(n // hop, n_ar - 1)
        af = ar_frames[frame_idx]
        A = build_companion(af["a"])
        Q_excite = af["Q"]

        noise_vec = np.zeros((M, p)) 
        noise_vec[:, 0] = np.random.normal(0, np.sqrt(Q_excite), M) # Sample from the noise distribution
        particles = (A @ particles.T).T + noise_vec # Update the particles

        pred = particles @ H
        res = noisy[n] - pred
        log_w = c - half_dfp1 * np.log1p(res * res * inv_df_s2)
        log_w -= log_w.max()
        w = np.exp(log_w)
        w_sum = w.sum()
        if w_sum < 1e-30:
            weights[:] = 1.0 / M
        else:
            weights = w / w_sum

        x_hat[n] = weights @ particles[:, 0] # Update the estimated clean signal

        ess = 1.0 / (weights ** 2).sum()
        if ess < M / 2: # Resample if the ESS is less than half the number of particles
            idx = systematic_resample(weights)
            particles = particles[idx]
            weights = np.ones(M) / M

    return x_hat.astype(np.float32) # Return the estimated clean signal
