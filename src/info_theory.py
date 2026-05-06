"""Information-theoretic metrics: KSG Mutual Information, KL Divergence, Surprisal."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
import librosa


# ── KSG Mutual Information estimator ─────────────────────────────────────────

def ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 4) -> float:
    """Estimate I(X; Y) using KSG Algorithm"""
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x # Convert 1-D arrays to 2-D arrays (scipy requires 2-D arrays)
    y = np.atleast_2d(y.T).T if y.ndim == 1 else y
    N = len(x)

    # Add tiny jitter to break ties from near-identical points
    rng = np.random.default_rng(42)
    x = x + rng.normal(0, 1e-10, x.shape)
    y = y + rng.normal(0, 1e-10, y.shape)

    xy = np.hstack([x, y])
    # Create KD-trees for the joint and marginal spaces
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf) 
    eps = dists[:, -1]
    eps = np.maximum(eps, 1e-12) # Keep last distance as epsilon
    # Count in marginal spaces
    nx = np.array([tree_x.query_ball_point(x[i], eps[i], p=np.inf, return_length=True) - 1 for i in range(N)])
    ny = np.array([tree_y.query_ball_point(y[i], eps[i], p=np.inf, return_length=True) - 1 for i in range(N)])

    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)

    mi = digamma(k) - np.mean(digamma(nx) + digamma(ny)) + digamma(N)
    return max(float(mi), 0.0)


# ── KL Divergence on STFT magnitude distributions ────────────────────────────

def stft_kl_divergence(clean: np.ndarray, estimated: np.ndarray,
                       n_fft: int = 1024, n_bins: int = 100) -> float:
    """Compute average D_KL(p_clean || p_est) over frequency bands.
    Uses histogram-based density estimation on STFT magnitudes.
    """
    S_c = np.abs(librosa.stft(clean.astype(np.float32), n_fft=n_fft))
    S_e = np.abs(librosa.stft(estimated.astype(np.float32), n_fft=n_fft))
    n_freq = S_c.shape[0]

    kl_sum = 0.0
    valid = 0
    for f in range(n_freq):
        lo = min(S_c[f].min(), S_e[f].min())
        hi = max(S_c[f].max(), S_e[f].max())
        if hi - lo < 1e-12:
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(S_c[f], bins=bins, density=True)
        q, _ = np.histogram(S_e[f], bins=bins, density=True)
        # Add small epsilon to avoid log(0)
        p = p + 1e-12
        q = q + 1e-12
        p = p / p.sum()
        q = q / q.sum()
        kl = float(np.sum(p * np.log(p / q)))
        kl_sum += kl
        valid += 1

    return kl_sum / max(valid, 1)
