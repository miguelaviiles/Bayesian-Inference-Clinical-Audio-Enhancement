"""AR-based Kalman Filter for speech enhancement."""

import numpy as np
import librosa
from tqdm import tqdm


def fit_ar_per_frame(clean: np.ndarray, order: int = 10, frame_len: int = 320, hop: int = 160) -> list[dict]: # 50% overlap, 20ms frames
    """Burg's method via librosa.lpc. Returns a list of dicts with AR Coefficients and 'Q' (excitation variance)."""
    n_frames = 1 + (len(clean) - frame_len) // hop
    frames = []
    for i in range(n_frames):
        seg = clean[i * hop: i * hop + frame_len].astype(np.float64) 
        if np.max(np.abs(seg)) < 1e-8:
            frames.append({"a": np.zeros(order), "Q": 1e-10}) # Silence frames have no AR coefficients
            continue
        a_full = librosa.lpc(seg, order=order)  # [1, -a1, -a2, ...]
        a = -a_full[1:]                          # [a1, a2, ..., ap]
        residual = seg.copy()
        for k in range(order):
            residual[k + 1:] -= a[k] * seg[: len(seg) - k - 1] # e_t = x_t - a1*x_{t-1} - a2*x_{t-2} - ... - a_p*x_{t-p}
        Q = float(np.var(residual))
        frames.append({"a": a, "Q": max(Q, 1e-12)})
    return frames # List of dicts with AR Coefficients and 'Q' (excitation variance)


def build_companion(a: np.ndarray) -> np.ndarray:
    p = len(a)
    A = np.zeros((p, p))
    A[0, :] = a
    if p > 1:
        A[1:, :-1] = np.eye(p - 1)
    return A


def kalman_filter_signal(noisy: np.ndarray, ar_frames: list[dict], R: float, order: int = 10, frame_len: int = 320, hop: int = 160) -> np.ndarray:
    """Apply frame-adaptive Kalman filter to a noisy signal. Returns the estimated clean signal."""
    N = len(noisy)
    x_hat = np.zeros(N, dtype=np.float64)
    p = order

    H = np.zeros((1, p))
    H[0, 0] = 1.0

    s = np.zeros(p)
    P = np.eye(p) * 1.0

    n_frames = len(ar_frames)

    for n in tqdm(range(N), desc="Kalman", leave=False, miniters=50000): # Iterate over the noisy signal
        frame_idx = min(n // hop, n_frames - 1)
        af = ar_frames[frame_idx] # AR Coefficients and 'Q' (excitation variance) for the current frame
        A = build_companion(af["a"])
        Q_mat = np.zeros((p, p))
        Q_mat[0, 0] = af["Q"]

        # Predict
        s_pred = A @ s
        P_pred = A @ P @ A.T + Q_mat

        # Update
        z = noisy[n]
        y = z - H @ s_pred                      # innovation
        S_inn = (H @ P_pred @ H.T + R)[0, 0]    # innovation covariance
        K = (P_pred @ H.T) / S_inn              # Kalman gain
        s = s_pred + (K * y).ravel() # Update the state vector
        P = (np.eye(p) - K @ H) @ P_pred # Update the covariance matrix

        x_hat[n] = s[0] # The estimated clean signal is the first component of the state vector

    return x_hat.astype(np.float32) # Return the estimated clean signal as a float32 array


def estimate_R(clean: np.ndarray, noisy: np.ndarray) -> float:
    return float(np.var(noisy - clean))


def compute_snr(clean: np.ndarray, estimated: np.ndarray) -> float:
    noise_est = estimated - clean
    return float(10 * np.log10(np.mean(clean ** 2) / (np.mean(noise_est ** 2) + 1e-30)))

