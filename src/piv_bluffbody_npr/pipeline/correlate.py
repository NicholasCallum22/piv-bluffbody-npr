"""Correlation routines using FFT with normalization and window tapering."""
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from typing import Tuple

def _tukey_window(shape, alpha=0.5):
    """
    Return a 2D tapering window. Prefer scipy.signal.windows.tukey when available,
    otherwise fall back to a separable Hann window to avoid hard dependency issues.
    """
    try:
        # preferred location for tukey in modern SciPy
        from scipy.signal.windows import tukey
        wy = tukey(shape[0], alpha)
        wx = tukey(shape[1], alpha)
        return np.outer(wy, wx)
    except Exception:
        # fallback: separable Hann windows (smooth taper)
        wy = 0.5 * (1 - np.cos(2 * np.pi * np.arange(shape[0]) / max(1, shape[0]-1)))
        wx = 0.5 * (1 - np.cos(2 * np.pi * np.arange(shape[1]) / max(1, shape[1]-1)))
        return np.outer(wy, wx)

def cross_correlation_fft(patch_a: np.ndarray, patch_b: np.ndarray) -> np.ndarray:
    """Compute normalized cross correlation via FFT and return correlation map centered."""
    a = patch_a.astype(np.float32)
    b = patch_b.astype(np.float32)
    # subtract mean to reduce DC
    a = a - a.mean()
    b = b - b.mean()
    # apply window to reduce edge effects
    w = _tukey_window(a.shape, alpha=0.25)
    fa = fft2(a * w)
    fb = fft2(b * w)
    corr = fftshift(ifft2(fa * np.conj(fb)).real)
    # normalize by energy
    denom = np.sqrt(np.sum((a*w)**2) * np.sum((b*w)**2))
    if denom > 0:
        corr = corr / denom
    # shift to positive range for SNR computations
    corr -= corr.min()
    return corr

def find_peak(corr: np.ndarray) -> Tuple[int, int, float]:
    idx = np.unravel_index(np.argmax(corr), corr.shape)
    return int(idx[0]), int(idx[1]), float(corr[idx])
