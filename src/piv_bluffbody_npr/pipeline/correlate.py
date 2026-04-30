"""
Correlation routines using FFT
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from typing import Tuple

def cross_correlation_fft(patch_a: np.ndarray, patch_b: np.ndarray) -> np.ndarray:
    """Compute normalized cross correlation via FFT and return correlation map."""
    fa = fft2(patch_a.astype(np.float32))
    fb = fft2(patch_b.astype(np.float32))
    corr = fftshift(ifft2(fa * np.conj(fb)).real)
    # normalize
    corr -= corr.min()
    if corr.max() > 0:
        corr = corr / corr.max()
    return corr

def find_peak(corr: np.ndarray) -> Tuple[int, int, float]:
    """Return integer peak coordinates and peak value."""
    idx = np.unravel_index(np.argmax(corr), corr.shape)
    return int(idx[0]), int(idx[1]), float(corr[idx])

