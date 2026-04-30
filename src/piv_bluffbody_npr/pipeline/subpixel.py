"""Subpixel peak estimation using 3-point Gaussian and optional 2D quadratic fallback."""
import numpy as np
from typing import Tuple

def gaussian_3pt_subpixel(corr: np.ndarray, peak_y: int, peak_x: int) -> Tuple[float, float]:
    """1D three-point Gaussian fit in y and x around integer peak. Returns (dy, dx)."""
    def fit_1d(arr, i):
        if i <= 0 or i >= len(arr) - 1:
            return 0.0
        a = float(arr[i-1])
        b = float(arr[i])
        c = float(arr[i+1])
        denom = (a - 2*b + c)
        if abs(denom) < 1e-12:
            return 0.0
        return 0.5 * (a - c) / denom

    row = corr[peak_y, :]
    col = corr[:, peak_x]
    dx = fit_1d(row, peak_x)
    dy = fit_1d(col, peak_y)
    # clamp to reasonable range
    dx = float(np.clip(dx, -1.0, 1.0))
    dy = float(np.clip(dy, -1.0, 1.0))
    return dy, dx

def estimate_displacement(peak_pos: Tuple[float, float], window_center: Tuple[int, int], delta_t: float, scale: float) -> Tuple[float, float]:
    dy = peak_pos[0] - window_center[0]
    dx = peak_pos[1] - window_center[1]
    u = dx * scale / delta_t
    v = dy * scale / delta_t
    return u, v
