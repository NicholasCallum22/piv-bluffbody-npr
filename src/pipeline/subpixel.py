"""
Subpixel peak estimation
"""
import numpy as np
from typing import Tuple

def gaussian_3pt_subpixel(corr: np.ndarray, peak_y: int, peak_x: int) -> Tuple[float, float]:
    """
    1D three-point Gaussian fit in y and x around integer peak.
    Returns subpixel offsets (dy, dx) relative to integer peak.
    """
    def fit_1d(arr, i):
        if i <= 0 or i >= len(arr) - 1:
            return 0.0
        a = arr[i-1]
        b = arr[i]
        c = arr[i+1]
        denom = (a - 2*b + c)
        if denom == 0:
            return 0.0
        return 0.5 * (a - c) / denom

    # extract 1D slices
    row = corr[peak_y, :]
    col = corr[:, peak_x]
    dx = fit_1d(row, peak_x)
    dy = fit_1d(col, peak_y)
    return float(dy), float(dx)

def estimate_displacement(peak_pos: Tuple[float, float], window_center: Tuple[int, int], delta_t: float, scale: float) -> Tuple[float, float]:
    """
    Convert pixel displacement to velocity.
    peak_pos is (y, x) in pixel coordinates (can be fractional).
    window_center is (y0, x0) integer center of the interrogation window.
    delta_t is time between frames in seconds.
    scale is meters per pixel.
    Returns (u, v) in m/s where u is x-direction.
    """
    dy = peak_pos[0] - window_center[0]
    dx = peak_pos[1] - window_center[1]
    u = dx * scale / delta_t
    v = dy * scale / delta_t
    return u, v

