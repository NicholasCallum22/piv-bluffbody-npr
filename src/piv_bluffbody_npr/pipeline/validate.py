"""
Validation and uncertainty helpers
"""
import numpy as np
from typing import Dict, Tuple

def sig2noise(corr: np.ndarray) -> float:
    """Simple SNR: peak divided by mean of correlation excluding neighborhood."""
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    peak = corr[py, px]
    mask = np.ones_like(corr, dtype=bool)
    r = 3
    y0 = max(0, py - r); y1 = min(corr.shape[0], py + r + 1)
    x0 = max(0, px - r); x1 = min(corr.shape[1], px + r + 1)
    mask[y0:y1, x0:x1] = False
    noise = corr[mask].mean() if np.any(mask) else 1e-6
    return float(peak / (noise + 1e-12))

def peak_ratio(corr: np.ndarray, peak_y: int, peak_x: int) -> float:
    """Ratio of primary peak to second highest peak."""
    flat = corr.flatten()
    primary = corr[peak_y, peak_x]
    flat_idx = np.argmax(flat)
    flat[flat_idx] = -np.inf
    second = flat.max()
    if second <= 0:
        return float(primary)
    return float(primary / (second + 1e-12))

def adaptive_snr_mask(corr: np.ndarray, metadata: Dict) -> bool:
    """Return True if vector accepted based on metadata thresholds and fallback rules."""
    snr = sig2noise(corr)
    pr = peak_ratio(corr, *np.unravel_index(np.argmax(corr), corr.shape))
    snr_thresh = metadata.get("processing", {}).get("validation", {}).get("snr_threshold", 1.5)
    pr_thresh = metadata.get("processing", {}).get("validation", {}).get("peak_ratio", 1.2)
    if snr >= snr_thresh and pr >= pr_thresh:
        return True
    # adaptive fallback example
    if metadata.get("processing", {}).get("validation", {}).get("adaptive_fallback", False):
        return snr >= (snr_thresh * 0.8)
    return False

def remove_outliers(u: np.ndarray, v: np.ndarray, method: str='local_median', kernel: int=3) -> Tuple[np.ndarray, np.ndarray]:
    """Replace outliers with NaN using local median thresholding."""
    if method != 'local_median':
        return u, v
    from scipy.ndimage import median_filter
    um = median_filter(u, size=kernel, mode='nearest')
    vm = median_filter(v, size=kernel, mode='nearest')
    diff = np.sqrt((u - um)**2 + (v - vm)**2)
    thresh = np.nanmedian(diff) + 3 * np.nanstd(diff)
    mask = diff > thresh
    u_out = u.copy()
    v_out = v.copy()
    u_out[mask] = np.nan
    v_out[mask] = np.nan
    return u_out, v_out

