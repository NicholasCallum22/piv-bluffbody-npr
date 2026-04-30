"""Preprocessing utilities for PIV images."""
from typing import Tuple
import numpy as np
from scipy import ndimage

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Scale image to uint8 range 0-255 preserving contrast and handling NaNs."""
    a = np.asarray(img, dtype=np.float64)
    # replace NaN with min
    if np.isnan(a).any():
        a = np.nan_to_num(a, nan=np.nanmin(a))
    a -= a.min()
    if a.max() > 0:
        a = a / a.max() * 255.0
    return a.astype(np.uint8)

def subtract_background(img: np.ndarray, background: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32) - background.astype(np.float32)
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)

def window_image(img: np.ndarray, window_size: int, overlap: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    step = max(1, int(window_size * (1 - overlap / 100.0)))
    coords = []
    patches = []
    for y in range(0, h - window_size + 1, step):
        for x in range(0, w - window_size + 1, step):
            coords.append((y, x))
            patches.append(img[y:y+window_size, x:x+window_size])
    if len(patches) == 0:
        return np.empty((0, window_size, window_size), dtype=img.dtype), np.empty((0,2), dtype=int)
    return np.stack(patches, axis=0), np.array(coords, dtype=int)
