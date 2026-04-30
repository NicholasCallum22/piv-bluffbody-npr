import os
import numpy as np
from experiments.canonical_case.generate_synthetic import main as gen_main
from src.piv_bluffbody_npr.pipeline.preprocess import normalize_image
from src.piv_bluffbody_npr.pipeline.correlate import cross_correlation_fft, find_peak
from src.piv_bluffbody_npr.pipeline.subpixel import gaussian_3pt_subpixel

def test_synthetic_pair_and_correlation(tmp_path):
    # generate synthetic pair into data/canonical
    gen_main()
    a = np.array(__import__('PIL').Image.open("data/canonical/frame_0001_a.png").convert("L"))
    b = np.array(__import__('PIL').Image.open("data/canonical/frame_0001_b.png").convert("L"))
    a_n = normalize_image(a)
    b_n = normalize_image(b)
    # small patch around center
    h, w = a_n.shape
    y0, x0 = h//2 - 16, w//2 - 16
    pa = a_n[y0:y0+32, x0:x0+32]
    pb = b_n[y0:y0+32, x0:x0+32]
    corr = cross_correlation_fft(pa, pb)
    py, px, val = find_peak(corr)
    dy, dx = gaussian_3pt_subpixel(corr, py, px)
    assert val > 0.0
    assert abs(dy) <= 1.0 and abs(dx) <= 1.0
