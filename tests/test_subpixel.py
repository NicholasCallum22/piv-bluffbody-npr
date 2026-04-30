import numpy as np
from piv_bluffbody_npr.pipeline.subpixel import gaussian_3pt_subpixel, estimate_displacement
def test_gaussian_subpixel_center():
    corr = np.zeros((5,5), dtype=np.float32)
    corr[2,2] = 10.0
    corr[2,1] = 6.0
    corr[2,3] = 6.0
    dy, dx = gaussian_3pt_subpixel(corr, 2, 2)
    assert abs(dy) < 0.5 and abs(dx) < 0.5
def test_estimate_displacement_units():
    peak_pos = (16.2, 16.7)
    center = (16, 16)
    delta_t = 0.002
    scale = 1e-4
    u, v = estimate_displacement(peak_pos, center, delta_t, scale)
    # small pixel offsets map to small velocities
    assert abs(u) < 1.0
    assert abs(v) < 1.0
