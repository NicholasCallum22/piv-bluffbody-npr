import numpy as np
from piv_bluffbody_npr.pipeline.correlate import cross_correlation_fft, find_peak
def test_cross_correlation_peak_shift():
    a = np.zeros((32,32), dtype=np.float32)
    b = np.zeros_like(a)
    a[16,16] = 1.0
    b[16,18] = 1.0
    corr = cross_correlation_fft(a,b)
    py, px, val = find_peak(corr)
    assert val > 0.1
    # peak should be near center offset corresponding to shift
    assert abs(py - corr.shape[0]//2) <= 2
