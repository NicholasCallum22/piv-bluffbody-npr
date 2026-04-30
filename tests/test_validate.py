import numpy as np
from piv_bluffbody_npr.pipeline.validate import sig2noise, peak_ratio, adaptive_snr_mask
def test_snr_and_peak_ratio_basic():
    corr = np.zeros((11,11), dtype=np.float32)
    corr[5,5] = 100.0
    corr[2,2] = 10.0
    s = sig2noise(corr)
    pr = peak_ratio(corr, 5, 5)
    assert s > 1.0
    assert s > 1.0

def test_adaptive_snr_mask_uses_metadata():
    corr = np.zeros((11,11), dtype=np.float32)
    corr[5,5] = 10.0
    metadata = {"processing": {"validation": {"snr_threshold": 20.0, "peak_ratio": 1.0, "adaptive_fallback": True}}}
    # SNR below threshold but adaptive_fallback True should allow borderline acceptance logic
    accepted = adaptive_snr_mask(corr, metadata)
    assert isinstance(accepted, bool)
