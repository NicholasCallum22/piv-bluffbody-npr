import numpy as np
from piv_bluffbody_npr.pipeline.preprocess import normalize_image, window_image
def test_normalize_image_range():
    a = np.array([[0.0, 50.0],[100.0,200.0]], dtype=np.float32)
    out = normalize_image(a)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
def test_window_image_counts():
    img = np.zeros((64,64), dtype=np.uint8)
    patches, coords = window_image(img, window_size=32, overlap=50)
    # step = 16 -> (0,16) positions -> 3x3 = 9 patches
    assert patches.shape[0] == coords.shape[0]
    assert patches.shape[1:] == (32,32)
