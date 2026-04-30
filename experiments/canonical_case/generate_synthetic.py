"""Deterministic synthetic PIV pair generator for tests and smoke runs."""
from pathlib import Path
import numpy as np
from PIL import Image
import yaml

def generate_particles(shape, n_particles, seed=0, psf_sigma=1.0):
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, dtype=np.float32)
    ys = rng.integers(0, shape[0], size=n_particles)
    xs = rng.integers(0, shape[1], size=n_particles)
    for y, x in zip(ys, xs):
        img[y, x] += 1.0
    # simple gaussian PSF via convolution
    try:
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=psf_sigma)
    except Exception:
        # if scipy not available, apply a tiny local blur via simple kernel
        kernel = np.array([[0.05,0.1,0.05],[0.1,0.4,0.1],[0.05,0.1,0.05]])
        from scipy.signal import convolve2d as _conv  # will raise if scipy missing
        img = _conv(img, kernel, mode='same', boundary='wrap')
    # normalize to 0-255
    img -= img.min()
    if img.max() > 0:
        img = img / img.max() * 255.0
    return img.astype(np.uint8), np.column_stack((ys, xs))

def shift_particles(img, coords, dy, dx):
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    for (y, x) in coords:
        ny = int(round(y + dy))
        nx = int(round(x + dx))
        if 0 <= ny < h and 0 <= nx < w:
            out[ny, nx] += img[y, x]
    try:
        from scipy.ndimage import gaussian_filter
        out = gaussian_filter(out, sigma=1.0)
    except Exception:
        pass
    out -= out.min()
    if out.max() > 0:
        out = out / out.max() * 255.0
    return out.astype(np.uint8)

def save_pair(out_dir, name, img1, img2):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img1).save(p / f"{name}_a.png")
    Image.fromarray(img2).save(p / f"{name}_b.png")

def main():
    cfg_path = Path("experiments/canonical_case/metadata.yaml")
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(open(cfg_path)) or {}
        except Exception:
            cfg = {}
    else:
        cfg = {}
    gen = cfg.get("generator", {})
    # defaults
    shape = tuple(gen.get("shape", [256, 256]))
    n = int(gen.get("n_particles", 200))
    seed = int(gen.get("seed", 42))
    dy = float(gen.get("dy", 0.5))
    dx = float(gen.get("dx", 1.2))
    img1, coords = generate_particles(shape, n, seed=seed, psf_sigma=1.0)
    img2 = shift_particles(img1, coords, dy, dx)
    save_pair("data/canonical", "frame_0001", img1, img2)
    print("Wrote data/canonical/frame_0001_a.png and _b.png")

if __name__ == "__main__":
    main()
