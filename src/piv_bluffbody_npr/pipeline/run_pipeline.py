from PIL import Image
"""
Orchestration script to run the pipeline for a pair of frames using metadata
"""
import yaml
from pathlib import Path
import numpy as np
from .preprocess import normalize_image, window_image
from .correlate import cross_correlation_fft, find_peak
from .subpixel import gaussian_3pt_subpixel, estimate_displacement
from .validate import adaptive_snr_mask, remove_outliers

def run_pipeline(config_path: str, input_dir: str, out_dir: str) -> dict:
    cfg = yaml.safe_load(open(config_path))
    res = cfg["acquisition"]["camera"]["resolution"]
    # simple example: load two frames
    p = Path(input_dir)
    f1 = np.array(Image.open(p / "frame_0001.png").convert("L"))
    f2 = np.array(Image.open(p / "frame_0002.png").convert("L"))
    f1n = normalize_image(f1)
    f2n = normalize_image(f2)
    window_sizes = cfg["processing"]["window_sizes_px"]
    overlap = cfg["processing"]["overlap_percent"]
    # single pass using first window size
    patches1, coords = window_image(f1n, window_sizes[0], overlap)
    patches2, _ = window_image(f2n, window_sizes[0], overlap)
    u_list = []
    v_list = []
    for i, (pa, pb) in enumerate(zip(patches1, patches2)):
        corr = cross_correlation_fft(pa, pb)
        py, px, _ = find_peak(corr)
        dy, dx = gaussian_3pt_subpixel(corr, py, px)
        peak_pos = (py + dy, px + dx)
        center = (window_sizes[0]//2, window_sizes[0]//2)
        delta_t = cfg["acquisition"]["timing"]["delta_t_s"]
        scale = cfg.get("pixel_scale_m_per_px", 1e-4)  # default scale
        u, v = estimate_displacement(peak_pos, center, delta_t, scale)
        if adaptive_snr_mask(corr, cfg):
            u_list.append(u)
            v_list.append(v)
        else:
            u_list.append(np.nan)
            v_list.append(np.nan)
    u_arr = np.array(u_list)
    v_arr = np.array(v_list)
    u_f, v_f = remove_outliers(u_arr.reshape(-1), v_arr.reshape(-1))
    # minimal summary
    summary = {"n_vectors": int(np.sum(~np.isnan(u_f)))}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(out_dir) / "u.npy", u_f)
    np.save(Path(out_dir) / "v.npy", v_f)
    return summary

if __name__ == "__main__":
    import argparse
    from PIL import Image
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    print(run_pipeline(args.config, args.input, args.out))
