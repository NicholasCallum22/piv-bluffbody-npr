#!/usr/bin/env python3
"""
Generate synthetic PIV image pairs and analytic velocity field
based on metadata YAML. Deterministic when random_seed is set.
This is a minimal, well-documented stub to be expanded in Week 1.
"""
import argparse
import json
import os
from pathlib import Path
import yaml
import numpy as np
from PIL import Image

def analytic_velocity_field(metadata):
    # Example: simple linear shear in x-direction
    res = metadata["acquisition"]["camera"]["resolution"]
    max_u = metadata["ground_truth"]["parameters"]["max_velocity_m_s"]
    ny, nx = res[1], res[0]
    y = np.linspace(0, 1, ny)
    u = np.tile(max_u * y[:, None], (1, nx))
    v = np.zeros_like(u)
    return {"u": u, "v": v}

def synthesize_images(metadata, out_dir):
    res = metadata["acquisition"]["camera"]["resolution"]
    seed = int(metadata.get("random_seed", 0))
    rng = np.random.default_rng(seed)
    # simple particle field: random dots blurred to approximate particles
    nx, ny = res
    img1 = np.zeros((ny, nx), dtype=np.uint16)
    img2 = np.zeros_like(img1)
    # particle count proportional to concentration
    concentration = metadata["seeding"]["concentration_ppp"]
    n_particles = int(concentration * nx * ny)
    xs = rng.integers(0, nx, size=n_particles)
    ys = rng.integers(0, ny, size=n_particles)
    for x, y in zip(xs, ys):
        img1[y, x] = min(255, img1[y, x] + 200)
    # simple advection: shift by a small integer amount derived from analytic field
    u_field = analytic_velocity_field(metadata)["u"]
    shift_x = int(np.round(np.mean(u_field) * 10))  # toy mapping to pixels
    img2 = np.roll(img1, shift=shift_x, axis=1)
    # add gaussian noise
    noise_level = 5
    img1 = np.clip(img1, 0, 255).astype(np.uint8)
    img2 = np.clip(img2 + rng.normal(0, noise_level, img2.shape), 0, 255).astype(np.uint8)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Image.fromarray(img1).save(Path(out_dir) / "frame_0001.png")
    Image.fromarray(img2).save(Path(out_dir) / "frame_0002.png")
    # save analytic field as JSON (downsampled)
    gt = analytic_velocity_field(metadata)
    np.save(Path(out_dir) / "gt_u.npy", gt["u"])
    np.save(Path(out_dir) / "gt_v.npy", gt["v"])
    return {"images": ["frame_0001.png", "frame_0002.png"], "gt": ["gt_u.npy", "gt_v.npy"]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    m = yaml.safe_load(open(args.metadata))
    result = synthesize_images(m, args.out)
    print(json.dumps(result))

if __name__ == "__main__":
    main()

