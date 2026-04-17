# piv_tutorial_min.py
import os
import numpy as np
import matplotlib.pyplot as plt
from openpiv import tools, pyprocess, validation, filters

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
IMG_A = os.path.join(DATA_DIR, 'img_001.tif')
IMG_B = os.path.join(DATA_DIR, 'img_002.tif')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Load images (replace with your sample images)
img_a = tools.imread(IMG_A)
img_b = tools.imread(IMG_B)

# PIV parameters
window_size = 64
overlap = 32
dt = 1.0
search_area_size = 64

u0, v0, sig2noise = pyprocess.extended_search_area_piv(
    img_a, img_b,
    window_size=window_size, overlap=overlap,
    dt=dt, search_area_size=search_area_size, sig2noise_method='peak2peak'
)

x, y = pyprocess.get_coordinates(image_size=img_a.shape, window_size=window_size, overlap=overlap)
u1, v1, mask = validation.sig2noise_val(u0, v0, sig2noise, threshold=1.2)
u2, v2 = filters.replace_outliers_by_median(u1, v1, 3)

plt.figure(figsize=(8,6))
plt.quiver(x, y, u2, v2, scale=50)
plt.gca().invert_yaxis()
plt.title('PIV vector field (tutorial)')
plt.savefig(os.path.join(OUT_DIR, 'vector_field.png'), dpi=300)
print("Saved vector field to figures/vector_field.png")
