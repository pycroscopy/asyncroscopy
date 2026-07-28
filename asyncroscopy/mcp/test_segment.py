import numpy as np
from PIL import Image
from segment import SEGMENTATION

# Test the methods directly (bypass Tango)
seg = SEGMENTATION.__new__(SEGMENTATION)

# Manual init (skip Tango boilerplate)
import torch
seg._device = "cuda" if torch.cuda.is_available() else "cpu"
seg._points_per_side = 48
seg._iou_threshold = 0.57
seg._stability_thresh = 0.75
seg._min_area_px = 200
seg._crop_n_layers = 1
seg._n_areas = 0
seg._centroids = []
seg._area_stats = []

from sam2.build_sam import build_sam2_hf
seg._sam2 = build_sam2_hf("facebook/sam2-hiera-large", device=seg._device)
print("SAM loaded!")

# Test on an image
image = np.array(Image.open('/Users/ikshvaku/Desktop/Screenshot 2026-07-06 at 9.42.25 AM.png'))
prepared = seg._prepare_image(image)
print(f"Prepared: {prepared.shape}, dtype={prepared.dtype}")

areas = seg.segment_image(prepared)
print(f"Found {len(areas)} areas")

stats, centroids = seg.area_statistics(areas)
print(f"Stats: {len(stats)} areas")
for s in stats[:5]:
    print(f"  Area {s['id']}: {s['area_px']}px, circularity={s['circularity']:.3f}")
print(f"Centroids: {centroids[:5]}")