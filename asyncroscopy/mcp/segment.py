"""Tango device using SAM2 model to segment images into regions of interest."""

import json
import numpy as np
import cv2
import torch
from scipy import ndimage
from PIL import Image

import tango
from tango import AttrWriteType, DevState
from tango.server import Device, attribute, command, device_property

try:
    from sam2.build_sam import build_sam2_hf
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    print("SAM 2 not installed. Run: pip install git+https://github.com/facebookresearch/sam2.git")


class SEGMENTATION(Device):
    model_size = device_property(dtype=str, default_value="facebook/sam2-hiera-large", doc="HuggingFace model ID for SAM 2")
    points_per_side = attribute(dtype=int, access=AttrWriteType.READ_WRITE, doc="Number of points SAM samples along each edge of the image.")
    iou_threshold = attribute(dtype=float, access=AttrWriteType.READ_WRITE, doc="IoU threshold for mask merging")
    stability_thresh = attribute(dtype=float, access=AttrWriteType.READ_WRITE, doc="Stability threshold for mask filtering")
    min_area_px = attribute(dtype=int, access=AttrWriteType.READ_WRITE, doc="Minimum area in pixels for valid masks")
    n_areas = attribute(dtype=int, access=AttrWriteType.READ, doc="Number of segmented areas")
    centroids = attribute(dtype=str, access=AttrWriteType.READ, doc="Centroids of segmented areas")

    def init_device(self) -> None:
        """Initialize the SAM 2 segmentation device, set SAM 2 parameters and segmentation statistics."""
        Device.init_device(self)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.set_state(tango.DevState.INIT)
        self._points_per_side = 48
        self._iou_threshold = 0.57
        self._stability_thresh = 0.75
        self._min_area_px = 200
        self._crop_n_layers = 1

        self._n_areas = 0
        self._centroids = []
        self._area_stats = []

        try:
            self._sam2 = build_sam2_hf(self.model_size, device=self._device)
            self.set_state(tango.DevState.ON)
            self.info_stream("Segmentation device initialized")
        except Exception as e:
            self.set_state(tango.DevState.FAULT)
            self.set_status(f"Initialization failed: {e}")

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare the image for segmentation by converting to RGB and enhancing."""
        gray = image.astype(np.float32)
        if gray.ndim == 3:
            gray = np.mean(gray, axis=-1)
        lo, hi = np.percentile(gray, [0.5, 99.5])
        gray = np.clip((gray - lo) / (hi - lo + 1e-8), 0, 1)

        gray_8bit = (gray * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_8bit)

        image = np.stack([enhanced] * 3, axis=-1)
        return image

    def segment_image(self, image: np.ndarray) -> list:
        """Segment the image with SAM 2 and return a list of masks."""
        mask_generator = SAM2AutomaticMaskGenerator(
            model=self._sam2,
            points_per_side=self._points_per_side,
            pred_iou_thresh=self._iou_threshold,
            stability_score_thresh=self._stability_thresh,
            min_mask_region_area=self._min_area_px,
            crop_n_layers=self._crop_n_layers,
        )
        masks = mask_generator.generate(image)
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        total_pixels = image.shape[0] * image.shape[1]

        areas = [m for m in masks if m["area"] < total_pixels * 0.5]
        areas = [m for m in areas if m["area"] > self._min_area_px]

        return areas

    def area_statistics(self, areas: list) -> list:
        """Compute statistics of segmented areas."""
        area_stats = []
        centroids = []

        for i, area in enumerate(areas):
            m = area["segmentation"]
            area_px = area["area"]
            cy, cx = ndimage.center_of_mass(m)
            equiv_d = 2 * np.sqrt(area_px / np.pi)
            eroded = ndimage.binary_erosion(m)
            perimeter = np.sum(m & ~eroded)
            circularity = min((4 * np.pi * area_px) / (perimeter**2), 1.0) if perimeter > 0 else 0

            area_stats.append({
                "id": i,
                "area_px": area_px,
                "equiv_diameter_px": equiv_d,
                "circularity": circularity,
                "confidence": area["predicted_iou"],
            })
            centroids.append([float(cx), float(cy)])

        self._n_areas = len(areas)
        self._centroids = centroids
        self._area_stats = area_stats
        return area_stats, centroids

    # ------------------------------------------------------------------
    # Attribute read / write
    # ------------------------------------------------------------------

    def read_iou_threshold(self) -> float:
        return self._iou_threshold

    def write_iou_threshold(self, value: float) -> None:
        self._iou_threshold = value

    def read_points_per_side(self) -> int:
        return self._points_per_side

    def write_points_per_side(self, value: int) -> None:
        self._points_per_side = value

    def read_stability_thresh(self) -> float:
        return self._stability_thresh

    def write_stability_thresh(self, value: float) -> None:
        self._stability_thresh = value

    def read_min_area_px(self) -> int:
        return self._min_area_px

    def write_min_area_px(self, value: int) -> None:
        self._min_area_px = value

    def read_n_areas(self) -> int:
        return self._n_areas

    def read_centroids(self) -> str:
        return json.dumps(self._centroids)

    @attribute(dtype=str, doc="Statistics of segmented areas")
    def area_stats(self) -> str:
        """Return the statistics of segmented areas as a JSON string."""
        return json.dumps(self._area_stats)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @command(dtype_in=str, dtype_out=str)
    def segment(self, image_path: str) -> str:
        """Segment areas in microscope images using SAM 2.
        
        Input: file path to user image (png, tif, emd, etc.
        Output: JSON list of area statistics including id, area in pixels,
        equivalent diameter, circularity, and confidence score.
        This also updates n_areas, centroids, and area_stats attributes."""
        
        try:
            image_data = np.array(Image.open(image_path))
            prepared = self._prepare_image(image_data)
            areas = self.segment_image(prepared)
            area_stats, centroids = self.area_statistics(areas)
            return json.dumps(area_stats)
        except Exception as e:
            self.error_stream(f"Failed to segment: {e}")
            return json.dumps({"error": str(e)})

    @command(dtype_in=int, dtype_out=str)
    def get_centroid(self, area_id: int) -> str:
        """Return the centroid of a specific area as a JSON string."""
        try:
            return json.dumps(self._centroids[area_id])
        except IndexError:
            self.error_stream(f"Area ID {area_id} does not exist.")
            return json.dumps({"error": "Area ID does not exist."})


# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    SEGMENTATION.run_server()
