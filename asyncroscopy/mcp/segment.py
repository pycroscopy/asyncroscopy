"""Tango device using SAM2 to segment DATA/Tiled image acquisitions."""

from __future__ import annotations

import json
import os
from typing import Any, TypedDict, cast
from uuid import uuid4

import numpy as np
import tango
from scipy import ndimage
from tango import AttrWriteType, DevState
from tango.server import Device, attribute, command, device_property
from tiled.client import from_uri

try:
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf
except ImportError as exc:
    torch = None
    SAM2AutomaticMaskGenerator = None
    build_sam2_hf = None
    _SAM2_IMPORT_ERROR: ImportError | None = exc
else:
    _SAM2_IMPORT_ERROR = None


class SAM2Mask(TypedDict):
    """SAM2 fields used by the segmentation device."""

    segmentation: np.ndarray
    area: int
    predicted_iou: float


class AreaStatistic(TypedDict):
    """JSON-serializable statistics for one segmented region."""

    id: int
    area_px: int
    equiv_diameter_px: float
    circularity: float
    confidence: float


def _read_first_array(node: Any) -> np.ndarray:
    """Read the first array below a Tiled node in deterministic key order."""
    read = getattr(node, "read", None)
    if callable(read):
        return np.asarray(read())

    keys = getattr(node, "keys", None)
    if callable(keys):
        child_names = sorted(str(name) for name in keys())
        for child_name in child_names:
            try:
                return _read_first_array(node[child_name])
            except ValueError:
                continue

    raise ValueError("The DATA/Tiled key does not contain a readable array")


def _prepare_image(image: np.ndarray) -> np.ndarray:
    """Normalize a two-dimensional image and expand it to RGB for SAM2."""
    gray = np.squeeze(np.asarray(image)).astype(np.float32)
    if gray.ndim == 3 and gray.shape[-1] in {3, 4}:
        gray = np.mean(gray[..., :3], axis=-1)
    if gray.ndim != 2:
        raise ValueError(
            f"Segmentation requires a 2-D grayscale or RGB image; received shape {gray.shape}"
        )

    finite = np.isfinite(gray)
    if not np.any(finite):
        raise ValueError("Segmentation input contains no finite values")

    lo, hi = np.percentile(gray[finite], [0.5, 99.5])
    normalized = np.clip((gray - lo) / (hi - lo + 1e-8), 0, 1)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    gray_8bit = (normalized * 255).astype(np.uint8)
    return np.repeat(gray_8bit[..., np.newaxis], 3, axis=-1)


def _label_areas(areas: list[SAM2Mask], shape: tuple[int, int]) -> np.ndarray:
    """Convert overlapping SAM2 masks into one integer-labeled image."""
    labels = np.zeros(shape, dtype=np.uint32)
    for label, area in enumerate(areas, start=1):
        mask = np.asarray(area["segmentation"], dtype=bool)
        if mask.shape != shape:
            raise ValueError(
                f"SAM2 returned mask shape {mask.shape}; expected image shape {shape}"
            )
        # Areas are ordered largest to smallest, so more-specific small masks win
        # when SAM2 returns overlapping regions.
        labels[mask] = label
    return labels


class SEGMENTATION(Device):
    """Segment DATA/Tiled image keys with SAM2 and save integer label images."""

    model_size = device_property(
        dtype=str,
        default_value="facebook/sam2-hiera-large",
        doc="HuggingFace model ID for SAM2",
    )
    data_device_address = device_property(
        dtype=str,
        default_value="asyncroscopy/data/default",
        doc="Tango DATA device used to read input keys and save segmentation labels",
    )
    compute_device = device_property(
        dtype=str,
        default_value="auto",
        doc="PyTorch compute device: 'auto', 'cpu', 'cuda', or a CUDA index such as 'cuda:1'",
    )

    points_per_side = attribute(
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        doc="Number of points SAM samples along each image edge",
    )
    iou_threshold = attribute(
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        doc="IoU threshold for mask merging",
    )
    stability_thresh = attribute(
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        doc="Stability threshold for mask filtering",
    )
    min_area_px = attribute(
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        doc="Minimum area in pixels for valid masks",
    )
    n_areas = attribute(
        dtype=int,
        access=AttrWriteType.READ,
        doc="Number of segmented areas",
    )
    centroids = attribute(
        dtype=str,
        access=AttrWriteType.READ,
        doc="JSON-encoded centroids of segmented areas",
    )

    def init_device(self) -> None:
        """Initialize SAM2 parameters, model, and segmentation statistics."""
        Device.init_device(self)
        self.set_state(DevState.INIT)
        self._points_per_side = 48
        self._iou_threshold = 0.57
        self._stability_thresh = 0.75
        self._min_area_px = 200
        self._crop_n_layers = 1
        self._n_areas = 0
        self._centroids: list[list[float]] = []
        self._area_stats: list[AreaStatistic] = []
        self._data_proxy = None
        self._sam2 = None
        self._active_compute_device = "unavailable"

        if _SAM2_IMPORT_ERROR is not None:
            message = (
                "SAM2 dependencies are not installed. "
                "Run `uv sync --extra segment` before starting this server. "
                f"Import error: {_SAM2_IMPORT_ERROR}"
            )
            self.set_state(DevState.FAULT)
            self.set_status(message)
            self.error_stream(message)
            return

        try:
            assert torch is not None
            requested_device = self.compute_device.strip().lower()
            if requested_device == "auto":
                requested_device = "cuda" if torch.cuda.is_available() else "cpu"

            device = torch.device(requested_device)
            if device.type not in {"cpu", "cuda"}:
                raise ValueError(
                    f"Unsupported compute_device {self.compute_device!r}; "
                    "expected 'auto', 'cpu', 'cuda', or 'cuda:<index>'"
                )
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    f"compute_device is {self.compute_device!r}, but CUDA is unavailable "
                    f"in PyTorch {torch.__version__} (torch.version.cuda={torch.version.cuda!r}). "
                    "Install a CUDA-enabled PyTorch build on this machine."
                )

            self._active_compute_device = str(device)
            assert build_sam2_hf is not None
            self._sam2 = build_sam2_hf(self.model_size, device=self._active_compute_device)
            self.set_state(DevState.ON)
            accelerator = (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            )
            message = (
                f"Segmentation device initialized on {self._active_compute_device} "
                f"({accelerator})"
            )
            self.set_status(message)
            self.info_stream(message)
        except Exception as exc:
            self.set_state(DevState.FAULT)
            self.set_status(f"Initialization failed: {exc}")

    def _get_data_proxy(self):
        """Return the configured DATA proxy, retrying startup-time failures."""
        if self._data_proxy is None:
            if not self.data_device_address:
                raise RuntimeError("No DATA device is configured for segmentation")
            self._data_proxy = tango.DeviceProxy(self.data_device_address)
            self._data_proxy.set_timeout_millis(120_000)
        return self._data_proxy

    def _load_image_from_key(self, key: str, data_proxy: Any) -> np.ndarray:
        """Resolve a DATA/Tiled key and read its first array dataset."""
        config = json.loads(data_proxy.get_config())
        uri = config.get("uri")
        if not uri:
            raise RuntimeError(
                f"DATA device {self.data_device_address!r} did not provide a Tiled URI"
            )

        client = from_uri(uri)
        try:
            node = client[key]
        except KeyError as exc:
            raise FileNotFoundError(
                f"Could not resolve data key {key!r} from Tiled server {uri!r}"
            ) from exc
        return _read_first_array(node)

    def segment_image(self, image: np.ndarray) -> list[SAM2Mask]:
        """Segment an RGB image with SAM2 and return filtered masks."""
        if self._sam2 is None or SAM2AutomaticMaskGenerator is None:
            raise RuntimeError("SAM2 model is not initialized")

        mask_generator = SAM2AutomaticMaskGenerator(
            model=self._sam2,
            points_per_side=self._points_per_side,
            pred_iou_thresh=self._iou_threshold,
            stability_score_thresh=self._stability_thresh,
            min_mask_region_area=self._min_area_px,
            crop_n_layers=self._crop_n_layers,
        )
        assert torch is not None
        with torch.inference_mode():
            if self._active_compute_device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    masks = cast(list[SAM2Mask], mask_generator.generate(image))
            else:
                masks = cast(list[SAM2Mask], mask_generator.generate(image))
        masks.sort(key=lambda mask: mask["area"], reverse=True)
        total_pixels = image.shape[0] * image.shape[1]
        return [
            mask
            for mask in masks
            if self._min_area_px < mask["area"] < total_pixels * 0.5
        ]

    def area_statistics(
        self, areas: list[SAM2Mask]
    ) -> tuple[list[AreaStatistic], list[list[float]]]:
        """Compute JSON-serializable statistics for segmented areas."""
        area_stats: list[AreaStatistic] = []
        centroids: list[list[float]] = []

        for area_id, area in enumerate(areas, start=1):
            mask = np.asarray(area["segmentation"], dtype=bool)
            area_px = int(area["area"])
            cy, cx = ndimage.center_of_mass(mask)
            equiv_diameter = float(2 * np.sqrt(area_px / np.pi))
            eroded = ndimage.binary_erosion(mask)
            perimeter = int(np.sum(mask & ~eroded))
            circularity = (
                min(float((4 * np.pi * area_px) / (perimeter**2)), 1.0)
                if perimeter > 0
                else 0.0
            )

            area_stats.append(
                {
                    "id": area_id,
                    "area_px": area_px,
                    "equiv_diameter_px": equiv_diameter,
                    "circularity": circularity,
                    "confidence": float(area["predicted_iou"]),
                }
            )
            centroids.append([float(cx), float(cy)])

        self._n_areas = len(areas)
        self._centroids = centroids
        self._area_stats = area_stats
        return area_stats, centroids

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
        """Return the statistics of segmented areas as JSON."""
        return json.dumps(self._area_stats)

    @command(dtype_in=str, dtype_out=str)
    def segment(self, data_key: str) -> str:
        """Segment a DATA/Tiled image key and return a label-image DATA key."""
        try:
            data_proxy = self._get_data_proxy()
            image_data = self._load_image_from_key(data_key, data_proxy)
            prepared = _prepare_image(image_data)
            areas = self.segment_image(prepared)
            area_stats, _ = self.area_statistics(areas)
            labels = _label_areas(areas, prepared.shape[:2])
            key = f"segmentation_{uuid4().hex}"
            tiled = from_uri(
                json.loads(data_proxy.get_config())["uri"],
                api_key=os.environ.get("ASYNCROSCOPY_TILED_API_KEY", "secret"),
            )
            tiled.write_array(
                labels,
                key=key,
                metadata={
                    "acquisition_type": "segmentation",
                    "detector": "sam2",
                    "source_data_key": data_key,
                    "model": self.model_size,
                    "area_statistics": area_stats,
                },
            )
            return key
        except Exception as exc:
            message = f"Failed to segment DATA key {data_key!r}: {exc}"
            self.error_stream(message)
            raise RuntimeError(message) from exc

    @command(dtype_in=int, dtype_out=str)
    def get_centroid(self, area_id: int) -> str:
        """Return the centroid of a one-based segmented area ID as JSON."""
        if area_id < 1 or area_id > len(self._centroids):
            raise ValueError(f"Area ID {area_id} does not exist")
        return json.dumps(self._centroids[area_id - 1])


if __name__ == "__main__":
    SEGMENTATION.run_server()
