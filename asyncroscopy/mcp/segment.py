"""Tango device using SAM2 model to segment images into regions of interest."""

import json
import numpy as np
import cv2
import torch
from scipy import ndimage

import tango
from tango.server import Device, attribute, command, device_property

try:
    from sam2.build_sam import build_sam2_hf
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    print("SAM 2 not installed. Run: pip install git+https://github.com/facebookresearch/sam2.git")
    
    class SEGMENTATION(Device): 
        model_size = device_property(dtype=str, default_value="facebook/sam2-hiera-large", doc="HuggingFace model ID for SAM 2")
        points_per_side = attribute(dtype=int, access= AttrWriteType.READ_WRITE, doc="Grid density for automatic mask generation",)

        def init_device(self) -> None:
        """Initialize the segmentation device."""

        Device.init_device(self)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.set_state(tango.DevState.INIT)
        self._points_per_side = 48
        
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

            def read_points_per_side(self) -> int:
                return self._points_per_side
            
            def write_points_per_side(self, value: int) -> None:
                return self._points_per_side = value

    @attribute(dtype=str, doc ="Statistics of segmented areas")
    def area_stats(self) -> str:
        """Return the statistics of segmented areas as a JSON string."""
        return json.dumps(self._area_stats)

