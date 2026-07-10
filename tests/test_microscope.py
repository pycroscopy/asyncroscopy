# """
# Tests for the STEMMicroscope Tango device.

# AutoScript is not available in CI — the STEMMicroscope device falls back
# to simulation mode, so image shape/dtype assertions are against the
# simulated output.
# """

# import json

# import numpy as np
# import pytest
# import tango


# class TestMicroscopeState:

#     def test_initial_state_is_on(self, microscope_proxy):
#         assert microscope_proxy.state() == tango.DevState.ON


# class TestGetImage:

#     def test_get_haadf_image_returns_encoded(self, microscope_proxy):
#         result = microscope_proxy.acquire_scanned_image()
#         # DevEncoded comes back as a two-element sequence
#         assert len(result) == 2

#     def test_metadata_is_valid_json(self, microscope_proxy):
#         json_meta, _ = microscope_proxy.acquire_scanned_image()
#         meta = json.loads(json_meta)
#         assert "shape" in meta
#         assert "dtype" in meta
#         assert "detector" in meta
#         assert "timestamp" in meta

#     def test_image_shape_matches_metadata(self, microscope_proxy):
#         json_meta, raw_bytes = microscope_proxy.acquire_scanned_image()
#         meta = json.loads(json_meta)
#         image = np.frombuffer(raw_bytes, dtype=meta["dtype"]).reshape(meta["shape"])
#         assert image.shape == tuple(meta["shape"])

#     def test_unknown_detector_raises(self, microscope_proxy):
#         with pytest.raises(tango.DevFailed):
#             microscope_proxy.acquire_scanned_image("nonexistent_detector")