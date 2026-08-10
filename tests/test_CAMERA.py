"""Tests for the CAMERA settings Tango device."""

import pytest
import tango


class TestCAMERAAttributes:
    """Attribute read/write round-trips and validation."""

    def test_defaults(self, camera_proxy):
        camera_proxy.exposure_time = 1e-3
        camera_proxy.imsize = 1024
        camera_proxy.readout_area = "Full"
        camera_proxy.camera_detector = "BM-Ceta"
        camera_proxy.frame_combining = 1
        camera_proxy.electron_counting = True
        camera_proxy.output_format = ".h5"

        assert camera_proxy.exposure_time == pytest.approx(1e-3)
        assert camera_proxy.imsize == 1024
        assert camera_proxy.readout_area == "Full"
        assert camera_proxy.camera_detector == "BM-Ceta"
        assert camera_proxy.frame_combining == 1
        assert bool(camera_proxy.electron_counting) is True
        assert camera_proxy.output_format == ".h5"

    def test_write_all_settings(self, camera_proxy):
        camera_proxy.exposure_time = 0.5
        camera_proxy.imsize = 2048
        camera_proxy.readout_area = "Half"
        camera_proxy.camera_detector = "EF-Ceta"
        camera_proxy.frame_combining = 6
        camera_proxy.electron_counting = False
        camera_proxy.output_format = ".tiff"

        assert camera_proxy.exposure_time == pytest.approx(0.5)
        assert camera_proxy.imsize == 2048
        assert camera_proxy.readout_area == "Half"
        assert camera_proxy.camera_detector == "EF-Ceta"
        assert camera_proxy.frame_combining == 6
        assert bool(camera_proxy.electron_counting) is False
        assert camera_proxy.output_format == ".tiff"

    @pytest.mark.parametrize(
        ("attribute", "value"),
        [
            ("readout_area", "Third"),
            ("camera_detector", "Unknown-Camera"),
            ("output_format", ".png"),
        ],
    )
    def test_rejects_unsupported_string_settings(
        self, camera_proxy, attribute, value
    ):
        with pytest.raises(tango.DevFailed):
            setattr(camera_proxy, attribute, value)


class TestCAMERAState:
    def test_initial_state_is_on(self, camera_proxy):
        assert camera_proxy.state() == tango.DevState.ON
