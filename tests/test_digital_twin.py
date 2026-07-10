"""
Tests for the DigitalTwin Tango device.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

import h5py
import numpy as np
import pytest
import tango

class TestDigitalTwin:

    def test_state_is_on(self, twin_proxy: tango.DeviceProxy):
        assert twin_proxy.state() == tango.DevState.ON

    def test_manufacturer_is_digital_twin(self, twin_proxy: tango.DeviceProxy):
        assert twin_proxy.manufacturer == "UTKTeam"

    def test_defocus_commands_round_trip(self, twin_proxy: tango.DeviceProxy):
        twin_proxy.set_defocus(8e-9)

        assert twin_proxy.get_defocus() == pytest.approx(8e-9)

    def test_get_image_returns_saved_hdf5(self, twin_proxy: tango.DeviceProxy, scan_proxy: tango.DeviceProxy):
        scan_proxy.imsize = 32
        scan_proxy.dwell_time = 1e-6

        saved_path = Path(twin_proxy.acquire_scanned_image(["haadf"]))

        assert saved_path.suffix == ".h5"
        assert saved_path.exists()
        with h5py.File(saved_path, "r") as h5:
            image = h5["image/HAADF"][()]
            assert image.shape == (32, 32)
            assert h5["image/HAADF"].attrs["acquisition_type"] == "stem_image"
            assert h5["image/HAADF"].attrs["detector"] == "HAADF"

    def test_stage_navigation_changes_and_restores_view(
        self,
        twin_proxy: tango.DeviceProxy,
        scan_proxy: tango.DeviceProxy,
        monkeypatch: pytest.MonkeyPatch,
    ):
        def fake_stage_render(self, imsize: int, dwell_time: float, detector_list: list):
            self._sync_stage_from_proxy()
            stage_signal = int(round((self._stage_position[0] - self._stage_position[1]) * 1e10))
            return np.full((imsize, imsize), stage_signal, dtype=np.int16)

        from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin

        monkeypatch.setattr(DigitalTwin, "_render_stem_image", fake_stage_render)

        scan_proxy.imsize = 64
        scan_proxy.dwell_time = 1e-6

        twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
        with h5py.File(twin_proxy.acquire_scanned_image(["haadf"]), "r") as h5:
            image_a = h5["image/HAADF"][()]

        twin_proxy.move_stage([8e-9, -7e-9, 0.0, 0.0, 0.0])
        with h5py.File(twin_proxy.acquire_scanned_image(["haadf"]), "r") as h5:
            image_b = h5["image/HAADF"][()]
        assert not np.array_equal(image_a, image_b)

        twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
        with h5py.File(twin_proxy.acquire_scanned_image(["haadf"]), "r") as h5:
            image_a_again = h5["image/HAADF"][()]
        assert np.array_equal(image_a, image_a_again)

    def test_spectrum_is_repeatable_at_same_pose_and_beam(
        self,
        twin_proxy: tango.DeviceProxy,
        eds_proxy: tango.DeviceProxy,
    ):
        eds_proxy.exposure_time = 0.05
        twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
        twin_proxy.place_beam([0.45, 0.55])

        with h5py.File(twin_proxy.acquire_spectrum("eds"), "r") as h5:
            spec_1 = h5["spectrum"][()]
        with h5py.File(twin_proxy.acquire_spectrum("eds"), "r") as h5:
            spec_2 = h5["spectrum"][()]
        assert spec_1.tolist() == spec_2.tolist()
