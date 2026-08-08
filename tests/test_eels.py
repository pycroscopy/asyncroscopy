from pathlib import Path

import h5py
import numpy as np
import pytest

from asyncroscopy.instruments.electron_microscope.detectors.eels import EELSBase
from asyncroscopy.instruments.electron_microscope.detectors.eels_gatan import EELS


class FakeDataServer:
    def __init__(self, save_path: Path):
        self.save_path = str(save_path)
        self.registered_paths = []

    def register_path(self, path: str) -> str:
        self.registered_paths.append(path)
        return Path(path).name


class FakeEELSProxy:
    def __init__(self):
        self.offset = None
        self.dispersion_index = None
        self.aperture_index = None
        self.spectrum_settings = None

    def initialize_eels(self):
        return None

    def set_eels_offset(self, offset):
        self.offset = offset
        return f"EELS {offset}"

    def get_eels_spectrum(self, exposure_time, number_of_frames):
        self.spectrum_settings = (exposure_time, number_of_frames)
        return [10, 20, 30], -5.0, 0.25

    def get_available_dispersions(self):
        return "0: 0.1, 1: 0.25"

    def get_eels_dispersion(self):
        return "1", "0.25"

    def set_eels_dispersion(self, dispersion_index):
        self.dispersion_index = dispersion_index

    def get_eels_aperture(self):
        return "2.5 mm"

    def set_eels_aperture(self, aperture_index):
        self.aperture_index = aperture_index

    def get_available_apertures(self):
        return "0: 2.5 mm, 1: 5 mm"


def make_gatan_eels() -> tuple[EELS, FakeEELSProxy]:
    device = EELS.__new__(EELS)
    proxy = FakeEELSProxy()
    device._eels_proxy = proxy
    device._exposure_time = 0.05
    device._number_of_frames = 3
    device._offset = float("nan")
    device._data_proxy = None
    return device, proxy


def test_gatan_eels_inherits_hardware_independent_tango_api():
    assert issubclass(EELS, EELSBase)
    assert "acquire_eels_spectrum" in EELS.TangoClassClass.cmd_list
    assert "get_eels_spectrum" not in EELS.TangoClassClass.cmd_list
    assert "offset" in EELS.TangoClassClass.attr_list


def test_gatan_eels_forwards_hardware_settings():
    device, proxy = make_gatan_eels()

    assert device._initialize_eels() is True
    device.write_offset(-2.5)
    assert device.read_offset() == -2.5
    assert proxy.offset == -2.5

    assert device._get_available_dispersions() == "0: 0.1, 1: 0.25"
    assert device._get_eels_dispersion() == [1.0, 0.25]
    device._set_eels_dispersion(2)
    assert proxy.dispersion_index == 2

    assert device._get_eels_aperture() == "2.5 mm"
    device._set_eels_aperture(1)
    assert proxy.aperture_index == 1
    assert device._get_available_apertures() == "0: 2.5 mm, 1: 5 mm"


def test_gatan_eels_acquires_saves_and_registers_spectrum(tmp_path):
    device, proxy = make_gatan_eels()
    data_server = FakeDataServer(tmp_path)
    device._data_proxy = data_server

    key = device._acquire_eels_spectrum()

    assert key.startswith("spectrum_eels_")
    assert key.endswith(".h5")
    assert proxy.spectrum_settings == (0.05, 3)
    assert device.read_offset() == -5.0

    saved_path = tmp_path / key
    assert data_server.registered_paths == [str(saved_path)]
    with h5py.File(saved_path, "r") as h5:
        spectrum = h5["spectrum"]
        np.testing.assert_array_equal(spectrum[()], np.array([10, 20, 30]))
        assert spectrum.attrs["acquisition_type"] == "spectrum"
        assert spectrum.attrs["detector"] == "eels"
        assert spectrum.attrs["offset"] == pytest.approx(-5.0)
        assert spectrum.attrs["dispersion"] == pytest.approx(0.25)
        assert spectrum.attrs["energy_units"] == "eV"
        assert spectrum.attrs["dispersion_units"] == "eV/channel"
        assert spectrum.attrs["exposure_time"] == pytest.approx(0.05)
        assert spectrum.attrs["number_of_frames"] == 3
