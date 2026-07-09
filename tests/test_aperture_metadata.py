import json
import types

import h5py
import numpy as np

from asyncroscopy.instruments.electron_microscope.auto_script import AutoScriptMicroscope
from asyncroscopy.instruments.electron_microscope.hardware.aperture import (
    SimulatedApertureBackend,
)


class FakeDataServer:
    def register_path(self, path: str) -> str:
        return path


class FakeImage:
    data = np.array([[1, 2], [3, 4]], dtype=np.uint16)


class FakeAcquisition:
    def acquire_stem_images_advanced(self, settings):
        return [FakeImage()]


class FailingApertureAdapter:
    def list_mechanisms(self):
        raise RuntimeError("simulated aperture read failure")


def make_microscope(adapter) -> AutoScriptMicroscope:
    microscope = AutoScriptMicroscope.__new__(AutoScriptMicroscope)
    microscope._microscope = types.SimpleNamespace(acquisition=FakeAcquisition())
    microscope._detector_proxies = {"data": FakeDataServer()}
    microscope._aperture_adapter = adapter
    microscope._aperture_autoscript_available = False
    return microscope


def test_aperture_metadata_is_deterministic() -> None:
    microscope = make_microscope(SimulatedApertureBackend())

    first = microscope._get_aperture_metadata()
    second = microscope._get_aperture_metadata()

    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["mechanisms"] == [
        "condenser",
        "objective",
        "selected_area",
        "projector",
    ]
    assert first["selected_apertures"]["condenser"]["name"] == "50 um"
    assert first["insertion_states"]["condenser"] == "Inserted"
    assert first["errors"] == {}


def test_acquisition_hdf5_includes_aperture_metadata(monkeypatch, tmp_path) -> None:
    microscope = make_microscope(SimulatedApertureBackend())

    def fake_filename(device, acquisition_type, detector, data_server=None, extension="h5"):
        return tmp_path / "stem_with_apertures.h5"

    monkeypatch.setattr(
        "asyncroscopy.data.data_writer.acquisition_filename",
        fake_filename,
    )

    result = microscope._acquire_scanned_image(
        imsize=2,
        dwell_time=1e-6,
        detector_list=["haadf"],
        scan_region=[0.0, 0.0, 1.0, 1.0],
    )

    with h5py.File(result, "r") as h5:
        aperture_metadata = json.loads(h5.attrs["apertures"])
    assert aperture_metadata["source"] == "simulation"
    assert aperture_metadata["selected_apertures"]["condenser"]["name"] == "50 um"
    assert aperture_metadata["insertion_states"]["objective"] == "Retracted"


def test_aperture_read_failure_does_not_fail_acquisition(monkeypatch, tmp_path) -> None:
    microscope = make_microscope(FailingApertureAdapter())

    def fake_filename(device, acquisition_type, detector, data_server=None, extension="h5"):
        return tmp_path / "stem_with_aperture_error.h5"

    monkeypatch.setattr(
        "asyncroscopy.data.data_writer.acquisition_filename",
        fake_filename,
    )

    result = microscope._acquire_scanned_image(
        imsize=2,
        dwell_time=1e-6,
        detector_list=["haadf"],
        scan_region=[0.0, 0.0, 1.0, 1.0],
    )

    with h5py.File(result, "r") as h5:
        aperture_metadata = json.loads(h5.attrs["apertures"])
        assert h5["image/HAADF"].shape == (2, 2)
    assert aperture_metadata["mechanisms"] == []
    assert "simulated aperture read failure" in aperture_metadata["errors"]["adapter"]
