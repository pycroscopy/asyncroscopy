"""
Shared pytest fixtures for Tango device tests.

Starts BOTH the detector device(s) and the STEMMicroscope device in ONE Tango
test device server using MultiDeviceTestContext, so the STEMMicroscope can
create DeviceProxy connections to detectors by device name.

This avoids:
- "No proxy found for detector 'scan'. Available: []"
- Needing a real Tango DB
- Flaky multi-context issues from spinning up multiple separate servers
"""

import numpy as np
import pytest
import tango
from tango.test_context import MultiDeviceTestContext

# Import device classes to test
from asyncroscopy.instruments.electron_microscope.detectors.camera import CAMERA
from asyncroscopy.instruments.electron_microscope.detectors.eds import EDS
from asyncroscopy.instruments.electron_microscope.detectors.flucam import FLUCAM
from asyncroscopy.instruments.electron_microscope.hardware.scan import SCAN
from asyncroscopy.instruments.electron_microscope.hardware.stage import STAGE
from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
from asyncroscopy.instruments.electron_microscope.auto_script import AutoScriptMicroscope
from asyncroscopy.data.data import DATA


class FakeAdornedImage:
    def __init__(self, data: np.ndarray):
        self.data = data


# We use DigitalTwin as our simulated microscope for all tests.

@pytest.fixture(scope="session")
def data_save_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data-acquisitions")


@pytest.fixture(scope="session")
def tango_ctx(data_save_dir):
    """
    One Tango device server hosting SCAN + STEMMicroscope together.

    Device names here MUST match what you put into STEMMicroscope properties.
    """
    devices_info = [
        {
            "class": SCAN,
            "devices": [
                {
                    "name": "asyncroscopy/scan/default",
                    "properties": {
                        # put SCAN defaults here if you want
                        # e.g. "dwell_time": 2e-6  (only if it's a device_property)
                    },
                }
            ],
        },
        {
            "class": EDS,
            "devices": [
                {
                    "name": "asyncroscopy/eds/default",
                    "properties": {},
                }
            ],
        },
        {
            "class": CAMERA,
            "devices": [
                {
                    "name": "asyncroscopy/camera/default",
                    "properties": {},
                }
            ],
        },
        {
            "class": FLUCAM,
            "devices": [
                {
                    "name": "asyncroscopy/flucam/default",
                    "properties": {},
                }
            ],
        },
        {
            "class": STAGE,
            "devices": [
                {
                    "name": "asyncroscopy/stage/default",
                    "properties": {},
                }
            ],
        },
        {
            "class": DATA,
            "devices": [
                {
                    "name": "asyncroscopy/data/default",
                    "properties": {},
                }
            ],
        },
        {
            "class": DigitalTwin,
            "devices": [
                {
                    "name": "asyncroscopy/digitaltwin/default",
                    "properties": {
                        "scan_device_address": "asyncroscopy/scan/default",
                        "eds_device_address": "asyncroscopy/eds/default",
                        "stage_device_address": "asyncroscopy/stage/default",
                        "camera_device_address": "asyncroscopy/camera/default",
                        "flucam_device_address": "asyncroscopy/flucam/default",
                        "acquisition_save_directory": str(data_save_dir),
                    },
                }
            ],
        },

        {
            "class": AutoScriptMicroscope,
            "devices": [
                {
                    "name": "asyncroscopy/autoscriptmicroscope/default",
                    "properties": {
                        "testing_mode_bool": True,
                        "scan_device_address": "asyncroscopy/scan/default",
                        "camera_device_address": "asyncroscopy/camera/default",
                        "flucam_device_address": "asyncroscopy/flucam/default",
                        "eds_device_address": "asyncroscopy/eds/default",
                        "stage_device_address": "asyncroscopy/stage/default",
                        "data_device_address": "asyncroscopy/data/default",
                    },
                }
            ],
        },
    ]

    # Keep one in-process context for the whole session. Starting multiple
    # in-process Tango contexts in one interpreter can segfault in PyTango.
    ctx = MultiDeviceTestContext(devices_info, process=False)
    with ctx:
        data = tango.DeviceProxy(ctx.get_device_access("asyncroscopy/data/default"))
        data.save_path = str(data_save_dir)
        yield ctx



@pytest.fixture(scope="session")
def scan_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/scan/default"))


@pytest.fixture(scope="session")
def twin_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwin/default"))


@pytest.fixture(scope="session")
def eds_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/eds/default"))


@pytest.fixture(scope="session")
def camera_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/camera/default"))


@pytest.fixture(scope="session")
def flucam_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/flucam/default"))


@pytest.fixture(scope="session")
def stage_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/stage/default"))


@pytest.fixture(scope="session")
def data_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/data/default"))


@pytest.fixture(scope="session")
def auto_script_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/autoscriptmicroscope/default"))



@pytest.fixture
def patched_single_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch AutoScriptMicroscope._acquire_scanned_image so acquire_scanned_image() works
    without AutoScript/hardware.
    """
    def fake_acquire(self, imsize: int, dwell_time: float, detector_list: list = ["haadf"], scan_region: list[float] = [0.0, 0.0, 1.0, 1.0], output_format: str = ".h5"):
        # Deterministic image makes tests stable
        arr = np.arange(imsize * imsize, dtype=np.uint16)
        return FakeAdornedImage(arr.reshape(imsize, imsize))

    monkeypatch.setattr(
        AutoScriptMicroscope,
        "_acquire_scanned_image",
        fake_acquire,
    )
    monkeypatch.setattr(
        DigitalTwin,
        "_acquire_scanned_image",
        fake_acquire,
    )


@pytest.fixture
def patched_path_acquisition(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls = []

    def fake_acquire(self, imsize: int, dwell_time: float, detector_list: list = ["haadf"], scan_region: list[float] = [0.0, 0.0, 1.0, 1.0], output_format: str = ".h5"):
        calls.append(
            {
                "imsize": imsize,
                "dwell_time": dwell_time,
                "detector_list": list(detector_list),
                "scan_region": list(scan_region),
            }
        )
        path = tmp_path / f"stem_{imsize}.h5"
        path.write_bytes(b"fake-h5")
        return str(path)

    monkeypatch.setattr(AutoScriptMicroscope, "_acquire_scanned_image", fake_acquire)
    return calls


@pytest.fixture
def patched_scanned_path_acquisition(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls = []

    def fake_acquire(self, imsize: int, dwell_time: float, detector_list: list = ["haadf"], scan_region: list[float] = [0.0, 0.0, 1.0, 1.0], output_format: str = ".h5"):
        calls.append(
            {
                "imsize": imsize,
                "dwell_time": dwell_time,
                "detector_list": list(detector_list),
                "scan_region": list(scan_region),
            }
        )
        path = tmp_path / f"stem_{imsize}.h5"
        path.write_bytes(b"fake-stem-h5")
        return str(path)

    monkeypatch.setattr(AutoScriptMicroscope, "_acquire_scanned_image", fake_acquire)
    return calls


@pytest.fixture
def patched_scanned_data_acquisition(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_acquire(
        self,
        imsize: int,
        dwell_time: float,
        detector: str,
        scan_region: list[float],
    ):
        calls.append(
            {
                "imsize": imsize,
                "dwell_time": dwell_time,
                "detector": detector,
                "scan_region": list(scan_region),
            }
        )
        return "fake-stem-data-key"

    monkeypatch.setattr(AutoScriptMicroscope, "_acquire_scanned_data_advanced", fake_acquire)
    return calls


@pytest.fixture
def patched_camera_path_acquisition(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls = []

    def fake_acquire(self, imsize: int, exposure_time: float, detector: str, readout_area: str):
        calls.append(
            {
                "imsize": imsize,
                "exposure_time": exposure_time,
                "detector": detector,
                "readout_area": readout_area,
            }
        )
        path = tmp_path / f"camera_{imsize}.h5"
        path.write_bytes(b"fake-camera-h5")
        return str(path)

    monkeypatch.setattr(AutoScriptMicroscope, "_acquire_camera_image", fake_acquire)
    return calls


@pytest.fixture
def patched_spectrum_path_acquisition(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls = []

    def fake_acquire(self, detector_name: str, exposure_time: float):
        calls.append({"detector_name": detector_name, "exposure_time": exposure_time})
        path = tmp_path / f"spectrum_{detector_name}.h5"
        path.write_bytes(b"fake-spectrum-h5")
        return str(path)

    monkeypatch.setattr(AutoScriptMicroscope, "_acquire_spectrum", fake_acquire)
    return calls
