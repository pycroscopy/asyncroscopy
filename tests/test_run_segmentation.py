from pathlib import Path

import pytest

from startup_scripts import run_segmentation


class FakeDatabase:
    def __init__(self) -> None:
        self.device = None
        self.properties = None

    def add_device(self, device) -> None:
        self.device = device

    def put_device_property(self, name: str, properties: dict) -> None:
        self.properties = (name, properties)


def test_load_config(tmp_path: Path) -> None:
    config_path = tmp_path / "segmentation.yaml"
    config_path.write_text(
        """tango:
  host: gpu-host
  port: 9094
data_device_address: asyncroscopy/data/default
model_size: facebook/sam2-hiera-large
compute_device: cuda
device_timeout_seconds: 120
""",
        encoding="utf-8",
    )
    config = run_segmentation.load_config(config_path)

    assert config.tango == run_segmentation.TangoConfig("gpu-host", 9094)
    assert config.data_device_address == "asyncroscopy/data/default"
    assert config.model_size == "facebook/sam2-hiera-large"
    assert config.compute_device == "cuda"


def test_register_device(monkeypatch) -> None:
    database = FakeDatabase()
    monkeypatch.setattr(
        run_segmentation.tango,
        "Database",
        lambda host, port: database,
    )

    class FakeDeviceInfo:
        pass

    monkeypatch.setattr(run_segmentation.tango, "DbDevInfo", FakeDeviceInfo)
    config = run_segmentation.SegmentationConfig(
        tango=run_segmentation.TangoConfig("localhost", 9094)
    )

    run_segmentation.register_device(config)

    assert database.device.server == "SEGMENTATION/segment_instance"
    assert database.device._class == "SEGMENTATION"
    assert database.device.name == "asyncroscopy/segment/default"
    assert database.properties == (
        "asyncroscopy/segment/default",
        {
            "data_device_address": ["asyncroscopy/data/default"],
            "model_size": ["facebook/sam2-hiera-large"],
            "compute_device": ["auto"],
        },
    )


def test_missing_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_segmentation.load_config(tmp_path / "missing.yaml")
