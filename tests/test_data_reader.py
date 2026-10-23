"""End-to-end tests of the Tiled data path: acquisition file -> Tiled -> sidpy.

The Tiled server is built from the repository's own asyncroscopy/data/config.yml
and files are registered with the same arguments the DATA device uses, so these
tests exercise the real configuration rather than a copy of it.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import sidpy
import uvicorn
from SciFiReaders import NSIDReader
from tiled.catalog import from_uri as catalog_from_uri
from tiled.client import from_uri
from tiled.client.register import identity, register
from tiled.config import parse_configs
from tiled.server.app import build_app_from_config
from tiled.server.simple import ThreadedServer

from asyncroscopy.data import data_reader
from asyncroscopy.data.data import ONE_NODE_PER_FILE_WALKER
from asyncroscopy.data.data_reader import (
    MIMETYPES_BY_FILE_EXT,
    sidpy_to_tiled_metadata,
    tiled_array_to_sidpy,
    tiled_container_to_sidpy,
)
from asyncroscopy.data.data_writer import save_acquisition

CONFIG = Path(data_reader.__file__).with_name("config.yml")
ADAPTERS = {"application/x-sidpy": "asyncroscopy.data.data_reader:SIDPYAdapter"}


@pytest.fixture
def tiled(tmp_path, monkeypatch):
    """Serve tmp_path with Tiled exactly as DATA does; yields register(path) -> Tiled node."""
    catalog = tmp_path / ".asyncroscopy_tiled_catalog.db"
    monkeypatch.setenv("ASYNCROSCOPY_ACQUISITION_DIR", str(tmp_path))
    monkeypatch.setenv("ASYNCROSCOPY_TILED_CATALOG", str(catalog))
    catalog_from_uri(str(catalog), init_if_not_exists=True)  # what `tiled catalog init` does

    config = parse_configs(str(CONFIG))
    config.authentication.single_user_api_key = "secret"  # DATA's --api-key
    config.authentication.allow_anonymous_access = True  # DATA's --public
    server = ThreadedServer(uvicorn.Config(build_app_from_config(config), host="127.0.0.1", port=0, log_level="warning"))

    with server.run_in_thread() as url:
        writer = from_uri(url, api_key="secret")  # DATA's client
        reader = from_uri(url)  # anonymous, like notebooks and the MCP server

        def register_file(path: Path):
            asyncio.run(register(
                writer, str(path),
                adapters_by_mimetype=ADAPTERS,
                mimetypes_by_file_ext=MIMETYPES_BY_FILE_EXT,
                walkers=[ONE_NODE_PER_FILE_WALKER],
                key_from_filename=identity,
            ))
            if path.name not in reader:
                pytest.fail(f"Tiled did not register {path.name}; see the 'SKIPPED' log above")
            return reader[path.name]

        try:
            yield register_file
        finally:
            writer.context.close()
            reader.context.close()


class Instrument:
    """Stands in for an instrument device: save_acquisition only needs its save directory."""

    def __init__(self, directory):
        self.acquisition_save_directory = str(directory)


def calibrated_image(title="HAADF", shape=(8, 6), pixel_size=0.1e-9):
    """A sidpy image with spatial axes in metres, as the microscopes save it."""
    image = sidpy.Dataset.from_array(np.random.default_rng(0).random(shape).astype(np.float32),
                                     title=title, datatype="IMAGE", units="counts", quantity="intensity")
    image.set_dimension(0, sidpy.Dimension(np.arange(shape[0]) * pixel_size, "y", units="m", quantity="length", dimension_type="spatial"))
    image.set_dimension(1, sidpy.Dimension(np.arange(shape[1]) * pixel_size, "x", units="m", quantity="length", dimension_type="spatial"))
    image.metadata = {"dwell_time_s": 1e-6, "detector": title}
    return image


def read_locally(path):
    """The reference: what SciFiReaders returns when the file is opened directly."""
    reader = NSIDReader(str(path))
    datasets = reader.read()
    reader.close()
    return datasets


def assert_same_dataset(rebuilt, local):
    """Transport through Tiled must not change data, attributes or axes."""
    np.testing.assert_array_equal(np.asarray(rebuilt), np.asarray(local))
    for name in ("title", "quantity", "units", "data_type", "modality", "source"):
        assert getattr(rebuilt, name) == getattr(local, name), name
    for index in range(local.ndim):
        a, b = rebuilt._axes[index], local._axes[index]
        assert (a.name, a.units, a.quantity, a.dimension_type) == (b.name, b.units, b.quantity, b.dimension_type)
        np.testing.assert_array_equal(a.values, b.values)


def test_metadata_is_three_json_safe_groups():
    metadata = sidpy_to_tiled_metadata(calibrated_image())

    assert list(metadata) == ["sidpy_metadata", "metadata", "original_metadata"]
    json.dumps(metadata)  # numpy values and enums are already plain types
    assert metadata["sidpy_metadata"]["data_type"] == "IMAGE"
    assert [axis["units"] for axis in metadata["sidpy_metadata"]["dimensions"]] == ["m", "m"]


def test_channel_round_trip_keeps_calibration(tiled, tmp_path):
    path = Path(save_acquisition(Instrument(tmp_path), None, "stem_image", ["HAADF"], [calibrated_image()]))

    rebuilt = tiled_array_to_sidpy(tiled(path)["Channel_000"])

    assert_same_dataset(rebuilt, read_locally(path)["Channel_000"])
    assert rebuilt.x.units == "m"
    assert np.allclose(np.diff(rebuilt.x.values), 0.1e-9)  # 0.1 nm pixels
    assert rebuilt.metadata == {"dwell_time_s": 1e-6, "detector": "HAADF"}


def test_container_round_trip_keeps_every_channel(tiled, tmp_path):
    images = [calibrated_image("HAADF"), calibrated_image("BF")]
    path = Path(save_acquisition(Instrument(tmp_path), None, "stem_image", ["HAADF", "BF"], images))

    rebuilt = tiled_container_to_sidpy(tiled(path))
    local = read_locally(path)

    assert list(rebuilt) == list(local) == ["Channel_000", "Channel_001"]
    for key in local:
        assert_same_dataset(rebuilt[key], local[key])


def test_point_cloud_coordinates_round_trip(tiled, tmp_path):
    coordinates = np.random.default_rng(1).random((5, 2)) * 1e-6
    spectra = sidpy.Dataset.from_array(np.random.default_rng(2).random((5, 16)), title="iv",
                                       datatype="point_cloud", coordinates=coordinates)
    spectra.set_dimension(1, sidpy.Dimension(np.linspace(-5, 5, 16), "bias", units="V", quantity="voltage", dimension_type="spectral"))
    path = Path(save_acquisition(Instrument(tmp_path), None, "points", ["spec"], [spectra]))

    rebuilt = tiled_array_to_sidpy(tiled(path)["Channel_000"])

    assert rebuilt.data_type == sidpy.DataType.POINT_CLOUD
    np.testing.assert_allclose(rebuilt.point_cloud["coordinates"], coordinates)
    assert rebuilt.bias.units == "V"


def test_plain_hdf5_acquisition_is_served_as_before(tiled, tmp_path):
    # Instruments that still save raw arrays (twin EDS spectrum, Gatan EELS, ...)
    # must keep registering; they are served with Tiled's own HDF5 layout.
    spectrum = np.arange(16.0)
    path = Path(save_acquisition(Instrument(tmp_path), None, "spectrum", "eds", spectrum, dataset_name="spectrum"))

    node = tiled(path)

    assert list(node) == ["spectrum"]
    np.testing.assert_array_equal(node["spectrum"].read(), spectrum)