import json
import types

import h5py
import numpy as np

import tifffile

from asyncroscopy.data.data_writer import save_acquisition, save_acquisition_hdf5


class FakeDataServer:
    def __init__(self, save_path):
        self.save_path = str(save_path)

    def register_path(self, path: str) -> str:
        return path


def test_save_acquisition_hdf5_writes_data_and_xml_leaf_attrs(tmp_path):
    source = types.SimpleNamespace(
        data=np.array([[1, 2], [3, 4]], dtype=np.uint16),
        metadata=types.SimpleNamespace(
            metadata_as_xml="<root><detector>HAADF</detector><empty /></root>",
        ),
    )
    path = tmp_path / "acquisition.h5"

    save_acquisition_hdf5(
        path,
        [
            {
                "name": "images/HAADF",
                "source": source,
                "attrs": {"acquisition_type": "stem_image"},
            }
        ],
    )

    with h5py.File(path, "r") as h5:
        assert h5["images/HAADF"][()].tolist() == [[1, 2], [3, 4]]
        assert h5["images/HAADF"].attrs["acquisition_type"] == "stem_image"
        assert h5["images/HAADF"].attrs["detector"] == "HAADF"


def test_dict_dataset_attrs_are_written_as_attrs(tmp_path):
    data_server = FakeDataServer(tmp_path)
    # PyJEM returns raw pixels plus a get_detectorsetting()-style dict separately;
    # the dict is handed to save_acquisition via dataset_attrs and should land as
    # HDF5 attrs alongside the standard ones.
    metadata = {
        "ExposureTimeValue": 1000.0,
        "GainIndex": 5,
        "ImagingArea": {"X": 0, "Y": 0, "Width": 512, "Height": 512},
    }
    image = np.arange(4, dtype=np.uint16).reshape(2, 2)

    path = save_acquisition(object(), data_server, "stem_image", ["HAADF"], [image], dataset_attrs=[metadata])

    with h5py.File(path, "r") as h5:
        dset = h5["image/HAADF"]
        assert dset[()].tolist() == [[0, 1], [2, 3]]
        assert dset.attrs["ExposureTimeValue"] == 1000.0
        assert dset.attrs["GainIndex"] == 5
        # nested dict values are json-encoded, like other non-scalar attrs
        assert json.loads(dset.attrs["ImagingArea"]) == metadata["ImagingArea"]
        # the standard attrs are still written alongside the detector metadata
        assert dset.attrs["acquisition_type"] == "stem_image"
        assert dset.attrs["detector"] == "HAADF"


def test_save_acquisition_writes_scanned_images_as_ordered_image_detector_datasets(tmp_path):
    data_server = FakeDataServer(tmp_path)
    detectors = ["HAADF", "BF-S", "DF-S"]
    images = [np.full((2, 2), index, dtype=np.uint8) for index in range(len(detectors))]

    path = save_acquisition(object(), data_server, "stem_image", detectors, images)

    with h5py.File(path, "r") as h5:
        assert list(h5.keys()) == ["image"]
        assert list(h5["image"].keys()) == detectors
        for index, detector in enumerate(detectors):
            assert h5[f"image/{detector}"][()].tolist() == images[index].tolist()
            assert h5[f"image/{detector}"].attrs["detector"] == detector


def test_save_acquisition_tiff_writes_one_registered_file_per_detector(tmp_path):
    data_server = FakeDataServer(tmp_path)
    detectors = ["HAADF", "BF-S"]
    images = [np.full((2, 2), index, dtype=np.int16) for index in range(len(detectors))]
    # JEOL passes a get_detectorsetting()-style dict per detector; it should be
    # json-encoded into each TIFF's ImageDescription (nested dicts included).
    attrs = [{"GainIndex": index, "ImagingArea": {"Width": 512}} for index in range(len(detectors))]

    stem = save_acquisition(object(), data_server, "stem_image", detectors, images, dataset_attrs=attrs, output_format=".tiff")

    for index, detector in enumerate(detectors):
        path = tmp_path / f"{stem}_{detector}.tiff"
        assert path.exists()
        assert np.array_equal(tifffile.imread(path), images[index])
        with tifffile.TiffFile(path) as tif:
            assert json.loads(tif.pages[0].description) == attrs[index]
