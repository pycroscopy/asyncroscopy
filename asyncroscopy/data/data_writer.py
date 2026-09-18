"""Simple HDF5 acquisition writer."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import h5py
import numpy as np

DEFAULT_ACQUISITION_DIR = "outputs/tiled_acquisitions"


def acquisition_filename(
    device,
    acquisition_type: str,
    detector: str,
    data_server=None,
    extension: str = "h5",
) -> Path:
    """Create a timestamped acquisition filename(Tiled uses this to retrieve the data)"""
    save_directory = DEFAULT_ACQUISITION_DIR
    try:
        save_directory = device.acquisition_save_directory
    except AttributeError:
        pass
    if data_server is not None:
        save_directory = data_server.save_path

    directory = Path(save_directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    return directory / f"{acquisition_type}_{detector}_{stamp}.{extension.lower().lstrip('.')}"


def save_acquisition(
    device,
    data_server,
    acquisition_type: str,
    detectors,
    data,
    dataset_name: str = "image",
    dataset_attrs: dict | list[dict] | None = None,
    file_attrs: dict | None = None,
) -> str:
    """Save one HDF5 acquisition and return its DATA/Tiled key or local path."""
    detector_list = list(detectors) if isinstance(detectors, (list, tuple)) else [detectors]
    data_list = list(data) if isinstance(data, (list, tuple)) else [data]
    attrs_list = dataset_attrs if isinstance(dataset_attrs, list) else [dataset_attrs] * len(data_list)

    has_labeled_datasets = len(detector_list) > 1 or len(data_list) > 1
    detector_label = "_".join([str(detector) for detector in detector_list])
    # Naming statergy which is acts as a "key" for Tiled 
    path = acquisition_filename(device, acquisition_type, detector_label, data_server)


    datasets = []
    # Detector wise, Naming statergy which is acts as a "key" for Tiled 
    for index, source in enumerate(data_list):
        detector = str(detector_list[index]) if index < len(detector_list) else f"item_{index}"
        if dataset_name == "image" and isinstance(detectors, (list, tuple)):
            name = f"image/{detector}"
        else:
            name = f"{dataset_name}/{detector}" if has_labeled_datasets else dataset_name
        attrs = {"acquisition_type": acquisition_type, "detector": detector}
        attrs.update(attrs_list[index] or {})
        datasets.append({"name": name, "source": source, "attrs": attrs})

    save_acquisition_hdf5(path, datasets, file_attrs=file_attrs)
    return data_server.register_path(str(path)) if data_server is not None else str(path)


def save_acquisition_hdf5(path: str | Path, datasets: list[dict], file_attrs: dict | None = None) -> None:
    """Save one acquisition event to one HDF5 file.
    
    Currently supported for AdornedImage datastrucutre, which comes from ThermoFisher TEM's
    """
    with h5py.File(path, "w", track_order=True) as h5:
        for key, value in (file_attrs or {}).items():
            h5.attrs[key] = value if isinstance(value, (str, int, float, bool, np.number)) else json.dumps(value)

        for item in datasets:
            # data is list of AdornedImage(Note: vendor is Thermofisher TEM's), which is typically imported as from autoscript_tem_microscope_client.structures import AdornedImage
            source = item.get("source", item.get("data"))
            data = source.data if hasattr(source, "data") and not isinstance(source, np.ndarray) else source
            name = item["name"]
            if "/" in name:
                group_name, dataset_name = name.rsplit("/", 1)
                group = h5[group_name] if group_name in h5 else h5.create_group(group_name, track_order=True)
                dset = group.create_dataset(dataset_name, data=data, compression=None)
            else:
                dset = h5.create_dataset(name, data=data, compression=None)

            for key, value in item.get("attrs", {}).items():
                dset.attrs[key] = value if isinstance(value, (str, int, float, bool, np.number)) else json.dumps(value)

            metadata = getattr(source, "metadata", None)
            metadata_xml = getattr(metadata, "metadata_as_xml", None) # typically called as : AdornedImage.metadata.metadata_as_xml
            if metadata_xml:
                root = ET.fromstring(metadata_xml)
                for elem in root.iter():
                    if elem.text and elem.text.strip():
                        key = elem.tag
                        if key in dset.attrs:
                            key = f"{key}_{len(dset.attrs)}"
                        dset.attrs[key] = elem.text.strip()
            elif isinstance(metadata, dict):
                for key, value in metadata.items():
                    dset.attrs[key] = value if isinstance(value, (str, int, float, bool, np.number)) else json.dumps(value)
