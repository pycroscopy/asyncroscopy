"""Simple HDF5 acquisition writer."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import h5py
import numpy as np
import sidpy
from pyNSID.io.hdf_io import write_nsid_dataset

DEFAULT_ACQUISITION_DIR = "outputs/tiled_acquisitions"


def save_acquisition(
    device,
    data_server,
    acquisition_type: str,
    detectors,
    data=None,
    dataset_name: str = "image",
    dataset_attrs: dict | list[dict] | None = None,
    file_attrs: dict | None = None,
    *,
    datasets: list[dict] | None = None,
) -> str:
    """Save HDF5 and return its DATA/Tiled key or local path.

    Sidpy inputs use NSID; unmigrated raw inputs retain their existing layout.
    Explicit ``datasets`` supply exact names, sources, and attributes.
    """
    detector_list = list(detectors) if isinstance(detectors, (list, tuple)) else [detectors]
    detector_label = "_".join([str(detector) for detector in detector_list])
    save_directory = (data_server.save_path if data_server is not None
                      else getattr(device, "acquisition_save_directory", DEFAULT_ACQUISITION_DIR))
    directory = Path(save_directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    path = directory / f"{acquisition_type}_{detector_label}_{timestamp}.h5"

    if datasets is None:
        sources = list(data) if isinstance(data, (list, tuple)) else [data]
        dataset_attributes = dataset_attrs if isinstance(dataset_attrs, list) else [dataset_attrs] * len(sources)
        include_detector_in_path = len(detector_list) > 1 or len(sources) > 1
        datasets = []
        for index, source in enumerate(sources):
            detector = str(detector_list[index]) if index < len(detector_list) else f"item_{index}"
            if dataset_name == "image" and isinstance(detectors, (list, tuple)):
                name = f"image/{detector}"
            else:
                name = f"{dataset_name}/{detector}" if include_detector_in_path else dataset_name
            attrs = {"acquisition_type": acquisition_type, "detector": detector}
            attrs.update(dataset_attributes[index] or {})
            datasets.append({"name": name, "source": source, "attrs": attrs})

    with h5py.File(path, "w", track_order=True) as h5:
        for key, value in (file_attrs or {}).items():
            h5.attrs[key] = value if isinstance(value, (str, int, float, bool, np.number)) else json.dumps(value)

        for index, item in enumerate(datasets):
            source = item.get("source", item.get("data"))
            if isinstance(source, sidpy.Dataset):
                channel = h5.create_group(f"Measurement_000/Channel_{index:03d}", track_order=True)
                write_nsid_dataset(source, channel, main_data_name="data", compression=None)
                continue
            data = source.data if hasattr(source, "data") and not isinstance(source, np.ndarray) else source
            name = item["name"]
            if "/" in name:
                group_name, dataset_name = name.rsplit("/", 1)
                group = h5[group_name] if group_name in h5 else h5.create_group(group_name, track_order=True)
                dataset = group.create_dataset(dataset_name, data=data, compression=None)
            else:
                dataset = h5.create_dataset(name, data=data, compression=None)

            for key, value in item.get("attrs", {}).items():
                dataset.attrs[key] = value if isinstance(value, (str, int, float, bool, np.number)) else json.dumps(value)

            metadata = getattr(source, "metadata", None)
            metadata_xml = getattr(metadata, "metadata_as_xml", None)
            if metadata_xml:
                root = ET.fromstring(metadata_xml)
                for element in root.iter():
                    if element.text and element.text.strip():
                        key = element.tag
                        if key in dataset.attrs:
                            key = f"{key}_{len(dataset.attrs)}"
                        dataset.attrs[key] = element.text.strip()
            elif isinstance(metadata, dict):
                for key, value in metadata.items():
                    dataset.attrs[key] = value if isinstance(value, (str, int, float, bool, np.number)) else json.dumps(value)

    return data_server.register_acquisition_file(str(path)) if data_server is not None else str(path)
