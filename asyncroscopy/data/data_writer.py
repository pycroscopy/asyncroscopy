"""Simple HDF5 acquisition writer."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import h5py
import numpy as np
import tifffile

DEFAULT_ACQUISITION_DIR = "outputs/tiled_acquisitions"


def acquisition_filename(
    device,
    acquisition_type: str,
    detector: str,
    data_server=None,
    extension: str = "h5",
) -> Path:
    """Create a timestamped acquisition filename."""
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


def _write_tiff(source, path: Path, attrs: dict | None = None) -> None:
    """Write one image to ``path`` as TIFF.

    AutoScript adorned images save themselves natively (embedding their own
    metadata); anything else is a raw array (JEOL/twin) written via tifffile,
    with ``attrs`` (e.g. JEOL's get_detectorsetting() dict) json-encoded into the
    TIFF ImageDescription tag so the provenance survives the format.
    """
    if hasattr(source, "save"): # adorned images from AutoScript have a save method
        source.save(str(path))
    else:
        array = source.data if hasattr(source, "data") and not isinstance(source, np.ndarray) else source
        tifffile.imwrite(str(path), np.asarray(array), description=json.dumps(attrs) if attrs else None)


def save_acquisition(
    device,
    data_server,
    acquisition_type: str,
    detectors,
    data,
    dataset_name: str = "image",
    dataset_attrs: dict | list[dict] | None = None,
    file_attrs: dict | None = None,
    output_format: str = ".h5",
) -> str:
    """Save one acquisition and return its DATA/Tiled key.

    ``.h5`` writes all detectors into one stacked file; ``.tiff`` writes one
    file per detector sharing a timestamp (TIFF cannot stack), registering each.
    """
    if output_format not in (".h5", ".tiff"):
        raise ValueError(f"Unsupported output_format {output_format!r}; expected '.h5' or '.tiff'")
    detector_list = list(detectors) if isinstance(detectors, (list, tuple)) else [detectors]
    data_list = list(data) if isinstance(data, (list, tuple)) else [data]
    attrs_list = dataset_attrs if isinstance(dataset_attrs, list) else [dataset_attrs] * len(data_list)

    if output_format == ".tiff":
        save_dir = data_server.save_path if data_server is not None else DEFAULT_ACQUISITION_DIR
        directory = Path(save_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{acquisition_type}_{datetime.now().strftime('%Y%m%dT%H%M%S%f')}"
        for index, (source, detector) in enumerate(zip(data_list, detector_list)):
            path = directory / f"{stem}_{detector}.tiff"
            _write_tiff(source, path, attrs_list[index])
            if data_server is not None:
                data_server.register_path(str(path))
        return stem

    has_labeled_datasets = len(detector_list) > 1 or len(data_list) > 1
    detector_label = "_".join([str(detector) for detector in detector_list])
    path = acquisition_filename(device, acquisition_type, detector_label, data_server)

    datasets = []
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
    """Save one acquisition event to one HDF5 file."""
    with h5py.File(path, "w", track_order=True) as h5:
        for key, value in (file_attrs or {}).items():
            h5.attrs[key] = value if isinstance(value, (str, int, float, bool, np.number)) else json.dumps(value)

        for item in datasets:
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
            metadata_xml = getattr(metadata, "metadata_as_xml", None)
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
