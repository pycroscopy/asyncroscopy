"""Tiled adapter that serves SciFiReaders-readable files as sidpy datasets.

Each file is one Tiled container; each sidpy dataset the reader returns is one
array node in it (unchanged). Each array node's metadata has three keys,
mirroring the three groups a sidpy.Dataset keeps:

    sidpy_metadata     what is needed to rebuild the dataset: title, quantity,
                       units, data_type, modality, source, one entry per axis
                       (name, values, units, quantity, dimension_type) and, for
                       point clouds, the measurement coordinates.
    metadata           the dataset's own ``metadata`` dict (e.g. dwell_time).
    original_metadata  the vendor metadata as the reader returned it.

Client side, ``tiled_array_to_sidpy`` rebuilds one channel and
``tiled_container_to_sidpy`` rebuilds a whole file as {key: sidpy.Dataset}.
"""

from collections.abc import Mapping
from enum import Enum

import numpy as np
import sidpy
from SciFiReaders import AutoReader
from tiled.adapters.array import ArrayAdapter
from tiled.adapters.mapping import MapAdapter
from tiled.utils import path_from_uri

from SciFiReaders.auto_reader import _EXTENSION_READER_MAP

MIMETYPES_BY_FILE_EXT = {ext: "application/x-sidpy" for ext in [*_EXTENSION_READER_MAP, ".h5", ".hdf5"]}


def _jsonable(value):
    """Convert to plain Python types: numpy -> lists/scalars, bytes -> text, enums -> names."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Enum):
        return value.name
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def sidpy_to_tiled_metadata(dataset: sidpy.Dataset) -> dict:
    """Pack one sidpy dataset into the three-key Tiled metadata dict."""
    dimensions = [
        {
            "name": axis.name,
            # Full coordinates, not offset/scale: axes are not necessarily linear.
            "values": np.asarray(axis.values),
            "units": axis.units,
            "quantity": axis.quantity,
            "dimension_type": axis.dimension_type,
        }
        for axis in (dataset._axes[index] for index in range(dataset.ndim))
    ]
    # Point clouds: axis 0 only counts points; where each one was measured is
    # dataset.point_cloud["coordinates"], an (N, d) array. base_image is data,
    # not metadata, so it is not carried.
    point_cloud = dict(getattr(dataset, "point_cloud", None) or {})
    point_cloud.pop("base_image", None)

    sidpy_metadata = {
        "title": dataset.title,
        "quantity": dataset.quantity,
        "units": dataset.units,
        "data_type": dataset.data_type,
        "modality": dataset.modality,
        "source": dataset.source,
        "dimensions": dimensions,
        "point_cloud": point_cloud or None,
    }
    return {
        "sidpy_metadata": _jsonable(sidpy_metadata),
        "metadata": _jsonable(dict(dataset.metadata or {})),
        "original_metadata": _jsonable(dict(dataset.original_metadata or {})),
    }


def tiled_array_to_sidpy(node) -> sidpy.Dataset:
    """Rebuild one sidpy dataset from a Tiled array node, e.g. client[key]["Channel_000"]."""
    metadata = _jsonable(node.metadata)  # Tiled DictView -> plain nested dicts
    sidpy_metadata = metadata["sidpy_metadata"]

    dataset = sidpy.Dataset.from_array(node.read(), title=sidpy_metadata["title"])
    dataset.quantity = sidpy_metadata["quantity"]
    dataset.units = sidpy_metadata["units"]
    dataset.data_type = sidpy_metadata["data_type"]  # sidpy accepts the enum name
    dataset.modality = sidpy_metadata["modality"]
    dataset.source = sidpy_metadata["source"]
    for index, axis in enumerate(sidpy_metadata["dimensions"]):
        dataset.set_dimension(index, sidpy.Dimension(
            np.asarray(axis["values"]),
            name=axis["name"],
            quantity=axis["quantity"],
            units=axis["units"],
            dimension_type=axis["dimension_type"],
        ))
    point_cloud = sidpy_metadata.get("point_cloud")
    if point_cloud:
        dataset.point_cloud = {**point_cloud, "coordinates": np.asarray(point_cloud["coordinates"])}
    dataset.metadata = metadata["metadata"]
    dataset.original_metadata = metadata["original_metadata"]
    return dataset


def tiled_container_to_sidpy(node) -> dict:
    """Rebuild every channel of a Tiled container (one file), e.g. client[key]."""
    return {key: tiled_array_to_sidpy(node[key]) for key in node}


class SIDPYAdapter(MapAdapter):
    """Expose the sidpy datasets in a file as Tiled arrays."""

    @classmethod
    def from_uris(cls, data_uri, **kwargs):
        reader = AutoReader(str(path_from_uri(data_uri)))
        datasets = reader.read()
        adapter = cls({name: _array_adapter(dataset) for name, dataset in datasets.items()}, **kwargs)
        adapter.reader = reader
        return adapter

    @classmethod
    def from_catalog(cls, data_source, node, **kwargs):
        return cls.from_uris(data_source.assets[0].data_uri, metadata=node.metadata_, specs=node.specs)


def _array_adapter(dataset: sidpy.Dataset) -> ArrayAdapter:
    """One sidpy dataset -> one Tiled array node with the three-key metadata."""
    metadata = sidpy_to_tiled_metadata(dataset)
    dims = tuple(axis["name"] for axis in metadata["sidpy_metadata"]["dimensions"])
    return ArrayAdapter.from_array(dataset, dims=dims, metadata=metadata)