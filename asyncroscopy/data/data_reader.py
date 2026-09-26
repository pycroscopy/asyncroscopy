"""Tiled adapter for NSID acquisition files."""

from SciFiReaders import AutoReader
from tiled.adapters.array import ArrayAdapter
from tiled.adapters.mapping import MapAdapter
from tiled.utils import path_from_uri

from SciFiReaders.auto_reader import _EXTENSION_READER_MAP

MIMETYPES_BY_FILE_EXT = {ext: "application/x-sidpy" for ext in [*_EXTENSION_READER_MAP, ".h5", ".hdf5"]}


class SIDPYAdapter(MapAdapter):
    """Expose the sidpy datasets in an NSID file as Tiled arrays."""

    @classmethod
    def from_uris(cls, data_uri, **kwargs):
        reader = AutoReader(str(path_from_uri(data_uri)))
        datasets = reader.read()
        adapter = cls({name: ArrayAdapter.from_array(dataset, dims=tuple(axis.name for axis in dataset._axes.values()), metadata={**dataset.metadata, "original_metadata": dataset.original_metadata}) for name, dataset in datasets.items()}, **kwargs)
        adapter.reader = reader
        return adapter

    @classmethod
    def from_catalog(cls, data_source, node, **kwargs):
        return cls.from_uris(data_source.assets[0].data_uri, metadata=node.metadata_, specs=node.specs)
