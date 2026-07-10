# DATA acquisition workflow

See more at https://github.com/bluesky/tiled.

`AutoScriptMicroscope` saves real AutoScript acquisitions on the microscope side
and returns the registered Tiled key through Tango. `asyncroscopy/data/data.py`
is the Tango data device for registering those files with the Tiled HTTP server.

The default format is one HDF5 file per acquisition event: each correlated
output is a dataset, with parsed AutoScript XML leaf metadata as HDF5 attributes.
Image acquisitions can instead write `.tiff` (Velox-compatible, one file per
detector) via `scan.output_format = ".tiff"`; spectra and STEM data are always
HDF5.

Vendors return two shapes and `save_acquisition` absorbs both: AutoScript adorned
images bundle pixels and metadata, while PyJEM (JEOL) returns raw pixels plus a
separate `get_detectorsetting()` dict passed as `dataset_attrs`. Metadata lands as
HDF5 attributes for `.h5`; for `.tiff`, adorned images save it natively while
raw-array vendors get the dict json-encoded into the TIFF `ImageDescription` tag.

Acquisition commands that feed this pipeline include `acquire_scanned_image`,
`acquire_spectrum`, `acquire_camera_image`, `acquire_flucam_image`, and
`acquire_scanned_data_advanced`.

What a command returns — and how you read it back — depends on the format:

- **`.h5`** (default): returns one Tiled key, read nested.
  `client[key]["image"]["HAADF"]` (one sub-dataset per detector), `["spectrum"]`,
  or `["stem_data"]`; single camera / flucam frames use `image`.
- **`.tiff`**: returns a shared *stem*. Rebuild one key per detector as
  `client[f"{stem}_{DET}.tiff"]`; `.read()` returns the array directly (no nesting).

## Notebook setup

Connect to the DATA Tango device once at the beginning of a workflow. The
`save_path` directory should be visible to the Tiled HTTP server, and the
microscope device should have `data_device_address` set to this Tango device.

```python
import json
import tango

data = tango.DeviceProxy("asyncroscopy/data/default")
data.set_timeout_millis(120_000)
data.host = "10.46.217.241"
data.port = 9091
data.save_path = "/path/served/by/tiled"
```

Changing `data.save_path` creates the directory and restarts a DATA-managed
Tiled HTTP server. Each acquisition is registered explicitly after it is
written; DATA does not run a filesystem watcher. `startup_scripts/run_servers.py` sets
the extended Tango timeout automatically.

Acquire as usual. With the default `.h5` the return value is the Tiled key; with
`.tiff` it is the shared stem (see the format contract above):

```python
key = mic.acquire_scanned_image(["HAADF", "BF-S"])   # .h5  → client[key]["image"]["HAADF"]
# scan.output_format = ".tiff"                        # .tiff → client[f"{key}_HAADF.tiff"].read()
```

## Server Roles

There are two data-related servers:

- `asyncroscopy/data/default` is the DATA Tango device server. It belongs to asyncroscopy and bridges notebooks or microscope devices to Tiled.
- `http://10.46.217.241:9091` is the Tiled HTTP data server. It indexes and serves files.

`startup_scripts/run_servers.py` starts the DATA device and its managed Tiled HTTP
server together. It also shuts down the managed Tiled server with the rest of
the server stack. To inspect the active directory, use:

```python
import json

json.loads(data.get_config())["tiled_server_serving"]
```

If the DATA device connects to an already-running external Tiled HTTP server,
it manually registers acquisitions but does not terminate that external server
during shutdown.

## Direct Tiled access

We currently access the server in the notebook like this:

from tiled.client import from_uri

client = from_uri("http://10.46.217.241:9091")

list(client) # should print out some folders and files
