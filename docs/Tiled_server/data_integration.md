# DATA acquisition workflow

Acquisitions save one HDF5 file per event. `save_acquisition` writes arrays and
metadata attributes, closes the file, then registers it through the DATA Tango
device.

AutoScript images supply pixels and XML metadata; JEOL supplies pixels and
separate `dataset_attrs`. XML element text becomes string attributes; other
non-scalar attribute values are JSON encoded.

DATA coordinates registration; the Tiled HTTP server indexes and serves files.
Each completed acquisition uses `DATA.register_acquisition_file()`, which returns
its resolvable filename key. Optional startup indexing uses
`DATA.register_existing_directory()` for existing files and returns JSON status.
Neither uses a filesystem watcher.
The save directory must be writable by the acquisition process and readable by
Tiled. Inspect `data.get_config()` for the URI, save path, and serving path.
Changing the save path restarts a DATA-managed server; external servers are not
stopped by DATA.

```python
import json
import tango
from tiled.client import from_uri

data = tango.DeviceProxy("asyncroscopy/data/default")
data.set_timeout_millis(120_000)
config = json.loads(data.get_config())
client = from_uri(config["uri"])

key = mic.acquire_scanned_image(["HAADF", "BF-S"])
image = client[key]["image"]["HAADF"].read()
```

Commands return the exact registered filename key. Camera images use
`client[key]["image"]`, spectra use `["spectrum"]`, and 4D-STEM uses
`["stem_data"]`. Without DATA, saving returns a local file path.

`startup_scripts/run_servers.py` starts DATA and its managed Tiled server,
sets the extended DATA timeout, and stops the managed server on shutdown.
