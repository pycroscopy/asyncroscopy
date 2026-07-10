# Server Config Example

This file explains the YAML shape used by `startup_scripts/run_servers.py`.
Use it when adding a new instrument or making a site-specific startup config.

## Minimal Structure

```yaml
instrument:
  class_name: MyInstrument
  file: asyncroscopy/instruments/my_instrument.py
  description: "Short human-readable instrument name"
  hardware_host: localhost
  hardware_port: 9095
  timeout_seconds: 120

devices:
  data:
    module_name: asyncroscopy.data.data
  scan:
    module_name: asyncroscopy.instruments.electron_microscope.hardware.scan

tango:
  host: localhost
  port: 9094
  reset_database_file: false

tiled:
  host: localhost
  port: 9091
  acquisition_dir: outputs/tiled_acquisitions
  autostart: true

device_timeout_seconds: 120
```

## `instrument`

`instrument` selects the Tango device server that represents the microscope,
simulation, or other controllable instrument.

- `class_name`: Tango device class defined in the Python file.
- `file`: Python file containing that class. Use a path relative to the project
  root.
- `description`: optional label for humans.
- `hardware_host`: host or IP for a real hardware control server.
- `hardware_port`: port for a real hardware control server.
- `timeout_seconds`: hardware connection timeout passed to the instrument
  device as `hardware_timeout_seconds`.

For simulated instruments, omit `hardware_host` and `hardware_port`:

```yaml
instrument:
  class_name: DigitalTwin
  file: asyncroscopy/instruments/electron_microscope/digital_twin.py
  description: "Digital Twin Simulation Environment"
  timeout_seconds: 120
```

The runner registers the selected instrument as:

```text
asyncroscopy/instrument/default
```

and starts it with:

```text
<class_name> instrument_instance
```

## `devices`

`devices` lists support Tango device servers that must start before the
instrument. Each key becomes part of the Tango device name:

```yaml
devices:
  camera:
    module_name: asyncroscopy.instruments.electron_microscope.detectors.camera
```

This registers:

```text
asyncroscopy/camera/default
```

## `tango`

`tango` controls the Tango database endpoint used by clients and device
servers.

- `host`: database host.
- `port`: database port.
- `reset_database_file`: optional. When true, local `tango_database.db` files are
  deleted before the database starts.

## `tiled`

`tiled` controls the DATA-managed Tiled HTTP server.

- `host`: host where DATA starts Tiled.
- `port`: Tiled HTTP port.
- `acquisition_dir`: directory where acquisitions are written and served.
- `autostart`: when true, DATA starts the Tiled server during startup.

## `device_timeout_seconds`

`device_timeout_seconds` is the readiness wait for each Tango device to answer
`ping()`. It is separate from `instrument.timeout_seconds`, which is for the
instrument's own hardware connection.

## Adding A New Instrument

1. Create the instrument Python file, for example
   `asyncroscopy/instruments/my_instrument.py`.
2. Define one Tango device class in that file, for example `MyInstrument`.
3. Add any hardware properties the class needs, using generic names when
   possible: `hardware_host`, `hardware_port`, and `hardware_timeout_seconds`.
4. Create a YAML config that points `instrument.file` to the new file and
   `instrument.class_name` to the class.
5. Include the support devices your instrument expects under `devices`.
6. Start it with:

```bash
uv run startup_scripts/run_servers.py --yaml configs/my-instrument.yaml
```
