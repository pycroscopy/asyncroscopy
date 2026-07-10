# Running The Servers

`startup_scripts/run_servers.py` starts the Tango/device side of asyncroscopy. It
clears stale processes, starts the Tango database, registers devices, launches
device servers, starts the DATA-managed Tiled HTTP server, and starts the
selected instrument last.

MCP is started separately with `startup_scripts/run_mcp.py`; see
[mcp_server.md](../MCP/mcp_server.md).

## TL;DR

```bash
uv run startup_scripts/run_servers.py

uv run startup_scripts/run_servers.py --yaml configs/Spectra300.yaml
uv run startup_scripts/run_servers.py --yaml configs/STEMDigitalTwin.yaml
uv run startup_scripts/run_servers.py --yaml configs/ThinkPad-utkarsh-covalent-setup.yaml
```

GUI:

```bash
uv run python startup_guis/server_gui.py
```

- Press **Enter** at prompts to accept the value in brackets.
- Leave the terminal open while you work. Press **Ctrl+C** to stop the managed
  processes and the managed Tiled server.
- Start MCP in a second terminal or on another computer:

```bash
uv run startup_scripts/run_mcp.py --yaml configs/mcp.yaml
```

## What It Starts

| Order | Device(s) | Tango name |
|-------|-----------|------------|
| 1 | support devices | `asyncroscopy/{camera,corrector,data,eds,flucam,scan,stage}/default` |
| 2 | Tiled HTTP server | started through the `data` device |
| 3 | instrument | `asyncroscopy/instrument/default` |

The instrument starts last because it depends on the support devices. The runner
writes support-device addresses and hardware connection values into Tango
database properties before the instrument starts.

## Server GUI

`startup_guis/server_gui.py` is a small YAML launcher for the server stack. It
does not start servers directly. It formats the current selections into YAML,
writes that YAML to `outputs/startup_configs/server_gui.yaml`, and runs:

```bash
uv run python startup_scripts/run_servers.py --yaml outputs/startup_configs/server_gui.yaml
```

The GUI includes:

- YAML preview generated from the current selections.
- Terminal output from the running process.
- **Start** and **Stop** controls.
- **Save current config** to write a YAML file you can reuse later.

## Configs

Server startup configs live in [configs/](../../configs):

| File | For |
|------|-----|
| [configs/Spectra300.yaml](../../configs/Spectra300.yaml) | The real Spectra 300 stack. This is the default config. |
| [configs/ThinkPad-utkarsh-covalent-setup.yaml](../../configs/ThinkPad-utkarsh-covalent-setup.yaml) | A localhost-everywhere setup for local testing. |

Each server config has:

- `instrument:` for the selected instrument class, Python file, and hardware connection values.
- `devices:` for support device modules.
- `tango:` for the Tango database host, port, and optional database-file reset.
- `tiled:` for the DATA-managed Tiled HTTP server.
- `device_timeout_seconds:` for device readiness waits.

Device `class_name` defaults to the upper-cased key (`scan` becomes `SCAN`).
`instrument.hardware_host`, `instrument.hardware_port`, and
`instrument.timeout_seconds` become instrument device properties.

Set `tango.reset_database_file: true` to delete a stale local
`tango_database.db` / `Tango_database.db` before starting the Tango database.
The GUI exposes the same setting as **Delete tango_database.db before start**.

## MCP

MCP has its own config: [configs/mcp.yaml](../../configs/mcp.yaml).

Start the server stack first. Then, from the MCP machine:

```bash
uv run startup_scripts/run_mcp.py --yaml configs/mcp.yaml
```

If MCP runs on a different computer, edit `configs/mcp.yaml`:

- `tango.host` should point to the machine running the Tango database.
- `mcp.http_host` should be `127.0.0.1` for local-only clients or `0.0.0.0` when
  other machines need to connect.

## Prompts

Interactive mode is used only when `--yaml` is omitted. The defaults come from
`configs/Spectra300.yaml`.

| Prompt | What it controls |
|--------|------------------|
| Tango database host / port | `TANGO_HOST` for Tango clients and servers. |
| Tiled HTTP host / port | Where the DATA device starts Tiled. |
| Acquisition save path | Directory written and served by Tiled. |
| Start Tiled HTTP server | Whether DATA starts its managed Tiled server. |
| Clear old processes first | Frees stale Tango/Tiled ports and old device servers. |
| Start Tango database | Starts the DB or waits for an existing one. |
| Register devices | Adds device entries and instrument properties to the DB. |
| Device startup timeout seconds | How long to wait for each device to answer `ping()`. |
| Hardware host / port / timeout seconds | Connection values for the selected instrument. |

## Startup Stages

1. **Clearing old processes** frees the database/Tiled ports and kills old device
   server process groups.
2. **Starting Tango database** starts or waits for the database server.
3. **Registering devices** writes device entries and instrument properties.
4. **Starting device servers** starts support devices, Tiled, and then the
   selected instrument.
5. **Startup summary** prints `TANGO_HOST`, PIDs, ready times, and the Tiled URI.

## Manual Fallback

The runner automates this database-mode flow:

```bash
TANGO_HOST=localhost:9094 uv run python -m tango.databaseds.database 2

export TANGO_HOST=localhost:9094
uv run python -m asyncroscopy.instruments.electron_microscope.hardware.scan scan_instance
uv run python -m asyncroscopy.instruments.electron_microscope.auto_script instrument_instance
```

Manual device startup requires the devices to already be registered in Tango.
Let `run_servers.py` do registration once, then stop and relaunch individual
servers as needed.
