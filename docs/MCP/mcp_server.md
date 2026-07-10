# Asyncroscopy MCP Server

The MCP server is a FastMCP HTTP bridge over the live Tango database. It should
start after the Tango database, support devices, Tiled, and selected instrument
are ready.

## Start It

Start the device stack first:

```bash
uv run startup_scripts/run_servers.py --yaml configs/Spectra300.yaml
uv run startup_scripts/run_servers.py --yaml configs/STEMDigitalTwin.yaml
```

Then start MCP in another terminal or on the MCP computer:

```bash
uv run startup_scripts/run_mcp.py --yaml configs/mcp.yaml
```

GUI:

```bash
uv run python startup_guis/mcp_gui.py
```

The default endpoint is:

```text
http://127.0.0.1:8000/mcp
```

If MCP runs on another computer, set `tango.host` in `configs/mcp.yaml` to the
Tango database machine and set `mcp.http_host` to the MCP machine's bind address.
Use `0.0.0.0` when clients on other machines need to connect.

## YAML Contract

```yaml
tango:
  host: localhost
  port: 9094

mcp:
  name: Spectra300_MCP
  transport: streamable-http
  http_host: 127.0.0.1
  http_port: 8000
  data_device_address: asyncroscopy/data/default
  quiet: true
  blocked_classes:
    - DataBase
    - DServer
  blocked_functions:
    "*":
      - Init
      - Kill
      - RestartServer
```

`startup_scripts/run_mcp.py` maps this config directly to:

```bash
uv run python -m asyncroscopy.mcp.mcp_server ...
```

## MCP GUI

`startup_guis/mcp_gui.py` is a YAML launcher for MCP. It formats the current
selections into YAML, writes that YAML to `outputs/startup_configs/mcp_gui.yaml`,
and runs:

```bash
uv run python startup_scripts/run_mcp.py --yaml outputs/startup_configs/mcp_gui.yaml
```

The GUI includes a generated YAML preview, terminal output, **Start**, **Stop**,
and **Save current config**.

## Discovery

`MCPServer` connects to the Tango database, calls `get_device_exported("*")`,
opens each exported device with `DeviceProxy`, queries `command_list_query()`,
and registers every non-blocked Tango command as a FastMCP tool.

Tool signatures are built from Tango command types. NumPy values and Tango
`DevEncoded` payloads are normalized into JSON-safe results.

## Added MCP Tools

For device commands, add a Tango `@command` to the relevant device class. If the
device is registered and exported, MCP discovers it automatically.

For MCP-only behavior, add it directly to `MCPServer`. The required native tools
today are:

- `list_devices`
- `get_data_from_key`

`get_data_from_key` reads an acquired HDF5 DATA/Tiled key and returns dataset
metadata plus a small preview.

## Blacklisting

Use `blocked_classes` to hide whole Tango classes and `blocked_functions` to hide
commands. `blocked_functions` accepts global command names, fully qualified
`Class.command` entries, or class-specific lists:

```yaml
blocked_functions:
  "*":
    - Init
    - DATA.stop_tiled_server
  AutoScriptMicroscope:
    - Disconnect
```

## Manual MCP Only

If the Tango stack is already running:

```bash
uv run python -m asyncroscopy.mcp.mcp_server \
  --name Spectra300_MCP \
  --tango-host localhost \
  --tango-port 9094 \
  --transport streamable-http \
  --http-host 127.0.0.1 \
  --http-port 8000 \
  --data-device-address asyncroscopy/data/default \
  --blocked-classes-json '["DataBase", "DServer"]' \
  --blocked-functions-json '{"*": ["Init", "Kill", "RestartServer"]}'
```
