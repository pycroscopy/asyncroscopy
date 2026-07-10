# Adding MCP Capabilities

Asyncroscopy has one MCP server: `asyncroscopy.mcp.mcp_server.MCPServer`. It is
configured by [configs/mcp.yaml](../../configs/mcp.yaml) and started with
`startup_scripts/run_mcp.py`.

The server exposes two kinds of tools:

- Tango commands discovered from the live Tango database.
- Native MCP helpers defined directly on `MCPServer`.

## Discovery Model

MCP does not scan Python packages or require a custom server subclass. At
startup it:

1. Connects to the Tango database from `tango.host` and `tango.port`.
2. Calls `get_device_exported("*")`.
3. Opens each exported device with `DeviceProxy`.
4. Reads each device's `command_list_query()`.
5. Registers every non-blocked command as a FastMCP tool.

That means the Tango database is the source of truth. If a device is registered,
exported, and not blocked, MCP can expose its commands.

## Add A Device Command

Add a Tango command to the relevant device class:

```python
from tango.server import Device, command


class CAMERA(Device):
    @command(dtype_in=float, dtype_out=str)
    def set_exposure(self, exposure_ms: float) -> str:
        self.exposure_ms = exposure_ms
        return f'exposure set to {exposure_ms} ms'
```

Then start the Tango/device stack and MCP:

```bash
uv run startup_scripts/run_servers.py --yaml configs/Spectra300.yaml
uv run startup_scripts/run_mcp.py --yaml configs/mcp.yaml
```

If the device is live, MCP discovers `CAMERA.set_exposure` automatically.

## Add A Native MCP Helper

Add native helpers directly to `MCPServer` when the behavior is not a Tango
device command. Current examples are `list_devices` and `get_data_from_key`.

```python
from fastmcp import tool


class MCPServer:
    @tool()
    def get_data_from_key(self, key: str) -> dict:
        ...
```

Use this path for cross-device helpers, data lookups, or MCP-specific
convenience commands. Do not create a separate subclass for normal asyncroscopy
behavior.

## Block Commands

Use [configs/mcp.yaml](../../configs/mcp.yaml):

```yaml
mcp:
  blocked_classes:
    - DataBase
    - DServer
  blocked_functions:
    "*":
      - Init
      - Kill
      - RestartServer
    DATA:
      - stop_tiled_server
```

`blocked_classes` hides whole Tango classes. `blocked_functions` can hide global
command names, class-specific command names, or fully qualified
`Class.command` entries.

## Test Changes

Use the focused MCP tests:

```bash
uv run pytest tests/test_mcp_server.py tests/test_run_servers.py
```

Add or adjust tests when you change tool discovery, command filtering, argument
mapping, or native MCP helpers.
