"""FastMCP bridge for asyncroscopy Tango devices."""

import argparse
import base64
import inspect
import json
import traceback
from pathlib import Path
from typing import Annotated, Any, Callable

import h5py
import numpy as np
from pydantic import Field

from tango import Database, DeviceProxy, CommandInfo, CmdArgType
from tango.utils import (
    TO_TANGO_TYPE,
    is_array_type,
    is_scalar_type,
    is_bool_type,
    is_float_type,
    is_int_type,
    is_str_type,
)

from fastmcp import FastMCP
from fastmcp.tools import tool, Tool
from fastmcp.server.server import Transport


class MCPServer:
    def __init__(
        self,
        name: str,
        tango_host: str,
        tango_port: int,
        blocked_functions: dict[str, list[str]],
        blocked_classes: list[str],
        data_device_address: str,
        verbose: bool = True,
    ):
        """
        Args:
            name (str): Display name for the MCP server instance.
            tango_host (str): Hostname of the Tango database server (e.g. "localhost").
            tango_port (int): Port of the Tango database server (e.g. 9094).
            blocked_functions: Command names to exclude, keyed by Tango class name.
                Use "*" for global blocks.
            blocked_classes: Tango device class names to skip entirely.
            data_device_address: Tango DATA device used by get_data_from_key.
            verbose (bool, optional): If True, print device discovery and tool registration
                progress to stdout. Defaults to True.
        """
        self.database = Database(tango_host, tango_port)
        self.mcp = FastMCP(name)

        self.blocked_functions = {key: list(value) for key, value in blocked_functions.items()}
        self.blocked_classes = list(blocked_classes)
        self._blocked_classes_normalized = {cls_name.lower() for cls_name in self.blocked_classes}
        self.data_device_address = data_device_address
        self.verbose = verbose
        self.tools: dict[str, dict[str, Callable]] = {}

    def _is_blocked_class(self, class_name: str) -> bool:
        """Return True when a Tango class should be filtered out."""
        return class_name.lower() in self._blocked_classes_normalized

    def _list_all_devices(self) -> list[str]:
        """List all devices exported in the Tango DB."""
        devices = self.database.get_device_exported("*")
        return list(devices.value_string)

    @staticmethod
    def _is_admin_device(device_name: str) -> bool:
        """Return True for Tango admin (dserver) devices."""
        return device_name.lower().startswith("dserver/")

    @tool()
    def list_devices(self) -> list[str]:
        """List available devices filtered by blocked classes."""
        all_devices = self._list_all_devices()
        available = []
        for device_name in all_devices:
            if self._is_admin_device(device_name):
                continue
            try:
                dev = DeviceProxy(device_name)
                dev_class = dev.info().dev_class
                if not self._is_blocked_class(dev_class):
                    available.append(device_name)
            except Exception:
                pass
        return available

    @tool()
    def get_data_from_key(
        self,
        key: str,
        max_values: int = 64,
        data_device_address: str | None = None,
    ) -> dict[str, Any]:
        """Read a DATA/Tiled HDF5 acquisition key and return dataset metadata plus small previews."""
        address = data_device_address or self.data_device_address
        data = DeviceProxy(address)
        config = json.loads(data.get_config())

        path = None
        for root in (config.get("save_path"), config.get("tiled_server_serving")):
            if not root:
                continue
            candidate = Path(root).expanduser() / key
            if candidate.exists():
                path = candidate
                break

        if path is None:
            raise FileNotFoundError(f"Could not resolve data key {key!r} from DATA device {address!r}")

        result: dict[str, Any] = {
            "key": key,
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            result["format"] = path.suffix.lower().lstrip(".") or "unknown"
            return result

        result["format"] = "hdf5"
        datasets: list[dict[str, Any]] = []
        with h5py.File(path, "r") as h5:
            result["attrs"] = self._hdf5_attrs_to_json(h5.attrs)

            def visit(name: str, obj: Any) -> None:
                if not isinstance(obj, h5py.Dataset):
                    return
                item: dict[str, Any] = {
                    "name": name,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "attrs": self._hdf5_attrs_to_json(obj.attrs),
                }
                preview = np.asarray(obj[()]).reshape(-1)[: max(0, int(max_values))]
                item["preview"] = self._numpy_to_python(preview)
                datasets.append(item)

            h5.visititems(visit)
        result["datasets"] = datasets
        return result

    @staticmethod
    def _hdf5_attrs_to_json(attrs: Any) -> dict[str, Any]:
        return {key: MCPServer._numpy_to_python(value) for key, value in attrs.items()}

    @staticmethod
    def _tango_type_to_python(cmd_type: CmdArgType) -> Any:
        if cmd_type == CmdArgType.DevVoid:
            return type(None)
        if cmd_type == CmdArgType.DevEncoded:
            return dict

        if is_scalar_type(cmd_type):
            if is_bool_type(cmd_type):
                return bool
            if is_float_type(cmd_type):
                return float
            if is_int_type(cmd_type):
                return int
            if is_str_type(cmd_type):
                return str

            candidates = [py_type for py_type, tango_type in TO_TANGO_TYPE.items() if tango_type == cmd_type and isinstance(py_type, type)]
            if not candidates:
                return Any
            for py_type in candidates:
                if py_type.__module__ == "builtins":
                    return py_type
            return candidates[0]

        if is_array_type(cmd_type):
            if is_bool_type(cmd_type, inc_array=True):
                return list[bool]
            if is_float_type(cmd_type, inc_array=True):
                return list[float]
            if is_int_type(cmd_type, inc_array=True):
                return list[int]
            if is_str_type(cmd_type, inc_array=True):
                return list[str]
            return list

        return Any

    @staticmethod
    def _numpy_to_python(obj: Any) -> Any:
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return MCPServer._numpy_to_python(obj.tolist())
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: MCPServer._numpy_to_python(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            conv = [MCPServer._numpy_to_python(v) for v in obj]
            return tuple(conv) if isinstance(obj, tuple) else conv
        return obj

    @staticmethod
    def _normalize_command_result(out_type: CmdArgType, result: Any) -> Any:
        """Convert Tango command output into JSON-safe data for MCP transport."""

        # Convert numpy types (including nested containers) to native Python types
        result = MCPServer._numpy_to_python(result)
        if out_type != CmdArgType.DevEncoded:
            return result

        if not isinstance(result, tuple) or len(result) != 2:
            return result

        metadata_raw, payload_raw = result
        if isinstance(metadata_raw, bytes):
            metadata_str = metadata_raw.decode("utf-8", errors="replace")
        else:
            metadata_str = str(metadata_raw)

        try:
            metadata = json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            metadata = metadata_str

        if isinstance(payload_raw, memoryview):
            payload_bytes = payload_raw.tobytes()
        elif isinstance(payload_raw, bytearray):
            payload_bytes = bytes(payload_raw)
        elif isinstance(payload_raw, bytes):
            payload_bytes = payload_raw
        else:
            payload_bytes = str(payload_raw).encode("utf-8")

        payload_b64 = base64.b64encode(payload_bytes).decode("ascii")
        return {
            "encoding": "base64",
            "metadata": metadata,
            "payload": payload_b64,
        }

    def _create_wrapper(
        self,
        func: Callable,
        cmd_info: CommandInfo,
        command_name: str,
        dev_class: str,
    ) -> Callable:
        """Create a wrapper function with a proper signature for a Tango command.

        Args:
            func: The raw Tango device command method
            cmd_info: The CommandInfo object from Tango
            command_name: The name of the command
            dev_class: The Tango device class name

        Returns:
            A wrapper function with a proper signature
        """
        in_type = cmd_info.in_type
        py_type = self._tango_type_to_python(in_type)
        in_desc = cmd_info.in_type_desc

        out_type = cmd_info.out_type
        py_return_type = self._tango_type_to_python(out_type)
        doc_lines = [
            f"Tango Device Class: {dev_class}",
            f"Tango Command: {command_name}",
            f"Input Type: {in_type.name}",
        ]
        if in_desc:
            doc_lines.append(f"Input Description: {in_desc}")
        doc_lines.append(f"Output Type: {out_type.name}")
        if cmd_info.out_type_desc:
            doc_lines.append(f"Output Description: {cmd_info.out_type_desc}")
        doc = "\n".join(doc_lines)

        if in_desc and in_desc.lower() not in (
            "uninitialised",
            "none",
            "",
            "uninitialized",
        ):
            # Sanitize description - remove newlines to prevent JSON schema breakage
            clean_desc = in_desc.replace("\n", " ").strip()
            arg_type = Annotated[py_type, Field(description=clean_desc)]
        else:
            arg_type = py_type

        if in_type == CmdArgType.DevVoid:

            def wrapper():
                result = func()
                return self._normalize_command_result(out_type, result)

            params = []
        else:
            def wrapper(arg):
                result = func(arg)
                return self._normalize_command_result(out_type, result)

            params = [inspect.Parameter("arg", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=arg_type)]

        wrapper.__annotations__ = {p.name: p.annotation for p in params}
        wrapper.__annotations__["return"] = py_return_type

        wrapper.__signature__ = inspect.Signature(parameters=params, return_annotation=py_return_type)
        wrapper.__doc__ = doc

        unique_name = f"{dev_class}_{command_name}".replace("/", "_").replace("-", "_")
        wrapper.__name__ = unique_name
        wrapper.__qualname__ = unique_name

        return wrapper

    def _find_tools(self) -> dict[str, dict[str, tuple[Callable, CommandInfo]]]:
        """Discover tools by querying Tango DB for devices and their commands.

        Returns a dict mapping dev_class -> command_name -> (func, cmd_info)
        """
        devices = self._list_all_devices()
        tools: dict[str, dict[str, tuple[Callable, CommandInfo]]] = {}
        for device_name in devices:
            if self._is_admin_device(device_name):
                continue
            try:
                dev = DeviceProxy(device_name)
                info = dev.info()
                dev_class = info.dev_class
            except Exception as exc:
                if self.verbose:
                    print(f"Skipping {device_name}: failed to open proxy ({exc})")
                continue

            if self._is_blocked_class(dev_class):
                continue

            try:
                commands = dev.command_list_query()
            except Exception as exc:
                if self.verbose:
                    print(f"Skipping {device_name}: failed to query commands ({exc})")
                continue

            for cmd in commands:
                command_name = cmd.cmd_name if hasattr(cmd, "cmd_name") else str(cmd)
                global_blocks = self.blocked_functions.get("*", [])
                if command_name in global_blocks or f"{dev_class}.{command_name}" in global_blocks or command_name in self.blocked_functions.get(dev_class, []):
                    continue
                try:
                    func = getattr(dev, command_name)
                except Exception as exc:
                    if self.verbose:
                        print(
                            f"Skipping {device_name}.{command_name}: "
                            f"failed to resolve command ({exc})"
                        )
                    continue
                if dev_class not in tools:
                    tools[dev_class] = {}
                tools[dev_class][command_name] = (func, cmd)
        return tools

    def setup(self, print_summary: bool = True):
        """Configure tools and add them to the MCP instance.

        Args:
            print_summary: If True, print tool discovery and registration summary.
        """
        raw_tools = self._find_tools()

        wrapped_tools: dict[str, dict[str, Callable]] = {}
        for dev_class in raw_tools:
            wrapped_tools[dev_class] = {}
            for command_name, (func, cmd_info) in raw_tools[dev_class].items():
                wrapped = self._create_wrapper(func, cmd_info, command_name, dev_class)
                wrapped_tools[dev_class][command_name] = wrapped

        self.tools = wrapped_tools
        if print_summary and self.verbose:
            print("Discovered tools by Tango class:")
            for dev_class in sorted(raw_tools):
                command_names = sorted(raw_tools[dev_class].keys())
                print(f"- {dev_class}: {len(command_names)}")
                for command_name in command_names:
                    print(f"    - {command_name}")

        native_tools = [self.get_data_from_key, self.list_devices]
        for native_tool in native_tools:
            self.mcp.add_tool(native_tool)
            if self.verbose:
                print(f"Registered native tool: {native_tool.__name__}")

        num_device_tools = 0
        for dev_class in wrapped_tools:
            for command_name, wrapped_func in wrapped_tools[dev_class].items():
                try:
                    tool_obj = Tool.from_function(wrapped_func)
                    self.mcp.add_tool(tool_obj)
                    num_device_tools += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to wrap {dev_class}.{command_name}: {e}")
                        traceback.print_exc()

        if print_summary and self.verbose:
            print(f"\nRegistered {len(native_tools)} native tool(s)")
            print(f"Registered {num_device_tools} Tango device command tool(s)")
            print(f"Total: {len(native_tools) + num_device_tools} tools")
            print("\nAll MCP tools available:")
            for dev_class in sorted(self.tools.keys()):
                command_names = sorted(self.tools[dev_class].keys())
                for command_name in command_names:
                    wrapped_func = self.tools[dev_class][command_name]
                    sig = inspect.signature(wrapped_func)
                    print(f"  - {dev_class}.{command_name}{sig}")
                    if wrapped_func.__doc__:
                        for line in wrapped_func.__doc__.split("\n"):
                            stripped = line.strip()
                            if stripped:
                                print(f"{stripped}")
                    print("")
                print("")

    def start(self, transport: Transport | None = None, **kwargs):
        """
        Synchronizes with Tango DB and begins serving the MCP protocol.

        Args:
            transport: Transport protocol to use ("stdio", "http", "sse", or "streamable-http").
                       Defaults to None, which uses stdio for local piping to agents.
            **kwargs: Additional keyword arguments to pass to the MCP server
        """
        self.setup()
        self.mcp.run(transport=transport, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tango-host", required=True)
    parser.add_argument("--tango-port", type=int, required=True)
    parser.add_argument("--transport", required=True)
    parser.add_argument("--http-host", required=True)
    parser.add_argument("--http-port", type=int, required=True)
    parser.add_argument("--blocked-classes-json", required=True)
    parser.add_argument("--blocked-functions-json", required=True)
    parser.add_argument("--data-device-address", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    server = MCPServer(
        name=args.name,
        tango_host=args.tango_host,
        tango_port=args.tango_port,
        blocked_classes=json.loads(args.blocked_classes_json),
        blocked_functions=json.loads(args.blocked_functions_json),
        data_device_address=args.data_device_address,
        verbose=not args.quiet,
    )
    if args.transport == "streamable-http":
        print(
            f"Starting {args.name} at http://{args.http_host}:{args.http_port}/mcp "
            f"for Tango DB {args.tango_host}:{args.tango_port}",
            flush=True,
        )
        server.start(transport="streamable-http", host=args.http_host, port=args.http_port)
    else:
        server.start(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
