"""End-to-end DB-mode tests for MCP + DigitalTwin integration."""

from __future__ import annotations

import os
import json
import socket
import subprocess
import sys
import tempfile
import time
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pytest
import tango

import asyncio

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asyncroscopy.mcp.mcp_server import MCPServer


def mcp_kwargs(**overrides):
    config = {
        "blocked_classes": ["DataBase", "DServer"],
        "blocked_functions": {"*": ["Init"]},
        "data_device_address": "asyncroscopy/data/default",
    }
    config.update(overrides)
    return config


@dataclass
class ManagedProcess:
    """A subprocess wrapper with a name for logging."""

    name: str
    process: subprocess.Popen[str]


@pytest.mark.skip(
    reason=(
        "DB-mode Tango discovery cannot safely run in the same interpreter as "
        "the in-process MultiDeviceTestContext. Server startup is covered by "
        "tests/test_server_startup.py."
    )
)
class TestMCPServerDBMode:
    """Test suite for MCP server in DB mode with DigitalTwin."""

    @staticmethod
    def wait_for_device_ready(device_name: str, timeout: float = 10.0) -> None:
        """Wait until a Tango device can be opened and pinged successfully."""
        start = time.monotonic()
        last_error: Exception | None = None

        while time.monotonic() - start < timeout:
            try:
                dev = tango.DeviceProxy(device_name)
                dev.ping()
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.1)

        raise TimeoutError(f"Timed out waiting for device '{device_name}' readiness. Last error: {last_error}")

    @staticmethod
    def find_free_port(host: str = "127.0.0.1") -> int:
        """Find an available port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def make_env(tango_host: str) -> dict[str, str]:
        """Create environment with TANGO_HOST and unbuffered output."""
        env = os.environ.copy()
        env["TANGO_HOST"] = tango_host
        env["PYTHONUNBUFFERED"] = "1"
        return env

    @staticmethod
    def wait_for_process_output(
        proc: subprocess.Popen[str],
        expected_text: str,
        timeout: float,
        process_name: str,
    ) -> None:
        """Wait for a process to output a specific string."""
        start = time.monotonic()
        seen_lines: list[str] = []

        while time.monotonic() - start < timeout:
            if proc.poll() is not None:
                output = "\n".join(seen_lines)
                raise RuntimeError(
                    f"{process_name} exited early with code {proc.returncode}.\n"
                    f"Observed output:\n{output}"
                )

            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                line = line.rstrip("\n")
                seen_lines.append(line)
                print(f"[{process_name}] {line}")
                if expected_text in line:
                    return
            else:
                time.sleep(0.05)

        output = "\n".join(seen_lines)
        raise TimeoutError(
            f"Timed out waiting for '{expected_text}' from {process_name}.\n"
            f"Observed output:\n{output}"
        )

    @staticmethod
    def stop_process(managed: ManagedProcess, timeout: float = 5.0) -> None:
        """Terminate a managed process."""
        proc = managed.process
        if proc.poll() is not None:
            return

        print(f"[shutdown] terminating {managed.name} (pid={proc.pid})")
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"[shutdown] killing {managed.name} (pid={proc.pid})")
            proc.kill()
            proc.wait(timeout=timeout)

    def start_tango_db(
        self,
        python_bin: str,
        tango_host: str,
        work_dir: Path,
        timeout: float,
    ) -> ManagedProcess:
        """Start the Tango database server."""
        env = self.make_env(tango_host)
        proc = subprocess.Popen(
            [python_bin, "-m", "tango.databaseds.database", "2"],
            cwd=work_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        managed = ManagedProcess(name="tango-db", process=proc)
        self.wait_for_process_output(proc, "Ready to accept request", timeout, managed.name)
        return managed

    @staticmethod
    def register_digital_twin(
        db_host: str, db_port: int, instance: str, device_name: str
    ) -> None:
        """Register DigitalTwin device in Tango DB."""
        db = tango.Database(db_host, db_port)
        info = tango.DbDevInfo()
        info.server = f"DigitalTwin/{instance}"
        info._class = "DigitalTwin"
        info.name = device_name

        try:
            db.add_device(info)
            print(f"[register] registered: {device_name}")
        except tango.DevFailed:
            print(f"[register] device already present: {device_name}")

    def start_digital_twin(
        self,
        python_bin: str,
        tango_host: str,
        instance: str,
        timeout: float,
    ) -> ManagedProcess:
        """Start the DigitalTwin device server."""
        env = self.make_env(tango_host)
        proc = subprocess.Popen(
            [python_bin, "-m", "asyncroscopy.instruments.electron_microscope.digital_twin", instance],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        managed = ManagedProcess(name="digital-twin", process=proc)
        self.wait_for_process_output(
            proc, "Ready to accept request", timeout, managed.name
        )
        return managed

    @pytest.fixture(scope="class")
    def test_infrastructure(self) -> Generator[tuple[str, int], None, None]:
        """Pytest fixture that sets up the complete test infrastructure (DB + Digital Twin)."""
        host = "127.0.0.1"
        port = self.find_free_port(host)
        tango_host = f"{host}:{port}"
        python_bin = __import__("sys").executable

        print(f"\n[config] TANGO_HOST={tango_host}")
        os.environ["TANGO_HOST"] = tango_host

        managed_procs: list[ManagedProcess] = []

        try:
            # Start Tango DB
            with tempfile.TemporaryDirectory(prefix="tango-db-") as db_dir:
                db_path = Path(db_dir)
                print(f"[config] tango-db working dir={db_path}")

                try:
                    db_proc = self.start_tango_db(
                        python_bin=python_bin,
                        tango_host=tango_host,
                        work_dir=db_path,
                        timeout=30.0,
                    )
                except Exception as exc:
                    pytest.skip(f"Tango DB could not be started in this environment: {exc}")
                managed_procs.append(db_proc)

                # Register and start Digital Twin
                instance = "test_instance"
                device_name = "asyncroscopy/digitaltwin/default"
                self.register_digital_twin(host, port, instance, device_name)

                try:
                    twin_proc = self.start_digital_twin(
                        python_bin=python_bin,
                        tango_host=tango_host,
                        instance=instance,
                        timeout=30.0,
                    )
                except Exception as exc:
                    pytest.skip(f"DigitalTwin server could not be started in this environment: {exc}")
                managed_procs.append(twin_proc)

                try:
                    self.wait_for_device_ready(device_name, timeout=10.0)
                except Exception as exc:
                    pytest.skip(f"DigitalTwin not ready for queries: {exc}")

                yield host, port

        finally:
            # Cleanup in reverse order
            for proc in reversed(managed_procs):
                self.stop_process(proc)

    def test_mcp_tool_discovery(
        self,
        test_infrastructure: tuple[str, int],
    ) -> None:
        """Test that MCP discovers all Tango device tools."""
        host, port = test_infrastructure

        # Create MCPServer and discover tools
        server = MCPServer(name="MCPServerTest", tango_host=host, tango_port=port, **mcp_kwargs())

        server.setup(print_summary=True)
        tools = server.tools

        mcp_tools = asyncio.run(server.mcp.list_tools())
        mcp_tool_names = {tool.name for tool in mcp_tools}
        assert "list_devices" in mcp_tool_names, (
            "list_devices should be auto-registered as an MCP tool"
        )

        # Verify DigitalTwin was discovered
        assert "DigitalTwin" in tools, "DigitalTwin class not found in MCP tool discovery"

        # Verify expected tools exist
        digital_twin_tools = tools["DigitalTwin"]
        expected_tools = {
            "Connect",
            "Disconnect",
            "acquire_scanned_image",
        }

        for expected_tool in expected_tools:
            assert expected_tool in digital_twin_tools, (
                f"Expected tool {expected_tool} not found"
            )

        # Verify blocked classes are not exposed
        assert "DataBase" not in tools, "DataBase should be blocked"
        assert "DServer" not in tools, "DServer should be blocked"

    def test_list_devices_is_available(
        self,
        test_infrastructure: tuple[str, int],
    ) -> None:
        """Test that list_devices MCPServer method is available."""
        host, port = test_infrastructure

        server = MCPServer(
            name="MCPServerTest",
            tango_host=host,
            tango_port=port,
            **mcp_kwargs(),
        )

        server.setup(print_summary=False)

        # Verify list_devices method exists
        assert hasattr(server, "list_devices"), "list_devices method not found on MCPServer"

        # Call it to verify it works
        deadline = time.monotonic() + 3.0
        has_digital_twin = False
        devices: list[str] = []

        # Tango DB can transiently reject very early connection attempts.
        while time.monotonic() < deadline and not has_digital_twin:
            devices = server.list_devices()
            assert isinstance(devices, list), "list_devices should return a list"

            for device_name in devices:
                try:
                    dev = tango.DeviceProxy(device_name)
                    if dev.info().dev_class == "DigitalTwin":
                        has_digital_twin = True
                        break
                except Exception:
                    continue

            if not has_digital_twin:
                time.sleep(0.1)

        assert has_digital_twin, f"No DigitalTwin-class device found in list_devices output: {devices}"

    def test_blocked_classes_respected(
        self,
        test_infrastructure: tuple[str, int],
    ) -> None:
        """Test that blocked_classes parameter is respected."""
        host, port = test_infrastructure

        server = MCPServer(
            name="MCPServerTest",
            tango_host=host,
            tango_port=port,
            **mcp_kwargs(blocked_classes=["DataBase", "DServer", "DigitalTwin"]),
        )

        server.setup(print_summary=False)
        tools = server.tools

        # Verify blocked classes are not in tools
        for blocked_class in ["DataBase", "DServer", "DigitalTwin"]:
            assert blocked_class not in tools, f"Blocked class {blocked_class} was exposed"


class TestMCPSerialization:
    def test_devencoded_type_maps_to_object_schema(self) -> None:
        mapped = MCPServer._tango_type_to_python(tango.CmdArgType.DevEncoded)
        assert mapped is dict

    def test_devencoded_payload_is_json_safe(self) -> None:
        normalized = MCPServer._normalize_command_result(
            tango.CmdArgType.DevEncoded,
            ('{"shape":[2,2],"dtype":"uint8"}', b"\x00\x01\xff\x10"),
        )

        assert isinstance(normalized, dict)
        assert normalized["encoding"] == "base64"
        assert normalized["metadata"] == {"shape": [2, 2], "dtype": "uint8"}
        assert isinstance(normalized["payload"], str)
        assert base64.b64decode(normalized["payload"]) == b"\x00\x01\xff\x10"

    def test_numpy_to_python_converts_nested_structures(self) -> None:
        payload = {
            "array": np.array([1, 2], dtype=np.int64),
            "scalar": np.float32(1.5),
            "tuple": (np.int32(3), np.array([[4, 5]], dtype=np.int64)),
            "list": [np.uint8(6)],
        }

        converted = MCPServer._numpy_to_python(payload)

        assert converted == {
            "array": [1, 2],
            "scalar": 1.5,
            "tuple": (3, [[4, 5]]),
            "list": [6],
        }
        assert isinstance(converted["tuple"], tuple)

    def test_normalize_command_result_converts_numpy_non_encoded(self) -> None:
        result = np.array([1, 2, 3], dtype=np.uint16)
        normalized = MCPServer._normalize_command_result(tango.CmdArgType.DevString, result)
        assert normalized == [1, 2, 3]

    def test_normalize_command_result_converts_numpy_nested(self) -> None:
        result = {
            "array": np.array([1, 2], dtype=np.int64),
            "scalar": np.float32(1.25),
            "nested": {"values": [np.uint8(3), np.array([[4, 5]])]},
        }

        normalized = MCPServer._normalize_command_result(tango.CmdArgType.DevString, result)

        assert normalized == {
            "array": [1, 2],
            "scalar": 1.25,
            "nested": {"values": [3, [[4, 5]]]},
        }

    def test_get_data_from_key_reads_remote_tiled_preview(self, monkeypatch) -> None:
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        class FakeArray:
            metadata = {"detector": "HAADF"}
            shape = (3, 3)
            dtype = np.dtype("int64")

            def read(self, slices=None):
                array = np.arange(9).reshape(3, 3)
                return array[slices] if slices else array

        class FakeContainer(dict):
            metadata = {"source": "microscope"}

        tiled_node = FakeContainer(image=FakeContainer(HAADF=FakeArray()))

        class FakeDataProxy:
            def get_config(self):
                return json.dumps({"uri": "http://microscope:9091"})

        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", lambda address: FakeDataProxy())
        monkeypatch.setattr(
            "asyncroscopy.mcp.mcp_server.from_uri",
            lambda uri: {"frame.h5": tiled_node},
        )

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)
        result = server.get_data_from_key("frame.h5", max_values=4)

        assert result["key"] == "frame.h5"
        assert result["uri"] == "http://microscope:9091"
        assert result["format"] == "hdf5"
        assert result["attrs"] == {"source": "microscope"}
        assert result["datasets"] == [
            {
                "name": "image/HAADF",
                "shape": [3, 3],
                "dtype": "int64",
                "attrs": {"detector": "HAADF"},
                "preview": [0, 1, 2, 3],
            }
        ]


class TestMCPServerTypeMapping:
    def test_tango_types_map_to_python(self) -> None:
        assert MCPServer._tango_type_to_python(tango.CmdArgType.DevString) is str
        assert MCPServer._tango_type_to_python(tango.CmdArgType.DevVarDoubleArray) == list[float]
        assert MCPServer._tango_type_to_python(tango.CmdArgType.DevUChar) == np.uint8
        assert MCPServer._tango_type_to_python(tango.CmdArgType.DevEncoded) is dict


class TestMCPToolInvocation:
    def test_wrapper_supports_positional_and_keyword(self, monkeypatch) -> None:
        # Mock Database and DeviceProxy to avoid connection errors
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        # Mock objects for wrapping
        def mock_func(val):
            return val

        cmd_info = type(
            "CommandInfo",
            (),
            {
                "in_type": tango.CmdArgType.DevString,
                "out_type": tango.CmdArgType.DevString,
                "in_type_desc": ":param config_json: (not documented)\n:type config_json: DevString",
                "out_type_desc": "result",
            },
        )

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs())
        wrapper = server._create_wrapper(mock_func, cmd_info, "MyCmd", "MyClass")

        # 1. Positional call
        assert wrapper("hello") == "hello"

        # 2. Keyword call with the generic Tango argument name
        import inspect

        sig = inspect.signature(wrapper)
        param_name = list(sig.parameters.keys())[0]        
        assert param_name == "config_json" 
        assert wrapper(**{param_name: "world"}) == "world"

    def test_void_wrapper_supports_no_args(self, monkeypatch) -> None:
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        def mock_func():
            return "done"

        cmd_info = type(
            "CommandInfo",
            (),
            {
                "in_type": tango.CmdArgType.DevVoid,
                "out_type": tango.CmdArgType.DevString,
                "in_type_desc": "",
                "out_type_desc": "",
            },
        )

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs())
        wrapper = server._create_wrapper(mock_func, cmd_info, "VoidCmd", "MyClass")

        assert wrapper() == "done"


class TestMCPRegistration:
    def test_setup_registers_native_tools(self, monkeypatch) -> None:
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)
        calls = []

        def record_tool(method):
            calls.append(method.__name__)

        monkeypatch.setattr(server.mcp, "add_tool", record_tool)
        monkeypatch.setattr(server, "_find_tools", lambda: {})

        server.setup(print_summary=False)

        assert set(calls) == {"get_data_from_key", "list_devices"}
