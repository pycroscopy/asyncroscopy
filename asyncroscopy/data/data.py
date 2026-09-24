"""DATA Tango device.

This device is the Tango bridge to the Tiled HTTP data server. It stores the
server URI and acquisition save path used by notebooks and microscope devices.

Acquisitions are registered with Tiled explicitly through ``register_acquisition_file``.
The DATA device intentionally does not start a Tiled filesystem watcher:
in-situ experiments register each image as it is written and avoid the
overhead of monitoring the full acquisition directory.

Starting or restarting a managed Tiled server can take longer than Tango's
default client timeout. Callers that invoke ``start_tiled_server`` directly, or
change ``save_path`` while a managed server is active, should set an extended
timeout on their DATA ``DeviceProxy``.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath
from urllib.error import URLError
from urllib.request import urlopen

from tango import AttrWriteType, DevState
from tango.server import Device, attribute, command
from tiled.client import from_uri
from tiled.client.register import identity, register

DEFAULT_TILED_URI = "http://10.46.217.241:9091"
DEFAULT_ACQUISITION_DIR = "outputs/tiled_acquisitions"
ONE_NODE_PER_FILE_WALKER = "tiled.client.register:one_node_per_item"
REGISTER_TIMEOUT_SECONDS = 120
REGISTER_SAVE_PATH_TIMEOUT_SECONDS = 3600
REGISTER_POLL_SECONDS = 0.25


class DATA(Device):
    """Tango bridge to the Tiled HTTP data server."""

    # Attributes:
    host = attribute(label="Tiled Host", dtype=str, access=AttrWriteType.READ_WRITE, doc="Hostname or IP address for the Tiled HTTP data server.")
    port = attribute(label="Tiled Port", dtype=int, access=AttrWriteType.READ_WRITE, doc="TCP port for the Tiled HTTP data server.")
    save_path = attribute(label="Acquisition Save Path", dtype=str, access=AttrWriteType.READ_WRITE, doc="Directory where acquisition files are written and served by Tiled.")
    tiled_server = attribute(label="Tiled Server", dtype=str, access=AttrWriteType.READ, doc="yes if the configured Tiled HTTP data server responds, otherwise no.")

    # Init:
    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.ON)
        uri = os.environ.get("ASYNCROSCOPY_TILED_URI", DEFAULT_TILED_URI)
        host, _, port = uri.split("://", 1)[-1].strip("/").partition(":")
        self._host, self._port = host or "10.46.217.241", int(port or 9091)
        self._save_path = os.environ.get("ASYNCROSCOPY_ACQUISITION_DIR", DEFAULT_ACQUISITION_DIR)
        self._api_key = os.environ.get("ASYNCROSCOPY_TILED_API_KEY", "secret")
        self._tiled_process = None
        self._tiled_serve_path = None
        self._tiled_server = "yes" if self._tiled_server_is_reachable() else "no"
        self._tiled_server_status = ""
        self.info_stream("DATA device initialised")

    def delete_device(self) -> None:
        self._stop_managed_tiled_server()
        super().delete_device()

    # Attribute Read/Write Methods:
    def read_host(self) -> str:
        return self._host

    def write_host(self, value: str) -> None:
        value = value.strip()
        if value == self._host:
            return
        self._host = value
        self._restart_managed_tiled_server()

    def read_port(self) -> int:
        return self._port

    def write_port(self, value: int) -> None:
        value = int(value)
        if value == self._port:
            return
        self._port = value
        self._restart_managed_tiled_server()

    def read_save_path(self) -> str:
        return self._save_path

    def write_save_path(self, value: str) -> None:
        value = value.strip()
        if value == self._save_path:
            return

        if not (_is_windows_drive_path(value) and os.name != "nt"):
            Path(value).expanduser().mkdir(parents=True, exist_ok=True)
        self._save_path = value
        self._restart_managed_tiled_server()

    def read_tiled_server(self) -> str:
        self._tiled_server = "yes" if self._tiled_server_is_reachable() else "no"
        return self._tiled_server

    # Commands:
    @command(dtype_out=str)
    def get_config(self) -> str:
        config = {
            "host": self._host,
            "port": self._port,
            "uri": self._tiled_uri(),
            "save_path": self._save_path,
            "tiled_server": self._tiled_server,
            "tiled_server_status": self._tiled_server_status,
            "tiled_server_serving": self._tiled_serve_path,
        }
        return json.dumps(config)

    @command(dtype_in=str, dtype_out=str)
    def configure(self, config_json: str) -> str:
        config = json.loads(config_json) if config_json else {}
        for key, writer in {
            "host": self.write_host,
            "port": self.write_port,
            "save_path": self.write_save_path,
        }.items():
            if key in config:
                writer(config[key])
        return self.get_config()

    @command(dtype_out=str)
    def start_tiled_server(self, timeout=30) -> str:
        """Start the catalog HTTP server without a filesystem watcher."""
        if self._tiled_server_is_reachable():
            self._tiled_server = "yes"
            self._tiled_server_status = "running; files register manually"
            return self.get_config()

        save_path = PureWindowsPath(self._save_path) if _is_windows_drive_path(self._save_path) else Path(self._save_path).expanduser()
        catalog = save_path / ".asyncroscopy_tiled_catalog.db"
        catalog_database = f"sqlite:///{catalog.as_posix()}" if _is_windows_drive_path(catalog) else str(catalog)

        try:
            if not (_is_windows_drive_path(self._save_path) and os.name != "nt"):
                Path(self._save_path).expanduser().mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "tiled", "catalog", "init", "--if-not-exists", catalog_database]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as exc:
            self._tiled_server = "no"
            output = (exc.stdout or "").strip()
            self._tiled_server_status = f"{exc}; output: {output}" if output else str(exc)
            return self.get_config()
        except Exception as exc:
            self._tiled_server = "no"
            self._tiled_server_status = str(exc)
            return self.get_config()

        command = [
            sys.executable, "-m", "tiled", "serve", "config", str(Path(__file__).with_name("config.yml")),
            "--public", "--api-key", self._api_key,
            "--host", self._host, "--port", str(self._port),
        ]
        environment = {**os.environ, "ASYNCROSCOPY_TILED_CATALOG": catalog_database, "ASYNCROSCOPY_ACQUISITION_DIR": self._save_path}
        self._tiled_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True, env=environment)
        self._tiled_serve_path = self._save_path
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._tiled_server_is_reachable():
            if self._tiled_process.poll() is not None:
                break
            time.sleep(0.5)
        self._tiled_server = "yes" if self._tiled_server_is_reachable() else "no"
        if self._tiled_server == "yes":
            self._tiled_server_status = "running; serving path; files register manually"
        else:
            self._tiled_server_status = f"not running; exit_code={self._tiled_process.poll()}"
        return self.get_config()

    @command(dtype_out=str)
    def stop_tiled_server(self) -> str:
        self._stop_managed_tiled_server()
        self._tiled_server = "yes" if self._tiled_server_is_reachable() else "no"
        self._tiled_server_status = "stopped managed Tiled processes"
        return self.get_config()

    @command(dtype_in=str, dtype_out=str)
    def register_acquisition_file(self, path: str) -> str:
        """Register one completed file, wait for its Tiled key, and return that key."""
        path = path.strip()
        key = PureWindowsPath(path).name if _is_windows_drive_path(path) else Path(path).name

        async def register_file_and_wait_for_key() -> None:
            client = from_uri(self._tiled_uri(), api_key=self._api_key)
            # Acquisition files are already closed; expose this file before returning.
            await register(client, path, adapters_by_mimetype={"application/x-hdf5": "asyncroscopy.data.data_reader:NSIDAdapter"}, walkers=[ONE_NODE_PER_FILE_WALKER], key_from_filename=identity)
            if not hasattr(client, "__getitem__"):
                return
            deadline = time.monotonic() + REGISTER_TIMEOUT_SECONDS
            while True:
                try:
                    client[key]
                    return
                except KeyError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Tiled did not expose {key} within {REGISTER_TIMEOUT_SECONDS} seconds")
                    await asyncio.sleep(REGISTER_POLL_SECONDS)

        try:
            asyncio.run(asyncio.wait_for(register_file_and_wait_for_key(), REGISTER_TIMEOUT_SECONDS))
        except Exception as exc:
            message = (
                f"File registration failed: {exc}\n\n"
                f"Requested file:\n    {path}\n\n"
                f"Data save path:\n    {self._save_path}\n\n"
                f"Tiled server serving:\n    {self._tiled_serve_path or '(external server; path not managed by DATA)'}"
            )
            self._tiled_server_status = message
            raise RuntimeError(message) from exc
        self._tiled_server_status = "running; registered path"
        return key

    @command(dtype_out=str)
    def register_existing_directory(self) -> str:
        """Index existing files in the save directory and return a JSON status report."""
        # Optional startup indexing, not part of saving a new acquisition.
        save_path = str(Path(self._save_path).expanduser())

        try:
            client = from_uri(self._tiled_uri(), api_key=self._api_key)
            asyncio.run(asyncio.wait_for(
                register(client, save_path, adapters_by_mimetype={"application/x-hdf5": "asyncroscopy.data.data_reader:NSIDAdapter"}, walkers=[ONE_NODE_PER_FILE_WALKER], key_from_filename=identity),
                REGISTER_SAVE_PATH_TIMEOUT_SECONDS,
            ))
        except Exception as exc:
            message = (
                f"Save path registration failed: {exc}\n\n"
                f"Data save path:\n    {save_path}\n\n"
                f"Tiled server serving:\n    {self._tiled_serve_path or '(external server; path not managed by DATA)'}"
            )
            self._tiled_server_status = message
            raise RuntimeError(message) from exc

        result = {
            "registered_path": save_path,
            "tiled_server": self._tiled_server,
            "tiled_server_status": "running; registered save path",
            "tiled_server_serving": self._tiled_serve_path,
        }
        self._tiled_server_status = result["tiled_server_status"]
        return json.dumps(result)

    def _tiled_uri(self) -> str:
        return f"http://{self._host}:{self._port}"

    def _tiled_server_is_reachable(self) -> bool:
        try:
            with urlopen(self._tiled_uri(), timeout=0.3):
                return True
        except (OSError, URLError):
            return False

    def _restart_managed_tiled_server(self) -> None:
        if self._tiled_process is not None and self._tiled_process.poll() is None:
            self.info_stream(f"Restarting managed Tiled server for save path: {self._save_path}")
            self._stop_managed_tiled_server()
            self.start_tiled_server()
            return

        self._tiled_process = None
        self._tiled_serve_path = None
        if self._tiled_server_is_reachable():
            self._tiled_server_status = "running externally; files register manually"
        else:
            self._tiled_server_status = "not running"

    def _stop_managed_tiled_server(self) -> None:
        process = self._tiled_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._tiled_process = None
        self._tiled_serve_path = None


def _is_windows_drive_path(path: str | Path | PureWindowsPath) -> bool:
    windows_path = PureWindowsPath(path)
    return bool(windows_path.drive)


if __name__ == "__main__":
    DATA.run_server()
