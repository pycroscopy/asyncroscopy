from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pytest
import tango


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[str]

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def output(self) -> str:
        if self.process.stdout is None:
            return ""
        try:
            return self.process.stdout.read()
        except Exception:
            return ""


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_output(process: subprocess.Popen[str], text: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    seen: list[str] = []
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited early: {process.returncode}\n" + "\n".join(seen))
        line = process.stdout.readline() if process.stdout else ""
        if line:
            seen.append(line.rstrip())
            if text in line:
                return
        else:
            time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {text!r}\n" + "\n".join(seen))


def device_url(host: str, port: int, device_name: str) -> str:
    return f"tango://{host}:{port}/{device_name}"


def wait_for_device(host: str, port: int, device_name: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    url = device_url(host, port, device_name)
    while time.monotonic() < deadline:
        try:
            proxy = tango.DeviceProxy(url)
            proxy.ping()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.2)
    raise TimeoutError(f"{device_name} did not become ready: {last_error}")


def add_device(db: tango.Database, server: str, classname: str, device: str) -> None:
    info = tango.DbDevInfo()
    info.server = server
    info._class = classname
    info.name = device
    db.add_device(info)


@pytest.fixture
def tango_database() -> Generator[tuple[str, int, dict[str, str]], None, None]:
    host = "127.0.0.1"
    port = find_free_port(host)
    tango_host = f"{host}:{port}"
    env = {**os.environ, "TANGO_HOST": tango_host, "PYTHONUNBUFFERED": "1"}
    managed: list[ManagedProcess] = []

    old_tango_host = os.environ.get("TANGO_HOST")
    os.environ["TANGO_HOST"] = tango_host

    try:
        with tempfile.TemporaryDirectory(
            prefix="asyncroscopy-tango-db-", ignore_cleanup_errors=True
        ) as db_dir:
            proc = subprocess.Popen(
                [sys.executable, "-m", "tango.databaseds.database", "2"],
                cwd=db_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            managed.append(ManagedProcess("database", proc))
            try:
                wait_for_output(proc, "Ready to accept request", timeout=30)
            except Exception as exc:
                pytest.skip(f"Tango database server could not be started: {exc}")

            try:
                yield host, port, env
            finally:
                # Stop the DB process before the temp dir is removed, otherwise
                # Windows can't delete the still-open tango_database.db (WinError 32).
                for proc in reversed(managed):
                    proc.stop()
    finally:
        if old_tango_host is None:
            os.environ.pop("TANGO_HOST", None)
        else:
            os.environ["TANGO_HOST"] = old_tango_host


def start_device_server(module: str, instance: str, env: dict[str, str]) -> ManagedProcess:
    proc = subprocess.Popen(
        [sys.executable, "-m", module, instance],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return ManagedProcess(module, proc)


def test_can_start_scan_data_and_microscope_servers(tango_database, tmp_path) -> None:
    host, port, env = tango_database
    db = tango.Database(host, port)

    scan_device = "asyncroscopy/scan/default"
    data_device = "asyncroscopy/data/default"
    microscope_device = "asyncroscopy/instrument/default"

    add_device(db, "SCAN/scan_instance", "SCAN", scan_device)
    add_device(db, "DATA/data_instance", "DATA", data_device)
    add_device(db, "AutoScriptMicroscope/microscope_instance", "AutoScriptMicroscope", microscope_device)
    db.put_device_property(
        microscope_device,
        {
            "testing_mode_bool": [True],
            "scan_device_address": [scan_device],
            "camera_device_address": [""],
            "flucam_device_address": [""],
            "eds_device_address": [""],
            "stage_device_address": [""],
            "data_device_address": [data_device],
        },
    )

    managed = [
        start_device_server("asyncroscopy.instruments.electron_microscope.hardware.scan", "scan_instance", env),
        start_device_server("asyncroscopy.data.data", "data_instance", env),
    ]

    try:
        for device in [scan_device, data_device]:
            wait_for_device(host, port, device, timeout=20)

        managed.append(start_device_server("asyncroscopy.instruments.electron_microscope.auto_script", "microscope_instance", env))
        wait_for_device(host, port, microscope_device, timeout=20)

        scan = tango.DeviceProxy(device_url(host, port, scan_device))
        data = tango.DeviceProxy(device_url(host, port, data_device))
        microscope = tango.DeviceProxy(device_url(host, port, microscope_device))

        data.save_path = str(tmp_path)

        assert scan.state() == tango.DevState.ON
        assert data.state() == tango.DevState.ON
        assert microscope.state() == tango.DevState.ON
        assert tango.Database(host, port).get_device_info(microscope_device).name == microscope_device
    finally:
        for proc in reversed(managed):
            proc.stop()
