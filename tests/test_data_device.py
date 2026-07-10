import json
import subprocess
from pathlib import Path


import pytest
import tango

from asyncroscopy.data.data import DATA, _catalog_database_uri


class TestDataDevice:
    def test_state_is_on(self, data_proxy: tango.DeviceProxy) -> None:
        assert data_proxy.state() == tango.DevState.ON

    def test_config_round_trip(self, data_proxy: tango.DeviceProxy, tmp_path) -> None:
        config = {
            "host": "127.0.0.1",
            "port": 9091,
            "save_path": str(tmp_path),
        }

        returned = json.loads(data_proxy.configure(json.dumps(config)))

        assert returned["host"] == config["host"]
        assert returned["port"] == config["port"]
        assert returned["save_path"] == config["save_path"]
        assert returned["uri"] == "http://127.0.0.1:9091"

    def test_save_path_creates_missing_directory(self, data_proxy: tango.DeviceProxy, tmp_path) -> None:
        save_path = tmp_path / "new" / "acquisitions"

        data_proxy.save_path = str(save_path)

        assert save_path.is_dir()

    def test_start_tiled_server_uses_catalog_server_command(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        calls = []
        popen_calls = []
        run_commands = []

        def fake_alive(self):
            calls.append(None)
            return len(calls) > 1

        class FakeProcess:
            def __init__(self):
                self.running = True

            def poll(self):
                return None if self.running else 0

            def terminate(self):
                self.running = False

            def wait(self, timeout):
                return 0

            def kill(self):
                self.running = False

        def fake_popen(command, **kwargs):
            popen_calls.append({"command": command, "kwargs": kwargs})
            return FakeProcess()

        data_proxy.host = "127.0.0.1"
        data_proxy.port = 9091
        data_proxy.save_path = str(tmp_path)
        monkeypatch.setattr(DATA, "_tiled_alive", fake_alive)
        monkeypatch.setattr(DATA, "_tiled_command", lambda self: ["python", "-m", "tiled"])
        monkeypatch.setattr("asyncroscopy.data.data.subprocess.Popen", fake_popen)
        monkeypatch.setattr(
            "asyncroscopy.data.data.subprocess.run",
            lambda command, **_: (
                run_commands.append(command)
                or type("Result", (), {"returncode": 0, "stdout": ""})()
            ),
        )

        returned = json.loads(data_proxy.start_tiled_server())

        assert returned["tiled_server"] == "yes"
        command_prefix = ["python", "-m"]
        key_value = popen_calls[0]["command"][10]
        expected_command = [
            *command_prefix,
            "tiled",
            "serve",
            "catalog",
            str(tmp_path / ".asyncroscopy_tiled_catalog.db"),
            "--read",
            str(tmp_path),
            "--public",
            "--api-key",
            key_value,
            "--host",
            "127.0.0.1",
            "--port",
            "9091",
        ]

        assert len(popen_calls) == 1
        actual_command = popen_calls[0]["command"]

        expected_catalog = _catalog_database_uri(tmp_path / ".asyncroscopy_tiled_catalog.db")

        assert actual_command[:5] == expected_command[:5]
        assert actual_command[5] == expected_catalog
        assert actual_command[6] == expected_command[6]
        assert Path(actual_command[7]) == Path(expected_command[7])
        assert actual_command[8:] == expected_command[8:]

        assert popen_calls[0]["kwargs"] == {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.STDOUT,
            "text": True,
        }

        assert len(run_commands) == 1
        assert run_commands[0][:6] == [
            *command_prefix,
            "tiled",
            "catalog",
            "init",
            "--if-not-exists",
        ]
        assert run_commands[0][6] == expected_catalog
        data_proxy.stop_tiled_server()

    def test_catalog_database_uri_uses_sqlite_uri_for_windows_drive_path(self) -> None:
        assert (
            _catalog_database_uri("C:/tiled_catalog_test/.asyncroscopy_tiled_catalog.db")
            == "sqlite:///C:/tiled_catalog_test/.asyncroscopy_tiled_catalog.db"
        )
        assert (
            _catalog_database_uri("C:\\tiled_catalog_test\\.asyncroscopy_tiled_catalog.db")
            == "sqlite:///C:/tiled_catalog_test/.asyncroscopy_tiled_catalog.db"
        )

    def test_start_tiled_server_uses_sqlite_uri_for_windows_catalog(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
    ) -> None:
        calls = []
        popen_calls = []
        run_commands = []

        def fake_alive(self):
            calls.append(None)
            return len(calls) > 1

        class FakeProcess:
            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(DATA, "_tiled_alive", fake_alive)
        monkeypatch.setattr(DATA, "_tiled_command", lambda self: ["python", "-m", "tiled"])
        monkeypatch.setattr("asyncroscopy.data.data.subprocess.Popen", lambda command, **kwargs: popen_calls.append(command) or FakeProcess())
        monkeypatch.setattr(
            "asyncroscopy.data.data.subprocess.run",
            lambda command, **_: (
                run_commands.append(command)
                or type("Result", (), {"returncode": 0, "stdout": ""})()
            ),
        )
        # The Windows drive path below is fed only to exercise the
        # drive-path -> sqlite-URI branch; neutralize the mkdir side effect so
        # the test does not depend on the path being creatable (a non-admin user
        # cannot create directories under C:\, and the path can't exist on Linux).
        monkeypatch.setattr("asyncroscopy.data.data._ensure_directory", lambda path: None)

        data_proxy.host = "127.0.0.1"
        data_proxy.port = 9091
        data_proxy.save_path = "C:/tiled_catalog_test"
        calls.clear()

        returned = json.loads(data_proxy.start_tiled_server())

        expected_catalog = "sqlite:///C:/tiled_catalog_test/.asyncroscopy_tiled_catalog.db"
        assert returned["tiled_server"] == "yes"
        assert run_commands[0][6] == expected_catalog
        assert popen_calls[0][5] == expected_catalog
        data_proxy.stop_tiled_server()

    def test_register_path_registers_single_file(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        registrations = []
        saved = tmp_path / "frame.h5"
        saved.write_bytes(b"fake-h5")
        data_proxy.host = "127.0.0.1"
        data_proxy.port = 9091

        def fake_from_uri(*args, **kwargs):
            return object()

        async def fake_register(client, path, **kwargs):
            registrations.append(path)

        monkeypatch.setattr("asyncroscopy.data.data.from_uri", fake_from_uri)
        monkeypatch.setattr("asyncroscopy.data.data.register", fake_register)

        result = data_proxy.register_path(str(saved))

        assert result == "frame.h5"
        assert registrations == [str(saved)]

    def test_register_save_path_registers_configured_directory(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        registrations = []
        data_proxy.host = "127.0.0.1"
        data_proxy.port = 9091
        data_proxy.save_path = str(tmp_path)

        def fake_from_uri(*args, **kwargs):
            return object()

        async def fake_register(client, path, **kwargs):
            registrations.append(path)

        monkeypatch.setattr("asyncroscopy.data.data.from_uri", fake_from_uri)
        monkeypatch.setattr("asyncroscopy.data.data.register", fake_register)

        result = json.loads(data_proxy.register_save_path())

        assert result["registered_path"] == str(tmp_path)
        assert result["tiled_server_status"] == "running; registered save path"
        assert registrations == [str(tmp_path)]

    def test_register_path_returns_windows_tiled_key(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
    ) -> None:
        windows_path = "D:/microscopedata/tiled/ahoust17/frame.h5"
        data_proxy.host = "127.0.0.1"
        data_proxy.port = 9091

        def fake_from_uri(*args, **kwargs):
            return object()

        async def fake_register(*args, **kwargs):
            return None

        monkeypatch.setattr("asyncroscopy.data.data.from_uri", fake_from_uri)
        monkeypatch.setattr("asyncroscopy.data.data.register", fake_register)

        assert data_proxy.register_path(windows_path) == "frame.h5"

    def test_register_path_waits_until_tiled_key_is_readable(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        saved = tmp_path / "frame.h5"
        saved.write_bytes(b"fake-h5")
        data_proxy.host = "127.0.0.1"
        data_proxy.port = 9091

        class FakeClient:
            def __init__(self):
                self.calls = 0

            def __getitem__(self, key):
                self.calls += 1
                if self.calls < 3:
                    raise KeyError(key)
                return object()

        fake_client = FakeClient()
        sleeps = []

        def fake_from_uri(*args, **kwargs):
            return fake_client

        async def fake_register(*args, **kwargs):
            return None

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr("asyncroscopy.data.data.from_uri", fake_from_uri)
        monkeypatch.setattr("asyncroscopy.data.data.register", fake_register)
        monkeypatch.setattr("asyncroscopy.data.data.asyncio.sleep", fake_sleep)

        assert data_proxy.register_path(str(saved)) == "frame.h5"
        assert fake_client.calls == 3
        assert sleeps == [0.25, 0.25]

    def test_save_path_change_restarts_managed_server(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        popen_calls = []
        processes = []

        class FakeProcess:
            def __init__(self):
                self.running = True
                self.terminated = False

            def poll(self):
                return None if self.running else 0

            def terminate(self):
                self.terminated = True
                self.running = False

            def wait(self, timeout):
                return 0

            def kill(self):
                self.running = False

        def fake_popen(command, **kwargs):
            process = FakeProcess()
            processes.append(process)
            popen_calls.append({"command": command, "kwargs": kwargs})
            return process

        def fake_alive(self):
            return self._tiled_process is not None and self._tiled_process.poll() is None

        run_commands = []
        monkeypatch.setattr(DATA, "_tiled_alive", fake_alive)
        monkeypatch.setattr(DATA, "_tiled_command", lambda self: ["python", "-m", "tiled"])
        monkeypatch.setattr("asyncroscopy.data.data.subprocess.Popen", fake_popen)
        monkeypatch.setattr(
            "asyncroscopy.data.data.subprocess.run",
            lambda command, **_: (
                run_commands.append(command)
                or type("Result", (), {"returncode": 0, "stdout": ""})()
            ),
        )

        first_path = tmp_path / "first"
        second_path = tmp_path / "second"

        data_proxy.save_path = str(first_path)
        data_proxy.start_tiled_server()
        data_proxy.save_path = str(second_path)
        config = json.loads(data_proxy.get_config())


        assert processes[0].terminated is True
        assert [Path(call["command"][7]) for call in popen_calls] == [Path(first_path), Path(second_path)]
        assert Path(config["tiled_server_serving"]) == Path(second_path)
        assert config["tiled_server_status"] == "running; serving path; files register manually"

        data_proxy.stop_tiled_server()

    def test_register_path_error_reports_save_and_serving_paths(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        save_path = tmp_path / "current"
        requested_path = save_path / "missing.h5"
        data_proxy.save_path = str(save_path)

        async def fake_register(*args, **kwargs):
            raise FileNotFoundError(requested_path)

        monkeypatch.setattr("asyncroscopy.data.data.from_uri", lambda *args, **kwargs: object())
        monkeypatch.setattr("asyncroscopy.data.data.register", fake_register)

        with pytest.raises(tango.DevFailed) as exc_info:
            data_proxy.register_path(str(requested_path))

        message = str(exc_info.value)
        status = json.loads(data_proxy.get_config())["tiled_server_status"]
        assert "File registration failed:" in message
        assert f"Requested file:\n    {requested_path}" in status
        assert f"Data save path:\n    {save_path}" in status
        assert "Tiled server serving:\n    " in status
