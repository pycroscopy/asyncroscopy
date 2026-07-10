import os
import sys
import time

from startup_scripts import run_mcp, run_servers


class FakeDataProxy:
    def __init__(self):
        self.timeout_millis = None
        self.stop_called = False
        self.register_called = False

    def set_timeout_millis(self, timeout_millis):
        self.timeout_millis = timeout_millis

    def stop_tiled_server(self):
        self.stop_called = True

    def register_save_path(self):
        self.register_called = True
        return '{"registered_path": "outputs/tiled_acquisitions"}'


def test_get_data_proxy_sets_extended_timeout(monkeypatch):
    proxy = FakeDataProxy()
    monkeypatch.setattr(run_servers.tango, "DeviceProxy", lambda _: proxy)

    assert run_servers.get_data_proxy() is proxy
    assert proxy.timeout_millis == run_servers.TILED_COMMAND_TIMEOUT_MILLIS


def test_stop_tiled_server_uses_extended_data_proxy_timeout(monkeypatch):
    proxy = FakeDataProxy()
    monkeypatch.setattr(run_servers.tango, "DeviceProxy", lambda _: proxy)

    run_servers.stop_tiled_server()

    assert proxy.timeout_millis == run_servers.TILED_COMMAND_TIMEOUT_MILLIS
    assert proxy.stop_called is True


def test_register_tiled_save_path_uses_startup_registration_timeout(monkeypatch):
    proxy = FakeDataProxy()
    monkeypatch.setattr(run_servers.tango, "DeviceProxy", lambda _: proxy)

    result = run_servers.register_tiled_save_path()

    assert proxy.timeout_millis == run_servers.TILED_STARTUP_REGISTRATION_TIMEOUT_MILLIS
    assert proxy.register_called is True
    assert result == {"registered_path": "outputs/tiled_acquisitions"}


def test_start_process_tracks_process_group(monkeypatch):
    calls = {}

    class FakePopen:
        stdout = None
        stderr = None
        pid = 1234

        def __init__(self, command, **kwargs):
            calls["command"] = command
            calls["kwargs"] = kwargs

        def poll(self):
            return None

    monkeypatch.setattr(run_servers.subprocess, "Popen", FakePopen)

    process = run_servers.start_process("scan", "SCAN", ["uv", "run", "scan"], {"TANGO_HOST": "localhost:9094"})

    assert process.pid == 1234
    assert calls["command"] == ["uv", "run", "scan"]
    if run_servers.os.name == "nt":
        assert "creationflags" in calls["kwargs"]
    else:
        assert calls["kwargs"]["start_new_session"] is True


def test_start_process_drains_child_output():
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    command = [
        sys.executable,
        "-c",
        "import sys\nfor index in range(1000): print(f'line-{index}')\nprint('done', file=sys.stderr)",
    ]

    process = run_servers.start_process("writer", "Writer", command, environment)
    try:
        process.process.wait(timeout=5)
        deadline = time.monotonic() + 2
        while len(process.stdout_lines) < run_servers.PROCESS_OUTPUT_LINES and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        if process.running:
            run_servers.stop_process(process)

    assert len(process.stdout_lines) == run_servers.PROCESS_OUTPUT_LINES
    assert process.stdout_lines[-1] == "line-999"
    assert process.stderr_lines[-1] == "done"


def test_start_process_can_write_child_output_to_debug_log(tmp_path):
    environment = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    command = [
        sys.executable,
        '-c',
        "import sys\nprint('log-line', file=sys.stderr)",
    ]

    process = run_servers.start_process('writer', 'Writer', command, environment, tmp_path)
    assert process.log_path == tmp_path / 'writer.log'

    try:
        process.process.wait(timeout=5)
        deadline = time.monotonic() + 2
        while not process.log_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        while 'log-line' not in process.log_path.read_text(encoding='utf-8') and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        if process.running:
            run_servers.stop_process(process)

    log_text = process.log_path.read_text(encoding='utf-8')
    assert '[stderr] log-line' in log_text


def test_stop_process_terminates_process_group(monkeypatch):
    if run_servers.os.name == "nt":
        return

    signals = []

    class FakeProcess:
        pid = 4321

        def poll(self):
            return None

        def wait(self, timeout):
            return 0

        def terminate(self):
            raise AssertionError("process group should be signaled before direct terminate")

    monkeypatch.setattr(run_servers.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    process = run_servers.ManagedProcess("scan", "SCAN", ["uv", "run", "scan"], FakeProcess())
    run_servers.stop_process(process)

    assert signals == [(4321, run_servers.signal.SIGTERM)]


def test_delete_tango_database_files_removes_known_filenames(tmp_path, monkeypatch):
    monkeypatch.setattr(run_servers, "PROJECT_DIR", tmp_path)
    lowercase = tmp_path / "tango_database.db"
    uppercase = tmp_path / "Tango_database.db"
    lowercase.write_text("old database", encoding="utf-8")
    uppercase.write_text("old database", encoding="utf-8")

    deleted = run_servers.delete_tango_database_files()

    assert deleted
    assert {path.name for path in deleted} <= {"tango_database.db", "Tango_database.db"}
    assert not lowercase.exists()
    assert not uppercase.exists()


def test_load_spectra300_config_starts_servers_only():
    config = run_servers.load_config(run_servers.PROJECT_DIR / "configs" / "Spectra300.yaml")

    assert config.tango_host == "10.46.217.241"
    assert config.tiled.host == "10.46.217.241"
    assert config.tiled.register_on_startup is False
    assert config.instrument.class_name == "AutoScriptMicroscope"
    assert config.instrument.module_name == "asyncroscopy.instruments.electron_microscope.auto_script"
    assert config.reset_database_file is False
    assert not hasattr(config, "mcp")


def test_build_devices_adds_selected_instrument():
    config = run_servers.load_config(run_servers.PROJECT_DIR / "configs" / "DigitalTwin.yaml")

    devices = run_servers.build_devices(config)

    assert devices[-1].key == "instrument"
    assert devices[-1].class_name == "DigitalTwin"
    assert devices[-1].module_name == "asyncroscopy.instruments.electron_microscope.digital_twin"
    assert devices[-1].device_name == "asyncroscopy/instrument/default"


def test_load_mcp_config():
    config = run_mcp.load_config(run_mcp.PROJECT_DIR / "configs" / "mcp.yaml")

    assert config.mcp.name == "Spectra300_MCP"
    assert config.tango_host == "localhost"
    assert config.tango_port == 9094
    assert config.mcp.http_host == "127.0.0.1"
    assert config.mcp.http_port == 8000
    assert config.mcp.blocked_classes == ["DataBase", "DServer"]
    assert config.mcp.blocked_functions == {"*": ["Init", "Kill", "RestartServer"]}


def test_run_mcp_builds_server_command():
    config = run_mcp.Config(
        path=run_mcp.PROJECT_DIR / "configs" / "mcp.yaml",
        tango_host="localhost",
        tango_port=9094,
        mcp=run_mcp.MCPConfig(
            name="Spectra300_MCP",
            transport="streamable-http",
            http_host="127.0.0.1",
            http_port=8123,
            data_device_address="asyncroscopy/data/default",
            quiet=True,
            blocked_classes=["DataBase"],
            blocked_functions={"*": ["Init"], "DATA": ["stop_tiled_server"]},
        ),
    )

    command = run_mcp.build_command(config)

    assert command[:5] == ["uv", "run", "python", "-m", "asyncroscopy.mcp.mcp_server"]
    assert "--class-name" not in command
    assert command[command.index("--name") + 1] == "Spectra300_MCP"
    assert command[command.index("--http-port") + 1] == "8123"
    assert "--quiet" in command
    assert command[command.index("--blocked-classes-json") + 1] == '["DataBase"]'
    assert command[command.index("--blocked-functions-json") + 1] == (
        '{"*": ["Init"], "DATA": ["stop_tiled_server"]}'
    )
    assert "--search-packages-json" not in command
