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


def test_load_spectra300_config_starts_servers_only():
    config = run_servers.load_config(run_servers.PROJECT_DIR / "configs" / "Spectra300.yaml")
    stage = next(device for device in config.support_devices if device.key == "stage")

    assert config.tango_host == "10.46.217.241"
    assert config.tiled.host == "10.46.217.241"
    assert config.tiled.register_on_startup is False
    assert config.instrument.class_name == "AutoScriptMicroscope"
    assert config.instrument.module_name == "asyncroscopy.instruments.electron_microscope.auto_script"
    assert stage.class_name == "AutoScriptSTAGE"
    assert stage.module_name == "asyncroscopy.instruments.electron_microscope.hardware.stage_autoscript"
    assert stage.properties["hardware_host"] == ["10.46.217.241"]
    assert stage.properties["hardware_port"] == ["9095"]
    assert config.reset_database_file is False
    assert not hasattr(config, "mcp")


def test_build_devices_adds_selected_instrument():
    config = run_servers.load_config(run_servers.PROJECT_DIR / "configs" / "Test.yaml")

    devices = run_servers.build_devices(config)
    stage = next(device for device in devices if device.key == "stage")

    assert stage.class_name == "TestStage"
    assert stage.module_name == "asyncroscopy.instruments.electron_microscope.hardware.TestStage"
    assert devices[-1].key == "instrument"
    assert devices[-1].class_name == "DigitalTwin"
    assert devices[-1].module_name == "asyncroscopy.instruments.electron_microscope.digital_twin"
    assert devices[-1].device_name == "asyncroscopy/instrument/default"


def test_load_mcp_config():
    config = run_mcp.load_config(run_mcp.PROJECT_DIR / "configs" / "mcp.yaml")

    assert config.mcp.name == "Spectra300_MCP"
    assert config.tango_host == "10.46.217.241"
    assert config.tango_port == 9094
    assert config.mcp.http_host == "0.0.0.0"
    assert config.mcp.http_port == 8000
    assert config.mcp.blocked_classes == ["DataBase", "DServer", "LLM"]
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
