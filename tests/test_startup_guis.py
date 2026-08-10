import time

from startup_guis import mcp_gui, server_gui, shared


def test_server_gui_builds_server_yaml():
    config = server_gui.server_config_from_values(
        {
            'instrument': {
                'class_name': 'AutoScriptMicroscope',
                'file': 'asyncroscopy/instruments/electron_microscope/auto_script.py',
                'description': 'Real microscope',
            },
            'instrument_file': 'asyncroscopy/instruments/electron_microscope/auto_script.py',
            'hardware_host': '10.0.0.1',
            'hardware_port': '9095',
            'hardware_timeout_seconds': '120',
            'devices': {
                'data': {'module_name': 'asyncroscopy.data.data'},
                'scan': {'module_name': 'asyncroscopy.instruments.electron_microscope.hardware.scan'},
            },
            'enabled_devices': {'data': True, 'scan': False},
            'tango_host': 'localhost',
            'tango_port': '9094',
            'reset_database_file': True,
            'tiled_host': 'localhost',
            'tiled_port': '9091',
            'acquisition_dir': 'outputs/tiled_acquisitions',
            'tiled_autostart': True,
            'tiled_register_on_startup': False,
            'device_timeout_seconds': '120',
        }
    )

    assert config['instrument']['file'] == 'asyncroscopy/instruments/electron_microscope/auto_script.py'
    assert config['instrument']['hardware_host'] == '10.0.0.1'
    assert config['instrument']['hardware_port'] == 9095
    assert config['instrument']['timeout_seconds'] == 120
    assert config['devices'] == {'data': {'module_name': 'asyncroscopy.data.data'}}
    assert config['tango'] == {'host': 'localhost', 'port': 9094, 'reset_database_file': True}
    assert config['tiled']['register_on_startup'] is False
    assert config['device_timeout_seconds'] == 120


def test_server_gui_omits_hardware_host_port_for_digital_twin_file():
    config = server_gui.server_config_from_values(
        {
            'instrument': {
                'class_name': 'AutoScriptMicroscope',
                'file': 'asyncroscopy/instruments/electron_microscope/auto_script.py',
                'description': 'Real microscope',
                'hardware_host': '10.0.0.1',
                'hardware_port': 9095,
            },
            'instrument_file': 'asyncroscopy/instruments/electron_microscope/digital_twin.py',
            'hardware_host': '10.0.0.1',
            'hardware_port': '9095',
            'hardware_timeout_seconds': '120',
            'devices': {'data': {'module_name': 'asyncroscopy.data.data'}},
            'enabled_devices': {'data': True},
            'tango_host': 'localhost',
            'tango_port': '9094',
            'reset_database_file': True,
            'tiled_host': 'localhost',
            'tiled_port': '9091',
            'acquisition_dir': 'outputs/tiled_acquisitions',
            'tiled_autostart': True,
            'tiled_register_on_startup': False,
            'device_timeout_seconds': '120',
        }
    )

    assert config['instrument']['class_name'] == 'DigitalTwin'
    assert 'hardware_host' not in config['instrument']
    assert 'hardware_port' not in config['instrument']


def test_server_gui_reads_and_writes_line_and_combo_inputs():
    class FakeLineEdit:
        def __init__(self):
            self.value = ''

        def text(self):
            return self.value

        def setText(self, value):
            self.value = value

    class FakeComboBox:
        def __init__(self):
            self.value = ''

        def currentText(self):
            return self.value

        def setCurrentText(self, value):
            self.value = value

    class FakeGui:
        input_text = server_gui.ServerGui.input_text
        set_input_text = server_gui.ServerGui.set_input_text

    gui = FakeGui()
    gui.inputs = {'line': FakeLineEdit(), 'combo': FakeComboBox()}

    gui.set_input_text('line', 'localhost')
    gui.set_input_text('combo', server_gui.PROJECT_DIR / 'outputs' / 'tiled_acquisitions')

    assert gui.input_text('line') == 'localhost'
    assert gui.input_text('combo') == 'outputs/tiled_acquisitions'


def test_mcp_gui_builds_mcp_yaml():
    config = mcp_gui.mcp_config_from_values(
        {
            'tango_host': '10.0.0.2',
            'tango_port': '9094',
            'name': 'Spectra300_MCP',
            'transport': 'streamable-http',
            'http_host': '0.0.0.0',
            'http_port': '8000',
            'data_device_address': 'asyncroscopy/data/default',
            'quiet': True,
            'blocked_classes': 'DataBase, DServer',
            'blocked_functions': '"*":\n  - Init\n  - Kill\n',
        }
    )

    assert config['tango'] == {'host': '10.0.0.2', 'port': 9094}
    assert config['mcp']['http_host'] == '0.0.0.0'
    assert config['mcp']['blocked_classes'] == ['DataBase', 'DServer']
    assert config['mcp']['blocked_functions'] == {'*': ['Init', 'Kill']}


def test_discover_instrument_configs_finds_any_instrument_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, 'CONFIG_DIR', tmp_path)
    (tmp_path / 'jeol.yaml').write_text('instrument:\n  description: JEOL JEM-F200\ntango:\n  host: 10.0.0.5\n  port: 9094\n')
    (tmp_path / 'mcp_only.yaml').write_text('tango:\n  host: 10.0.0.9\n  port: 9094\nmcp:\n  name: x\n')

    found = shared.discover_instrument_configs()

    assert found == [('JEOL JEM-F200', '10.0.0.5', 9094)]


def test_discover_instrument_configs_falls_back_to_filename_without_description(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, 'CONFIG_DIR', tmp_path)
    (tmp_path / 'unlabeled.yaml').write_text('instrument:\n  class_name: Foo\ntango:\n  host: localhost\n  port: 9094\n')

    found = shared.discover_instrument_configs()

    assert found == [('unlabeled', 'localhost', 9094)]


def test_resolve_default_tango_prefers_last_started_server_config(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, 'GENERATED_CONFIG_DIR', tmp_path)
    (tmp_path / 'server_gui.yaml').write_text('tango:\n  host: 192.168.1.50\n  port: 9094\n')

    host, port = shared.resolve_default_tango({'tango': {'host': 'stale-host', 'port': 1234}})

    assert (host, port) == ('192.168.1.50', 9094)


def test_resolve_default_tango_falls_back_when_server_gui_never_ran(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, 'GENERATED_CONFIG_DIR', tmp_path)

    host, port = shared.resolve_default_tango({'tango': {'host': 'fallback-host', 'port': 4321}})

    assert (host, port) == ('fallback-host', 4321)


def test_managed_command_stop_delegates_to_process_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(shared.ProcessManager, 'cleanup_stale_state', lambda self: None)
    output_lines = []
    done_calls = []
    command = shared.ManagedCommand(output_lines.append, done_calls.append, name='test_gui')
    command._manager.state_dir = tmp_path
    command._manager.state_file = tmp_path / 'test_gui.json'

    stopped = []
    monkeypatch.setattr(command._manager, 'stop_process', stopped.append)

    class FakeManaged:
        running = True

    command._managed = FakeManaged()
    command.stop()

    deadline = time.time() + 1
    while not stopped and time.time() < deadline:
        time.sleep(0.01)

    assert stopped == [command._managed]
    assert 'Stop requested.\n' in output_lines


def test_managed_command_shutdown_delegates_to_process_manager(monkeypatch):
    monkeypatch.setattr(shared.ProcessManager, 'cleanup_stale_state', lambda self: None)
    command = shared.ManagedCommand(lambda _line: None, lambda _code: None, name='test_gui')

    calls = []
    monkeypatch.setattr(command._manager, 'shutdown_all', lambda: calls.append(True))

    command.shutdown()

    assert calls == [True]
