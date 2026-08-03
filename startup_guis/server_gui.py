#!/usr/bin/env python 
from __future__ import annotations 

import ast 
import sys 
from pathlib import Path 

PROJECT_DIR = Path(__file__).resolve().parents[1] 
if str(PROJECT_DIR) not in sys.path: 
    sys.path.insert(0, str(PROJECT_DIR)) 

# Import Qt through qt_compat so this GUI can use PyQt6 normally and PyQt5 on 
# legacy Windows 10 systems that cannot load Qt6. 
from startup_guis.qt_compat import (  # noqa: E402 
    HORIZONTAL, 
    VERTICAL, 
    QApplication, 
    QCheckBox, 
    QComboBox, 
    QFileDialog, 
    QFormLayout, 
    QGridLayout, 
    QGroupBox, 
    QHBoxLayout, 
    QLabel, 
    QLineEdit, 
    QMainWindow, 
    QPushButton, 
    QSplitter, 
    QTextEdit, 
    QVBoxLayout, 
    QWidget, 
    app_exec, 
) 
from startup_guis.shared import BODY_FONT, CONFIG_DIR, GENERATED_CONFIG_DIR, SECTION_FONT, TITLE_FONT, ManagedCommand, action_button, append_terminal_text, configure_terminal, load_yaml, write_yaml  # noqa: E402 


DEFAULT_CONFIG_PATH = CONFIG_DIR / 'Spectra300.yaml' 
GENERATED_CONFIG_PATH = GENERATED_CONFIG_DIR / 'server_gui.yaml' 
DEVICE_MODULES = { 
    'camera': 'asyncroscopy.instruments.electron_microscope.detectors.camera', 
    'corrector': 'asyncroscopy.instruments.electron_microscope.hardware.corrector', 
    'data': 'asyncroscopy.data.data', 
    'eds': 'asyncroscopy.instruments.electron_microscope.detectors.eds', 
    'flucam': 'asyncroscopy.instruments.electron_microscope.detectors.flucam', 
    'scan': 'asyncroscopy.instruments.electron_microscope.hardware.scan', 
    'stage': 'asyncroscopy.instruments.electron_microscope.hardware.stage', 
} 
INSTRUMENT_FILES = [ 
    'asyncroscopy/instruments/electron_microscope/auto_script.py', 
    'asyncroscopy/instruments/electron_microscope/jeol.py', 
    'asyncroscopy/instruments/electron_microscope/digital_twin.py', 
    'asyncroscopy/instruments/scanning_probe_microscope/jupyter_api.py', 
] 


def project_path_text(path: Path | str) -> str: 
    path = Path(path) 
    if path.is_absolute(): 
        try: 
            return path.relative_to(PROJECT_DIR).as_posix() 
        except ValueError: 
            return path.as_posix() 
    return path.as_posix() 


def class_name_from_file(path_text: str, fallback: str = 'Instrument') -> str: 
    path = Path(path_text).expanduser() 
    if not path.is_absolute(): 
        path = PROJECT_DIR / path 
    try: 
        tree = ast.parse(path.read_text(encoding='utf-8')) 
    except OSError: 
        return fallback 
    for node in tree.body: 
        if isinstance(node, ast.ClassDef): 
            return node.name 
    return fallback 


def uses_hardware_connection(path_text: str) -> bool: 
    return 'digital_twin' not in Path(path_text).stem 


def server_config_from_values(values: dict) -> dict: 
    devices = {key: spec for key, spec in values['devices'].items() if values['enabled_devices'][key]} 
    instrument = dict(values['instrument']) 
    selected_file = project_path_text(values['instrument_file']) 
    previous_file = project_path_text(instrument.get('file', selected_file)) 
    instrument['file'] = selected_file 
    if selected_file != previous_file or not instrument.get('class_name'): 
        instrument['class_name'] = class_name_from_file(selected_file) 
    instrument.pop('hardware_host', None) 
    instrument.pop('hardware_port', None) 
    uses_hardware = uses_hardware_connection(selected_file) 
    if uses_hardware and values['hardware_host']: 
        instrument['hardware_host'] = values['hardware_host'] 
    if uses_hardware and values['hardware_port']: 
        instrument['hardware_port'] = int(values['hardware_port']) 
    if values['hardware_timeout_seconds']: 
        instrument['timeout_seconds'] = int(values['hardware_timeout_seconds']) 
    return { 
        'instrument': instrument, 
        'devices': devices, 
        'tango': { 
            'host': values['tango_host'], 
            'port': int(values['tango_port']), 
            'reset_database_file': values['reset_database_file'], 
        }, 
        'tiled': { 
            'host': values['tiled_host'], 
            'port': int(values['tiled_port']), 
            'acquisition_dir': values['acquisition_dir'], 
            'autostart': values['tiled_autostart'], 
            'register_on_startup': values['tiled_register_on_startup'], 
        }, 
        'device_timeout_seconds': int(values['device_timeout_seconds']), 
    } 


class ServerGui(QMainWindow): 
    def __init__(self): 
        super().__init__() 
        self.setWindowTitle('Asyncroscopy Server Startup') 
        self.resize(800, 600) 
        self.command = ManagedCommand(self.enqueue_output, self.process_done) 
        self.default_config = load_yaml(DEFAULT_CONFIG_PATH) 
        self.device_config = self.default_config.get('devices', {}) 
        self.inputs: dict[str, QLineEdit | QComboBox | QCheckBox] = {} 
        self.device_checks: dict[str, QCheckBox] = {} 
        self.build() 

    def build(self) -> None: 
        self.setFont(BODY_FONT) 
        root = QSplitter(VERTICAL) 
        controls = QWidget() 
        terminal = QWidget() 
        root.addWidget(controls) 
        root.addWidget(terminal) 
        root.setSizes([400, 200]) 
        self.setCentralWidget(root) 

        self.build_controls(controls) 
        self.build_terminal(terminal) 

    def build_controls(self, parent: QWidget) -> None: 
        layout = QVBoxLayout(parent) 
        title_layout = QHBoxLayout() 
        title = QLabel('Asyncroscopy Server Startup') 
        title.setFont(TITLE_FONT) 
        title_layout.addWidget(title) 
        title_layout.addStretch() 
        self.config_combo = QComboBox() 
        title_layout.addWidget(self.config_combo) 
        layout.addLayout(title_layout) 

        database = self.section('Database') 
        self.add_row(database, 'Tango host', self.line_input('tango_host', self.default_config['tango'].get('host', 'localhost'))) 
        self.add_row(database, 'Tango port', self.line_input('tango_port', self.default_config['tango'].get('port', 9094))) 
        reset_database = self.check_input('reset_database_file', 'Delete tango_database.db before start', bool(self.default_config['tango'].get('reset_database_file', False))) 
        database.layout().addRow('', reset_database) 
        layout.addWidget(database) 

        instrument = self.section('Instrument') 
        default_instrument = self.default_config['instrument'] 
        self.add_row(instrument, 'Instrument file', self.path_input('instrument_file', default_instrument.get('file', INSTRUMENT_FILES[0]), files=INSTRUMENT_FILES)) 
        self.add_row(instrument, 'Hardware host', self.line_input('hardware_host', default_instrument.get('hardware_host', ''))) 
        self.add_row(instrument, 'Hardware port', self.line_input('hardware_port', default_instrument.get('hardware_port', 9095))) 
        self.add_row(instrument, 'Timeout (seconds)', self.line_input('hardware_timeout_seconds', default_instrument.get('timeout_seconds', 120))) 
        self.add_row(instrument, 'Device startup timeout', self.line_input('device_timeout_seconds', self.default_config.get('device_timeout_seconds', 120))) 
        layout.addWidget(instrument) 

        tiled = self.default_config['tiled'] 
         
        data_server_container = QWidget() 
        data_server_layout = QVBoxLayout(data_server_container) 
        data_server_layout.setContentsMargins(0, 0, 0, 0) 
         
        data_server_header = QPushButton('▼ Data server') 
        data_server_header.setFlat(True) 
        data_server_header.setFont(SECTION_FONT) 
        data_server_layout.addWidget(data_server_header) 
         
        data_server = self.section('') 
        data_server.setStyleSheet("QGroupBox { border: none; padding-top: 0px; margin-top: 0px; }") 
        self.add_row(data_server, 'Tiled host', self.line_input('tiled_host', tiled.get('host', 'localhost'))) 
        self.add_row(data_server, 'Tiled port', self.line_input('tiled_port', tiled.get('port', 9091))) 
        self.add_row(data_server, 'Acquisition dir', self.path_input('acquisition_dir', tiled.get('acquisition_dir', 'outputs/tiled_acquisitions'), directory=True)) 
        autostart = self.check_input('tiled_autostart', 'Start Tiled HTTP server', bool(tiled.get('autostart', True))) 
        data_server.layout().addRow('', autostart) 
        register_on_startup = self.check_input( 
            'tiled_register_on_startup', 
            'Register acquisition directory on startup (slow for large folders)', 
            bool(tiled.get('register_on_startup', False)), 
        ) 
        data_server.layout().addRow('', register_on_startup) 
        data_server.setHidden(True) 
        data_server_layout.addWidget(data_server) 
         
        def toggle_data_server(): 
            is_hidden = data_server.isHidden() 
            data_server.setHidden(not is_hidden) 
            data_server_header.setText(('▼ ' if is_hidden else '► ') + 'Data server') 
         
        data_server_header.clicked.connect(toggle_data_server) 
        layout.addWidget(data_server_container) 

        devices = QGroupBox('Devices') 
        devices.setFont(SECTION_FONT) 
        device_grid = QGridLayout(devices) 
        for index, key in enumerate(DEVICE_MODULES): 
            checkbox = QCheckBox(key) 
            checkbox.setChecked(key in self.device_config) 
            checkbox.stateChanged.connect(self.refresh_yaml) 
            self.device_checks[key] = checkbox 
            device_grid.addWidget(checkbox, index // 3, index % 3) 
        layout.addWidget(devices) 

        actions = QHBoxLayout() 
        start = action_button('Start', '#1f7a35', '#2ea043') 
        stop = action_button('Stop', '#b42318', '#dc2626') 
        load = QPushButton('Load config file') 
        save = QPushButton('Save current config') 
        start.clicked.connect(self.start) 
        stop.clicked.connect(self.command.stop) 
        load.clicked.connect(self.read_config) 
        save.clicked.connect(self.save_config) 
        for button in (start, stop, load, save): 
            button.setFont(BODY_FONT) 
            actions.addWidget(button) 
        layout.addLayout(actions) 
        layout.addStretch() 

        config_files = sorted([p.name for p in CONFIG_DIR.glob('*.yaml')] + [p.name for p in CONFIG_DIR.glob('*.yml')]) 
        self.config_combo.addItems(config_files) 
        self.config_combo.setCurrentText(DEFAULT_CONFIG_PATH.name) 
        self.config_combo.currentTextChanged.connect(self.config_changed) 

    def build_terminal(self, parent: QWidget) -> None: 
        layout = QVBoxLayout(parent) 
        label = QLabel('Terminal output') 
        label.setFont(SECTION_FONT) 
        layout.addWidget(label) 
        self.output = QTextEdit() 
        configure_terminal(self.output) 
        layout.addWidget(self.output) 

    def section(self, title: str) -> QGroupBox: 
        group = QGroupBox(title) 
        group.setFont(SECTION_FONT) 
        group.setLayout(QFormLayout()) 
        return group 

    def add_row(self, group: QGroupBox, label: str, widget: QWidget) -> None: 
        group.layout().addRow(label, widget) 

    def line_input(self, key: str, value) -> QLineEdit: 
        widget = QLineEdit(str(value)) 
        widget.textChanged.connect(self.refresh_yaml) 
        self.inputs[key] = widget 
        return widget 

    def check_input(self, key: str, label: str, checked: bool) -> QCheckBox: 
        widget = QCheckBox(label) 
        widget.setChecked(checked) 
        widget.stateChanged.connect(self.refresh_yaml) 
        self.inputs[key] = widget 
        return widget 

    def path_input(self, key: str, value, files: list[str] | None = None, directory: bool = False) -> QWidget: 
        row = QWidget() 
        layout = QHBoxLayout(row) 
        layout.setContentsMargins(0, 0, 0, 0) 
        combo = QComboBox() 
        combo.setEditable(True) 
        combo.addItems(files or [str(value)]) 
        combo.setCurrentText(project_path_text(value)) 
        combo.currentTextChanged.connect(self.refresh_yaml) 
        browse = QPushButton('Browse') 
        browse.clicked.connect(lambda: self.browse_path(combo, directory)) 
        layout.addWidget(combo) 
        layout.addWidget(browse) 
        self.inputs[key] = combo 
        return row 

    def browse_path(self, combo: QComboBox, directory: bool) -> None: 
        if directory: 
            path = QFileDialog.getExistingDirectory(self, 'Select directory', combo.currentText()) 
        else: 
            path, _ = QFileDialog.getOpenFileName(self, 'Select Python file', str(PROJECT_DIR), 'Python (*.py);;All files (*)') 
        if path: 
            combo.setCurrentText(project_path_text(path)) 

    def input_text(self, key: str) -> str: 
        widget = self.inputs[key] 
        if hasattr(widget, 'currentText'): 
            return widget.currentText() 
        return widget.text() 

    def set_input_text(self, key: str, value) -> None: 
        widget = self.inputs[key] 
        text = str(value) 
        if hasattr(widget, 'setCurrentText'): 
            widget.setCurrentText(project_path_text(text)) 
            return 
        widget.setText(text) 

    def current_config(self) -> dict: 
        values = { 
            'instrument_file': self.input_text('instrument_file'), 
            'hardware_host': self.input_text('hardware_host'), 
            'hardware_port': self.input_text('hardware_port'), 
            'hardware_timeout_seconds': self.input_text('hardware_timeout_seconds'), 
            'tango_host': self.input_text('tango_host'), 
            'tango_port': self.input_text('tango_port'), 
            'reset_database_file': self.inputs['reset_database_file'].isChecked(), 
            'tiled_host': self.input_text('tiled_host'), 
            'tiled_port': self.input_text('tiled_port'), 
            'acquisition_dir': self.input_text('acquisition_dir'), 
            'tiled_autostart': self.inputs['tiled_autostart'].isChecked(), 
            'tiled_register_on_startup': self.inputs['tiled_register_on_startup'].isChecked(), 
            'device_timeout_seconds': self.input_text('device_timeout_seconds'), 
            'enabled_devices': {key: checkbox.isChecked() for key, checkbox in self.device_checks.items()}, 
            'devices': {key: self.device_config.get(key, {'module_name': DEVICE_MODULES[key]}) for key in DEVICE_MODULES}, 
            'instrument': self.default_config['instrument'], 
        } 
        return server_config_from_values(values) 

    def refresh_yaml(self) -> None: 
        pass 

    def save_config(self) -> None: 
        path, _ = QFileDialog.getSaveFileName(self, 'Save config', str(CONFIG_DIR / 'server_config.yaml'), 'YAML (*.yaml *.yml);;All files (*)') 
        if path: 
            write_yaml(Path(path), self.current_config()) 
            self.enqueue_output(f'Saved {path}\n') 

    def config_changed(self, filename: str) -> None: 
        if filename: 
            self.load_config_from_path(CONFIG_DIR / filename) 

    def load_config_from_path(self, path: Path | str) -> None: 
        config = load_yaml(Path(path)) 
        self.default_config = config 
        self.device_config = config.get('devices', {}) 
        instrument = config.get('instrument', {}) 
        tango = config.get('tango', {}) 
        tiled = config.get('tiled', {}) 
        self.set_input_text('instrument_file', instrument.get('file', INSTRUMENT_FILES[0])) 
        self.set_input_text('hardware_host', instrument.get('hardware_host', '')) 
        self.set_input_text('hardware_port', instrument.get('hardware_port', '')) 
        self.set_input_text('hardware_timeout_seconds', instrument.get('timeout_seconds', 120)) 
        self.set_input_text('tango_host', tango.get('host', 'localhost')) 
        self.set_input_text('tango_port', tango.get('port', 9094)) 
        self.inputs['reset_database_file'].setChecked(bool(tango.get('reset_database_file', False))) 
        self.set_input_text('tiled_host', tiled.get('host', 'localhost')) 
        self.set_input_text('tiled_port', tiled.get('port', 9091)) 
        self.set_input_text('acquisition_dir', tiled.get('acquisition_dir', 'outputs/tiled_acquisitions')) 
        self.inputs['tiled_autostart'].setChecked(bool(tiled.get('autostart', True))) 
        self.inputs['tiled_register_on_startup'].setChecked(bool(tiled.get('register_on_startup', False))) 
        self.set_input_text('device_timeout_seconds', config.get('device_timeout_seconds', 120)) 
        for key, checkbox in self.device_checks.items(): 
            checkbox.setChecked(key in self.device_config) 
        self.refresh_yaml() 
        self.enqueue_output(f'Loaded {path}\n') 
        filename = Path(path).name 
        if self.config_combo.findText(filename) != -1: 
            self.config_combo.blockSignals(True) 
            self.config_combo.setCurrentText(filename) 
            self.config_combo.blockSignals(False) 

    def read_config(self) -> None: 
        path, _ = QFileDialog.getOpenFileName(self, 'Load config', str(CONFIG_DIR), 'YAML (*.yaml *.yml);;All files (*)') 
        if not path: 
            return 
        self.load_config_from_path(path) 

    def start(self) -> None: 
        config_path = write_yaml(GENERATED_CONFIG_PATH, self.current_config()) 
        self.command.start(['uv', 'run', 'python', '-u', 'startup_scripts/run_servers.py', '--yaml', str(config_path)]) 

    def enqueue_output(self, text: str) -> None: 
        append_terminal_text(self.output, text) 

    def process_done(self, returncode: int | None) -> None: 
        self.enqueue_output(f'\nProcess exited with return code {returncode}.\n') 


if __name__ == '__main__': 
    app = QApplication(sys.argv) 
    window = ServerGui() 
    window.show() 
    sys.exit(app_exec(app))