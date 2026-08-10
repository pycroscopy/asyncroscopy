#!/usr/bin/env python
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Import Qt through qt_compat so this GUI can use PyQt6 normally and PyQt5 on
# legacy Windows 10 systems that cannot load Qt6.
from startup_guis.qt_compat import (  # noqa: E402
    HORIZONTAL,
    NO_WRAP,
    VERTICAL,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
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
from startup_guis.shared import BODY_FONT, CONFIG_DIR, DIGITAL_TWIN_HOST, GENERATED_CONFIG_DIR, SPECTRA300_HOST, TEXT_FONT, TITLE_FONT, CheckBox, CollapsibleSection, HostToggle, ManagedCommand, action_button, append_terminal_text, apply_theme, configure_splitter, configure_terminal, load_yaml, scrollable, section_label, set_tool_count_badge, tool_count_badge, write_yaml, yaml_text  # noqa: E402


DEFAULT_CONFIG_PATH = CONFIG_DIR / 'mcp.yaml'
GENERATED_CONFIG_PATH = GENERATED_CONFIG_DIR / 'mcp_gui.yaml'
TOOL_COUNT_PATTERN = re.compile(r'MCP ready: (\d+) tool')


def mcp_config_from_values(values: dict) -> dict:
    blocked_functions = yaml.safe_load(values['blocked_functions']) if values['blocked_functions'].strip() else {}
    return {
        'tango': {'host': values['tango_host'], 'port': int(values['tango_port'])},
        'mcp': {
            'name': values['name'],
            'transport': values['transport'],
            'http_host': values['http_host'],
            'http_port': int(values['http_port']),
            'data_device_address': values['data_device_address'],
            'quiet': values['quiet'],
            'blocked_classes': [item.strip() for item in values['blocked_classes'].split(',') if item.strip()],
            'blocked_functions': blocked_functions or {},
        },
    }


class McpGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Asyncroscopy MCP Startup')
        self.resize(1280, 960)
        self.setMinimumSize(880, 560)
        self.command = ManagedCommand(self.enqueue_output, self.process_done)
        self.default_config = load_yaml(DEFAULT_CONFIG_PATH)
        self.inputs: dict[str, QLineEdit | QComboBox | QCheckBox] = {}
        self.build()
        self.refresh_yaml()

    def build(self) -> None:
        apply_theme(self)
        root = QSplitter(VERTICAL)
        top = QSplitter(HORIZONTAL)
        configure_splitter(root)
        configure_splitter(top)
        controls = QWidget()
        preview = QWidget()
        terminal = QWidget()
        root.addWidget(top)
        root.addWidget(terminal)
        top.addWidget(scrollable(controls))
        top.addWidget(preview)
        top.setStretchFactor(0, 0)
        top.setStretchFactor(1, 1)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setSizes([400, 560])
        top.setSizes([620, 640])
        container = QWidget()
        container.setObjectName('appRoot')
        wrapper = QVBoxLayout(container)
        wrapper.setContentsMargins(14, 14, 14, 14)
        wrapper.addWidget(root)
        self.setCentralWidget(container)

        self.build_controls(controls)
        self.build_preview(preview)
        self.build_terminal(terminal)

    def build_controls(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 12, 0)
        layout.setSpacing(10)
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title = QLabel('Asyncroscopy MCP Startup')
        title.setFont(TITLE_FONT)
        title_row.addWidget(title)
        self.tool_badge = tool_count_badge()
        title_row.addWidget(self.tool_badge)
        title_row.addStretch()
        self.host_toggle = HostToggle(self.apply_host_preset)
        title_row.addWidget(self.host_toggle)
        layout.addLayout(title_row)

        tango = self.default_config['tango']
        database = self.section('Database', expanded=False)
        self.add_row(database, 'Tango host', self.line_input('tango_host', tango.get('host', 'localhost')))
        self.inputs['tango_host'].textChanged.connect(self.sync_host_toggle)
        self.add_row(database, 'Tango port', self.line_input('tango_port', tango.get('port', 9094)))
        layout.addWidget(database)
        self.sync_host_toggle()

        mcp = self.default_config['mcp']
        mcp_server = self.section('MCP server', expanded=False)
        self.add_row(mcp_server, 'Name', self.line_input('name', mcp.get('name', 'Spectra300_MCP')))
        transport = QComboBox()
        transport.addItems(['streamable-http'])
        transport.setCurrentText(mcp.get('transport', 'streamable-http'))
        transport.currentTextChanged.connect(self.refresh_yaml)
        self.inputs['transport'] = transport
        self.add_row(mcp_server, 'Transport', transport)
        self.add_row(mcp_server, 'HTTP host', self.line_input('http_host', mcp.get('http_host', '127.0.0.1')))
        self.add_row(mcp_server, 'HTTP port', self.line_input('http_port', mcp.get('http_port', 8000)))
        quiet = self.check_input('quiet', 'Quiet mode', bool(mcp.get('quiet', True)))
        mcp_server.form.addRow('', quiet)
        layout.addWidget(mcp_server)

        data_access = self.section('Data access', expanded=False)
        self.add_row(data_access, 'DATA device', self.line_input('data_device_address', mcp.get('data_device_address', 'asyncroscopy/data/default')))
        layout.addWidget(data_access)

        access_control = self.section('Access control', expanded=False)
        self.add_row(access_control, 'Blocked classes', self.line_input('blocked_classes', ', '.join(mcp.get('blocked_classes', []))))
        blocked_label = section_label('Blocked functions YAML')
        access_control.form.addRow(blocked_label)
        self.blocked_functions = QTextEdit()
        self.blocked_functions.setFont(TEXT_FONT)
        self.blocked_functions.setLineWrapMode(NO_WRAP)
        self.blocked_functions.setMinimumHeight(120)
        self.blocked_functions.setPlainText(yaml.safe_dump(mcp.get('blocked_functions', {}), sort_keys=False))
        self.blocked_functions.textChanged.connect(self.refresh_yaml)
        access_control.form.addRow(self.blocked_functions)
        layout.addWidget(access_control)

        actions = QHBoxLayout()
        actions.setSpacing(8)
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
            button.setMinimumHeight(40)
            actions.addWidget(button)
        layout.addLayout(actions)
        layout.addStretch()

    def build_preview(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(12, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(section_label('Configuration (.yaml)'))
        self.yaml_preview = QTextEdit()
        self.yaml_preview.setFont(TEXT_FONT)
        self.yaml_preview.setReadOnly(True)
        self.yaml_preview.setLineWrapMode(NO_WRAP)
        layout.addWidget(self.yaml_preview)

    def build_terminal(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(section_label('Terminal output'))
        self.output = QTextEdit()
        configure_terminal(self.output)
        layout.addWidget(self.output)

    def section(self, title: str, layout_cls=QFormLayout, expanded: bool = True) -> CollapsibleSection:
        return CollapsibleSection(title, layout_cls=layout_cls, expanded=expanded)

    def add_row(self, group: CollapsibleSection, label: str, widget: QWidget) -> None:
        group.form.addRow(label, widget)

    def line_input(self, key: str, value) -> QLineEdit:
        widget = QLineEdit(str(value))
        widget.textChanged.connect(self.refresh_yaml)
        self.inputs[key] = widget
        return widget

    def check_input(self, key: str, label: str, checked: bool) -> QCheckBox:
        widget = CheckBox(label)
        widget.setChecked(checked)
        widget.stateChanged.connect(self.refresh_yaml)
        self.inputs[key] = widget
        return widget

    def apply_host_preset(self, host: str) -> None:
        self.inputs['tango_host'].setText(host)

    def sync_host_toggle(self) -> None:
        """Reflect whether the Tango host field currently matches a known preset."""
        host = self.inputs['tango_host'].text()
        key = {DIGITAL_TWIN_HOST: 'digital_twin', SPECTRA300_HOST: 'spectra300'}.get(host)
        self.host_toggle.set_active(key)

    def current_config(self) -> dict:
        values = {
            'tango_host': self.inputs['tango_host'].text(),
            'tango_port': self.inputs['tango_port'].text(),
            'name': self.inputs['name'].text(),
            'transport': self.inputs['transport'].currentText(),
            'http_host': self.inputs['http_host'].text(),
            'http_port': self.inputs['http_port'].text(),
            'data_device_address': self.inputs['data_device_address'].text(),
            'quiet': self.inputs['quiet'].isChecked(),
            'blocked_classes': self.inputs['blocked_classes'].text(),
            'blocked_functions': self.blocked_functions.toPlainText() if hasattr(self, 'blocked_functions') else '',
        }
        return mcp_config_from_values(values)

    def refresh_yaml(self) -> None:
        if not hasattr(self, 'yaml_preview'):
            return
        try:
            self.yaml_preview.setPlainText(yaml_text(self.current_config()))
        except yaml.YAMLError as exc:
            self.yaml_preview.setPlainText(f'Invalid blocked_functions YAML: {exc}')

    def save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Save config', str(CONFIG_DIR / 'mcp_config.yaml'), 'YAML (*.yaml *.yml);;All files (*)')
        if path:
            write_yaml(Path(path), self.current_config())
            self.enqueue_output(f'Saved {path}\n')

    def read_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'Load config', str(CONFIG_DIR), 'YAML (*.yaml *.yml);;All files (*)')
        if not path:
            return
        config = load_yaml(Path(path))
        tango = config.get('tango', {})
        mcp = config.get('mcp', {})
        self.inputs['tango_host'].setText(str(tango.get('host', 'localhost')))
        self.inputs['tango_port'].setText(str(tango.get('port', 9094)))
        self.inputs['name'].setText(mcp.get('name', 'Spectra300_MCP'))
        self.inputs['transport'].setCurrentText(mcp.get('transport', 'streamable-http'))
        self.inputs['http_host'].setText(mcp.get('http_host', '127.0.0.1'))
        self.inputs['http_port'].setText(str(mcp.get('http_port', 8000)))
        self.inputs['data_device_address'].setText(mcp.get('data_device_address', 'asyncroscopy/data/default'))
        self.inputs['quiet'].setChecked(bool(mcp.get('quiet', True)))
        self.inputs['blocked_classes'].setText(', '.join(mcp.get('blocked_classes', [])))
        self.blocked_functions.setPlainText(yaml.safe_dump(mcp.get('blocked_functions', {}), sort_keys=False))
        self.refresh_yaml()
        self.enqueue_output(f'Loaded {path}\n')

    def start(self) -> None:
        set_tool_count_badge(self.tool_badge, None)
        config_path = write_yaml(GENERATED_CONFIG_PATH, self.current_config())
        self.command.start(['uv', 'run', 'python', '-u', 'startup_scripts/run_mcp.py', '--yaml', str(config_path)])

    def enqueue_output(self, text: str) -> None:
        append_terminal_text(self.output, text)
        match = TOOL_COUNT_PATTERN.search(text)
        if match:
            set_tool_count_badge(self.tool_badge, int(match.group(1)))

    def process_done(self, returncode: int | None) -> None:
        self.enqueue_output(f'\nProcess exited with return code {returncode}.\n')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = McpGui()
    window.show()
    sys.exit(app_exec(app))
