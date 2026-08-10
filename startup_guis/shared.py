from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.utils.process_manager import ManagedProcess, ProcessManager  # noqa: E402

from startup_guis.qt_compat import (
    FIELDS_STAY_AT_SIZE_HINT,
    FONT_BOLD,
    FONT_MEDIUM,
    MONOSPACE_HINT,
    MOVE_END,
    NO_FRAME,
    PEN_CAP_ROUND,
    PEN_JOIN_ROUND,
    POINTING_HAND_CURSOR,
    RENDER_HINT_ANTIALIASING,
    SCROLLBAR_AS_NEEDED,
    SE_CHECKBOX_INDICATOR,
    QApplication,
    QCheckBox,
    QColor,
    QComboBox,
    QFont,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QObject,
    QPainter,
    QPainterPath,
    QPen,
    QPointF,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStyleOptionButton,
    QTextCharFormat,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    pyqtSignal,
    window_palette_color,
)


CONFIG_DIR = PROJECT_DIR / 'configs'
GENERATED_CONFIG_DIR = PROJECT_DIR / 'outputs' / 'startup_configs'

DARK_COLORS = {
    'window': '#0f1319',
    'panel': '#161b22',
    'panel_hover': '#1c232c',
    'field': '#0d1117',
    'border': '#2b3440',
    'border_strong': '#3d4854',
    'text': '#e6edf3',
    'text_dim': '#8b98a5',
    'accent': '#58a6ff',
    'check_mark': '#0b0f14',
    'terminal_bg': '#0b0f14',
    'terminal_text': '#c9d1d9',
}
LIGHT_COLORS = {
    'window': '#f5f6f8',
    'panel': '#ffffff',
    'panel_hover': '#eef0f3',
    'field': '#ffffff',
    'border': '#d0d7de',
    'border_strong': '#aeb7c2',
    'text': '#1f2328',
    'text_dim': '#59636e',
    'accent': '#0969da',
    'check_mark': '#ffffff',
    'terminal_bg': '#0b0f14',
    'terminal_text': '#c9d1d9',
}
COLORS = dict(DARK_COLORS)


def system_is_dark() -> bool:
    """Best-effort detection of the OS light/dark appearance setting."""
    app = QApplication.instance()
    if app is None:
        return True
    return window_palette_color(app).lightness() < 128


def _mono_family() -> str:
    if sys.platform == 'darwin':
        return 'Menlo'
    if os.name == 'nt':
        return 'Consolas'
    return 'DejaVu Sans Mono'


def _ui_font(size: int, weight=None) -> QFont:
    font = QFont()
    font.setPointSize(size)
    if weight is not None:
        font.setWeight(weight)
    return font


def _font(name: str) -> QFont:
    if name == 'BODY_FONT':
        return _ui_font(14)
    if name == 'TITLE_FONT':
        return _ui_font(21, FONT_BOLD)
    if name == 'SECTION_FONT':
        return _ui_font(15, FONT_MEDIUM)
    if name == 'LABEL_FONT':
        return _ui_font(12, FONT_MEDIUM)
    if name == 'TEXT_FONT':
        font = QFont(_mono_family())
        font.setPointSize(13)
        font.setStyleHint(MONOSPACE_HINT)
        return font
    if name == 'ACTION_FONT':
        return _ui_font(15, FONT_BOLD)
    raise AttributeError(name)


def __getattr__(name: str) -> QFont:
    try:
        return _font(name)
    except AttributeError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from None

OutputCallback = Callable[[str], None]
DoneCallback = Callable[[int | None], None]
ANSI_PATTERN = re.compile(r'\x1b\[[0-9;]*m')


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def yaml_text(config: dict) -> str:
    return yaml.safe_dump(config, sort_keys=False)


def write_yaml(path: Path, config: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_text(config), encoding='utf-8')
    return path


def resolve_default_tango(fallback: dict) -> tuple[str, int]:
    """Tango host/port to default a startup GUI to.

    Prefers whatever Server GUI last actually launched (`outputs/startup_configs/server_gui.yaml`),
    since that's the Tango DB that's genuinely running, over a separately hand-maintained config.
    """
    server_gui_path = GENERATED_CONFIG_DIR / 'server_gui.yaml'
    if server_gui_path.exists():
        try:
            tango = load_yaml(server_gui_path).get('tango', {})
        except (OSError, yaml.YAMLError):
            tango = {}
        if 'host' in tango and 'port' in tango:
            return str(tango['host']), int(tango['port'])
    tango = fallback.get('tango', {})
    return str(tango.get('host', 'localhost')), int(tango.get('port', 9094))


def app_stylesheet() -> str:
    """The dark, flat style shared by the startup windows."""
    c = COLORS
    return f'''
    QMainWindow, QDialog, QWidget#appRoot {{
        background: {c['window']};
    }}
    QWidget {{
        background: transparent;
        color: {c['text']};
    }}
    QLabel[role="heading"] {{
        color: {c['text']};
    }}
    QLabel[role="caption"] {{
        color: {c['text_dim']};
    }}
    QLineEdit, QComboBox, QTextEdit, QAbstractSpinBox {{
        background: {c['field']};
        color: {c['text']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 6px 8px;
        selection-background-color: {c['accent']};
        selection-color: #0b0f14;
    }}
    QLineEdit:hover, QComboBox:hover {{
        border-color: {c['border_strong']};
    }}
    QLineEdit:focus, QComboBox:focus, QTextEdit:focus {{
        border-color: {c['accent']};
    }}
    QComboBox QAbstractItemView {{
        background: {c['panel']};
        color: {c['text']};
        border: 1px solid {c['border']};
        selection-background-color: {c['accent']};
        selection-color: #0b0f14;
        outline: none;
    }}
    QPushButton {{
        background: {c['panel']};
        color: {c['text']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 7px 14px;
    }}
    QPushButton:hover {{
        background: {c['panel_hover']};
        border-color: {c['border_strong']};
    }}
    QPushButton:pressed {{
        background: {c['field']};
    }}
    QCheckBox {{
        background: transparent;
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 15px;
        height: 15px;
        border: 1px solid {c['text_dim']};
        border-radius: 4px;
        background: {c['field']};
    }}
    QCheckBox::indicator:hover {{
        border-color: {c['accent']};
    }}
    QCheckBox::indicator:checked {{
        background: {c['accent']};
        border-color: {c['accent']};
    }}
    QScrollArea {{
        border: none;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {c['border_strong']};
        border-radius: 5px;
        min-height: 32px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {c['text_dim']};
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 0px;
    }}
    QScrollBar::handle:horizontal {{
        background: {c['border_strong']};
        border-radius: 5px;
        min-width: 32px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {c['text_dim']};
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{
        width: 0px;
        height: 0px;
    }}
    QScrollBar::add-page, QScrollBar::sub-page {{
        background: transparent;
    }}
    '''


def apply_theme(window: QWidget) -> None:
    """Apply the shared palette and base font to a top-level window."""
    COLORS.clear()
    COLORS.update(DARK_COLORS if system_is_dark() else LIGHT_COLORS)
    window.setFont(_font('BODY_FONT'))
    window.setStyleSheet(app_stylesheet())


class CheckBox(QCheckBox):
    """A QCheckBox that paints a checkmark glyph over the filled indicator."""

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().paintEvent(event)
        if not self.isChecked():
            return
        option = QStyleOptionButton()
        self.initStyleOption(option)
        rect = self.style().subElementRect(SE_CHECKBOX_INDICATOR, option, self)
        painter = QPainter(self)
        painter.setRenderHint(RENDER_HINT_ANTIALIASING)
        pen = QPen(QColor(COLORS['check_mark']))
        pen.setWidthF(max(1.6, rect.width() * 0.16))
        pen.setCapStyle(PEN_CAP_ROUND)
        pen.setJoinStyle(PEN_JOIN_ROUND)
        painter.setPen(pen)
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        path = QPainterPath()
        path.moveTo(QPointF(x + w * 0.22, y + h * 0.54))
        path.lineTo(QPointF(x + w * 0.42, y + h * 0.74))
        path.lineTo(QPointF(x + w * 0.80, y + h * 0.28))
        painter.drawPath(path)
        painter.end()


def action_button(text: str, color: str, active_color: str) -> QPushButton:
    button = QPushButton(text)
    button.setFont(_font('ACTION_FONT'))
    button.setCursor(POINTING_HAND_CURSOR)
    button.setMinimumHeight(40)
    button.setStyleSheet(
        'QPushButton {'
        f'background: {color}; color: #ffffff; border: 1px solid {color};'
        'border-radius: 6px; padding: 8px 18px;'
        '}'
        'QPushButton:hover {'
        f'background: {active_color}; border-color: {active_color};'
        '}'
        'QPushButton:pressed {'
        f'background: {color}; border-color: {active_color};'
        '}'
    )
    return button


def section_label(text: str) -> QLabel:
    """A small, dimmed caption used above panes such as the terminal."""
    label = QLabel(text.upper())
    label.setFont(_font('LABEL_FONT'))
    label.setStyleSheet(f'color: {COLORS["text_dim"]}; letter-spacing: 1px; background: transparent;')
    return label


def tool_count_badge() -> QLabel:
    """A small pill next to a server name; starts neutral until a live count is known."""
    label = QLabel('no tools yet')
    label.setFont(_font('LABEL_FONT'))
    set_tool_count_badge(label, None)
    return label


def set_tool_count_badge(label: QLabel, count: int | None) -> None:
    """Update a tool_count_badge() label. Pass None to reset to the not-yet-known state."""
    ready = count is not None
    label.setText('no tools yet' if count is None else f'{count} tool{"" if count == 1 else "s"}')
    color = COLORS['accent'] if ready else COLORS['text_dim']
    border = COLORS['accent'] if ready else COLORS['border_strong']
    label.setStyleSheet(
        'QLabel {'
        f'color: {color}; border: 1px solid {border}; border-radius: 9px;'
        'padding: 1px 10px; background: transparent;'
        '}'
    )


def discover_instrument_configs() -> list[tuple[str, str, int]]:
    """(label, tango_host, tango_port) for each configs/*.yaml that defines an instrument.

    Any server-style config (has both `instrument` and `tango` sections) counts, so
    adding a new instrument is just dropping a new YAML file in configs/ - no code change.
    """
    found = []
    for path in sorted(CONFIG_DIR.glob('*.yaml')):
        try:
            data = load_yaml(path)
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict) or 'instrument' not in data:
            continue
        tango = data.get('tango')
        if not isinstance(tango, dict) or 'host' not in tango or 'port' not in tango:
            continue
        label = (data.get('instrument') or {}).get('description') or path.stem
        found.append((label, str(tango['host']), int(tango['port'])))
    return found


class InstrumentPicker(QWidget):
    """A dropdown of known instruments (configs/*.yaml with an `instrument` block) that applies the selected one's Tango host/port."""

    def __init__(self, on_change: Callable[[str, int], None], parent=None):
        super().__init__(parent)
        self._on_change = on_change
        self._instruments = discover_instrument_configs()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        self.combo.addItem('Instrument preset…')
        for label, _host, _port in self._instruments:
            self.combo.addItem(label)
        self.combo.currentIndexChanged.connect(self._select)
        layout.addWidget(self.combo)

    def _select(self, index: int) -> None:
        if index <= 0:
            return
        _label, host, port = self._instruments[index - 1]
        self._on_change(host, port)

    def set_active(self, host: str, port: int) -> None:
        """Reflect which known instrument (if any) currently matches host/port, without firing on_change."""
        self.combo.blockSignals(True)
        for row, (_label, preset_host, preset_port) in enumerate(self._instruments, start=1):
            if preset_host == host and preset_port == port:
                self.combo.setCurrentIndex(row)
                self.combo.blockSignals(False)
                return
        self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)


def configure_terminal(widget: QTextEdit) -> None:
    widget.setFont(_font('TEXT_FONT'))
    widget.setReadOnly(True)
    widget.setMinimumHeight(320)
    widget.setStyleSheet(
        'QTextEdit {'
        f'background: {COLORS["terminal_bg"]}; color: {COLORS["terminal_text"]};'
        f'border: 1px solid {COLORS["border"]}; border-radius: 8px; padding: 10px;'
        f'selection-background-color: {COLORS["accent"]}; selection-color: #0b0f14;'
        '}'
    )


def scrollable(content: QWidget) -> QScrollArea:
    """Wrap a controls pane so collapsing sections never squeeze the fields."""
    area = QScrollArea()
    area.setWidget(content)
    area.setWidgetResizable(True)
    area.setFrameShape(NO_FRAME)
    area.setHorizontalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
    area.setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
    return area


def configure_splitter(splitter: QSplitter, handle_width: int = 10) -> None:
    """Give a splitter a wide, visibly grabbable handle and stop panes from collapsing to zero."""
    splitter.setHandleWidth(handle_width)
    splitter.setChildrenCollapsible(False)
    splitter.setOpaqueResize(True)
    splitter.setStyleSheet(
        'QSplitter::handle { background: transparent; }'
        f'QSplitter::handle:horizontal {{ margin: 0px 4px; border-left: 2px solid {COLORS["border"]}; }}'
        f'QSplitter::handle:vertical {{ margin: 4px 0px; border-top: 2px solid {COLORS["border"]}; }}'
        f'QSplitter::handle:hover {{ border-color: {COLORS["accent"]}; }}'
        f'QSplitter::handle:pressed {{ border-color: {COLORS["accent"]}; }}'
    )


class CollapsibleSection(QWidget):
    """A titled card whose body can be shown or hidden by clicking the header, like a dropdown."""

    def __init__(self, title: str, layout_cls=QFormLayout, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._title = title

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.toggle = QPushButton()
        self.toggle.setFlat(True)
        self.toggle.setFont(_font('SECTION_FONT'))
        self.toggle.setCursor(POINTING_HAND_CURSOR)
        self.toggle.clicked.connect(lambda: self.set_expanded(not self._expanded))
        outer.addWidget(self.toggle)

        self.body = QWidget()
        self.body.setObjectName('sectionBody')
        self.form = layout_cls()
        self.form.setContentsMargins(14, 10, 14, 12)
        self.form.setSpacing(8)
        if isinstance(self.form, QFormLayout):
            self.form.setFieldGrowthPolicy(FIELDS_STAY_AT_SIZE_HINT)
            self.form.setHorizontalSpacing(14)
        self.body.setLayout(self.form)
        self.body.setStyleSheet(
            '#sectionBody {'
            f'background: {COLORS["panel"]}; border: 1px solid {COLORS["border"]};'
            'border-top: none; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px;'
            '}'
        )
        outer.addWidget(self.body)

        self._expanded = expanded
        self.set_expanded(expanded)

    def _style_toggle(self, expanded: bool) -> None:
        radius = '8px 8px 0px 0px' if expanded else '8px'
        self.toggle.setStyleSheet(
            'QPushButton {'
            f'background: {COLORS["panel"]}; color: {COLORS["text"]};'
            f'border: 1px solid {COLORS["border"]}; border-radius: {radius};'
            'text-align: left; padding: 9px 12px;'
            '}'
            'QPushButton:hover {'
            f'background: {COLORS["panel_hover"]}; border-color: {COLORS["border_strong"]};'
            '}'
        )

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self.body.setVisible(expanded)
        arrow = '⌄' if expanded else '›'
        self.toggle.setText(f'{arrow}   {self._title}')
        self._style_toggle(expanded)


def append_terminal_text(widget: QTextEdit, text: str) -> None:
    formats = {
        'command': _format('#79c0ff'),
        'ok': _format('#3fb950'),
        'run': _format('#39c5cf'),
        'wait': _format('#d29922'),
        'fail': _format('#ff7b72'),
        'skip': _format('#8b949e'),
        'plain': _format('#c9d1d9'),
    }
    cursor = widget.textCursor()
    cursor.movePosition(MOVE_END)
    for line in text.splitlines(keepends=True):
        clean = ANSI_PATTERN.sub('', line)
        cursor.insertText(clean, formats[_line_tag(clean)])
    widget.setTextCursor(cursor)
    widget.ensureCursorVisible()


class ManagedCommand(QObject):
    """Launches and stops a GUI-driven startup command through `ProcessManager`.

    Delegating to `ProcessManager` (instead of a bare `subprocess.Popen`) means Stop
    gets the same guarantees the CLI launchers get: a SIGTERM/killpg to the whole
    process group, a bounded wait, and an escalation to SIGKILL if the process (or
    something it spawned) is still alive afterwards - so the launched `uv run ...`
    tree can't outlive the button press.
    """

    output_ready = pyqtSignal(str)
    done = pyqtSignal(object)

    def __init__(self, output: OutputCallback, done: DoneCallback, name: str = 'startup_gui'):
        super().__init__()
        self.output_ready.connect(output)
        self.done.connect(done)
        self._manager = ProcessManager(name=name)
        self._manager.cleanup_stale_state()
        self._managed: ManagedProcess | None = None

    @property
    def running(self) -> bool:
        return self._managed is not None and self._managed.running

    def start(self, command: list[str]) -> None:
        if self.running:
            self.output_ready.emit('A process is already running.\n')
            return
        env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
        self.output_ready.emit(f'$ {" ".join(command)}\n')
        self._managed = self._manager.start_process(
            key='gui_process',
            label=' '.join(command),
            command=command,
            env=env,
            stderr=subprocess.STDOUT,
            on_output=self.output_ready.emit,
        )
        threading.Thread(target=self._await_exit, args=(self._managed,), daemon=True).start()

    def stop(self) -> None:
        if not self.running:
            self.output_ready.emit('No process is running.\n')
            return
        self.output_ready.emit('Stop requested.\n')
        threading.Thread(target=self._manager.stop_process, args=(self._managed,), daemon=True).start()

    def shutdown(self) -> None:
        """Force-stop the managed process; call this from the window's close handler."""
        self._manager.shutdown_all()

    def _await_exit(self, managed: ManagedProcess) -> None:
        returncode = managed.process.wait()
        self.done.emit(returncode)


def _format(color: str) -> QTextCharFormat:
    text_format = QTextCharFormat()
    text_format.setForeground(QColor(color))
    return text_format


def _line_tag(clean: str) -> str:
    upper = clean.upper()
    if clean.startswith('$ '):
        return 'command'
    if 'FAIL' in upper or 'ERROR' in upper or 'TRACEBACK' in upper or 'FAILED' in upper:
        return 'fail'
    if ' OK ' in upper or upper.strip().startswith('OK') or ' READY ' in upper:
        return 'ok'
    if 'RUN' in upper:
        return 'run'
    if 'WAIT' in upper:
        return 'wait'
    if 'SKIP' in upper:
        return 'skip'
    return 'plain'
