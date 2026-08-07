from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

import yaml

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


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / 'configs'
GENERATED_CONFIG_DIR = PROJECT_DIR / 'outputs' / 'startup_configs'

# Two palettes shared by every startup GUI, so the windows read as one app and
# follow whichever appearance (light/dark) the OS is currently set to.
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
    # The terminal keeps its console-style dark background in both themes, so
    # output stays readable regardless of the surrounding chrome.
    'terminal_bg': '#0b0f14',
    'terminal_text': '#c9d1d9',
}
# Mutated in place by apply_theme() once the system appearance is known, so
# every helper below (which reads COLORS at call time) picks up the right set.
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
    # An empty family string leaves Qt on the platform's own UI font (SF on
    # macOS, Segoe UI on Windows), which is what makes the window look native.
    font = QFont()
    font.setPointSize(size)
    if weight is not None:
        font.setWeight(weight)
    return font


def _font(name: str) -> QFont:
    # Fonts are built lazily (rather than as module-level constants) because
    # QFont requires a QApplication to already exist; callers only need them
    # once the app is up.
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


def app_stylesheet() -> str:
    """The dark, flat style shared by the startup windows."""
    c = COLORS
    return f'''
    QMainWindow, QDialog, QWidget#appRoot {{
        background: {c['window']};
    }}
    /* Panes stay transparent so only the window and the section cards paint a
       background; otherwise nested containers punch dark holes in the cards. */
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
    /* The drop-down button is left native: overriding it costs the arrow, and a
       combo without an arrow is indistinguishable from a line edit. */
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
    /* Qt ships no checkmark image we can reference from a stylesheet, so
       checked state is a solid accent fill; the unchecked border is kept bright
       enough to stay visible against the dark card. */
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
    /* The checkmark glyph itself is painted by CheckBox.paintEvent (Qt style
       sheets can't reference an inline image), so the indicator only supplies
       the filled square here. */
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
    """Apply the shared palette and base font to a top-level window.

    The palette is picked from the OS appearance (light/dark) at call time,
    so every startup GUI automatically matches the system theme.
    """
    COLORS.clear()
    COLORS.update(DARK_COLORS if system_is_dark() else LIGHT_COLORS)
    window.setFont(_font('BODY_FONT'))
    window.setStyleSheet(app_stylesheet())


class CheckBox(QCheckBox):
    """A QCheckBox that paints a checkmark glyph over the filled indicator.

    Qt's stylesheet engine can't reference an inline image for
    QCheckBox::indicator, so a plain checked box only shows as a solid color
    square; this subclass draws the checkmark on top with QPainter instead.
    """

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


# The two Tango hosts this project's startup GUIs are actually pointed at day
# to day: the local digital-twin stack, and the real Spectra 300 instrument.
DIGITAL_TWIN_HOST = 'localhost'
SPECTRA300_HOST = '10.46.217.241'
HOST_PRESETS = {'digital_twin': DIGITAL_TWIN_HOST, 'spectra300': SPECTRA300_HOST}


class HostToggle(QWidget):
    """A Digital Twin / Spectra300 segmented switch that sets host field(s) to a known-good preset.

    This only covers the Tango (and optionally Tiled) *host* value(s) - the
    recurring pain point where a GUI is left pointed at the wrong machine.
    It does not switch instrument file/device classes; that's config-driven
    (load DigitalTwin.yaml / Spectra300.yaml) since those differ per instrument,
    not just per host.
    """

    def __init__(self, on_change: Callable[[str], None], parent=None):
        super().__init__(parent)
        self._on_change = on_change
        self._buttons: dict[str, QPushButton] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for key, label in (('digital_twin', 'Digital Twin'), ('spectra300', 'Spectra300')):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setFont(_font('LABEL_FONT'))
            button.setCursor(POINTING_HAND_CURSOR)
            button.clicked.connect(lambda _checked, k=key: self._select(k, notify=True))
            layout.addWidget(button)
            self._buttons[key] = button
        self.set_active(None)

    def _select(self, key: str, notify: bool) -> None:
        for button_key, button in self._buttons.items():
            button.setChecked(button_key == key)
        self._restyle()
        if notify:
            self._on_change(HOST_PRESETS[key])

    def set_active(self, key: str | None) -> None:
        """Reflect which preset (if any) a host field currently matches, without firing on_change."""
        self._select(key, notify=False)

    def _restyle(self) -> None:
        for key, button in self._buttons.items():
            checked = button.isChecked()
            first = key == 'digital_twin'
            radius = '6px 0px 0px 6px' if first else '0px 6px 6px 0px'
            border_fix = '' if first else 'border-left: none;'
            if checked:
                button.setStyleSheet(
                    'QPushButton {'
                    f'background: {COLORS["accent"]}; color: #ffffff; border: 1px solid {COLORS["accent"]};'
                    f'{border_fix} border-radius: {radius}; padding: 4px 12px;'
                    '}'
                )
            else:
                button.setStyleSheet(
                    'QPushButton {'
                    f'background: {COLORS["panel"]}; color: {COLORS["text_dim"]}; border: 1px solid {COLORS["border"]};'
                    f'{border_fix} border-radius: {radius}; padding: 4px 12px;'
                    '}'
                    'QPushButton:hover {'
                    f'background: {COLORS["panel_hover"]};'
                    '}'
                )


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
    # As-needed in both directions: a narrow window scrolls rather than clipping
    # the labels of whichever section happens to be open.
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
    output_ready = pyqtSignal(str)
    done = pyqtSignal(object)

    def __init__(self, output: OutputCallback, done: DoneCallback):
        super().__init__()
        self.output_ready.connect(output)
        self.done.connect(done)
        self.process: subprocess.Popen[str] | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, command: list[str]) -> None:
        if self.running:
            self.output_ready.emit('A process is already running.\n')
            return
        env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
        popen_kwargs = {'cwd': PROJECT_DIR, 'env': env, 'stdout': subprocess.PIPE, 'stderr': subprocess.STDOUT, 'text': True, 'bufsize': 1}
        if os.name == 'nt':
            popen_kwargs['creationflags'] = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
        else:
            popen_kwargs['start_new_session'] = True
        self.output_ready.emit(f'$ {" ".join(command)}\n')
        self.process = subprocess.Popen(command, **popen_kwargs)
        threading.Thread(target=self._read_output, daemon=True).start()

    def stop(self) -> None:
        if not self.running:
            self.output_ready.emit('No process is running.\n')
            return
        assert self.process is not None
        if os.name == 'nt':
            self.process.terminate()
        else:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            except OSError:
                self.process.terminate()
        self.output_ready.emit('Stop requested.\n')

    def _read_output(self) -> None:
        assert self.process is not None
        if self.process.stdout is not None:
            while True:
                line = self.process.stdout.readline()
                if line:
                    self.output_ready.emit(line)
                    continue
                if self.process.poll() is not None:
                    rest = self.process.stdout.read()
                    if rest:
                        self.output_ready.emit(rest)
                    break
                threading.Event().wait(0.05)
        self.done.emit(self.process.wait())


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
