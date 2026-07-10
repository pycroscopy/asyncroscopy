from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
from pathlib import Path
from typing import Callable

import yaml

from startup_guis.qt_compat import (
    FONT_BOLD,
    MOVE_END,
    POINTING_HAND_CURSOR,
    QColor,
    QFont,
    QObject,
    QPushButton,
    QTextCharFormat,
    QTextEdit,
    pyqtSignal,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / 'configs'
GENERATED_CONFIG_DIR = PROJECT_DIR / 'outputs' / 'startup_configs'
BODY_FONT = QFont('Arial', 15)
TITLE_FONT = QFont('Arial', 24, FONT_BOLD)
SECTION_FONT = QFont('Arial', 18, FONT_BOLD)
TEXT_FONT = QFont('Menlo', 16)
ACTION_FONT = QFont('Arial', 18, FONT_BOLD)

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


def action_button(text: str, color: str, active_color: str) -> QPushButton:
    button = QPushButton(text)
    button.setFont(ACTION_FONT)
    button.setCursor(POINTING_HAND_CURSOR)
    button.setStyleSheet(
        'QPushButton {'
        f'background: {color}; color: white; border: 2px solid #222; padding: 12px 18px;'
        '}'
        'QPushButton:hover, QPushButton:pressed {'
        f'background: {active_color};'
        '}'
    )
    return button


def configure_terminal(widget: QTextEdit) -> None:
    widget.setFont(TEXT_FONT)
    widget.setReadOnly(True)
    widget.setStyleSheet('background: #0d1117; color: #c9d1d9;')


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
