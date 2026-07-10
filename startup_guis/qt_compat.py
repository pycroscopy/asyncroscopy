from __future__ import annotations

import os

# Default to PyQt6 for modern systems. Set ASYNCROSCOPY_QT_API=pyqt5 on older
# Windows 10 machines where Qt6 cannot load because the OS build is too old.
QT_API_ENV = os.environ.get('ASYNCROSCOPY_QT_API', '').lower()

try:
    if QT_API_ENV == 'pyqt5':
        raise ImportError('PyQt5 requested by ASYNCROSCOPY_QT_API')
    from PyQt6.QtCore import QObject as QObject, Qt as Qt, pyqtSignal as pyqtSignal
    from PyQt6.QtGui import (
        QColor as QColor,
        QFont as QFont,
        QTextCharFormat as QTextCharFormat,
        QTextCursor as QTextCursor,
    )
    from PyQt6.QtWidgets import (
        QApplication as QApplication,
        QCheckBox as QCheckBox,
        QComboBox as QComboBox,
        QFileDialog as QFileDialog,
        QFormLayout as QFormLayout,
        QGridLayout as QGridLayout,
        QGroupBox as QGroupBox,
        QHBoxLayout as QHBoxLayout,
        QLabel as QLabel,
        QLineEdit as QLineEdit,
        QMainWindow as QMainWindow,
        QPushButton as QPushButton,
        QSplitter as QSplitter,
        QTextEdit as QTextEdit,
        QVBoxLayout as QVBoxLayout,
        QWidget as QWidget,
    )

    QT_API = 'PyQt6'
    HORIZONTAL = Qt.Orientation.Horizontal
    VERTICAL = Qt.Orientation.Vertical
    POINTING_HAND_CURSOR = Qt.CursorShape.PointingHandCursor
    MOVE_END = QTextCursor.MoveOperation.End
    NO_WRAP = QTextEdit.LineWrapMode.NoWrap
    FONT_BOLD = QFont.Weight.Bold

    def app_exec(app: QApplication) -> int:
        return app.exec()

except ImportError:
    from PyQt5.QtCore import QObject as QObject, Qt as Qt, pyqtSignal as pyqtSignal
    from PyQt5.QtGui import (
        QColor as QColor,
        QFont as QFont,
        QTextCharFormat as QTextCharFormat,
        QTextCursor as QTextCursor,
    )
    from PyQt5.QtWidgets import (
        QApplication as QApplication,
        QCheckBox as QCheckBox,
        QComboBox as QComboBox,
        QFileDialog as QFileDialog,
        QFormLayout as QFormLayout,
        QGridLayout as QGridLayout,
        QGroupBox as QGroupBox,
        QHBoxLayout as QHBoxLayout,
        QLabel as QLabel,
        QLineEdit as QLineEdit,
        QMainWindow as QMainWindow,
        QPushButton as QPushButton,
        QSplitter as QSplitter,
        QTextEdit as QTextEdit,
        QVBoxLayout as QVBoxLayout,
        QWidget as QWidget,
    )

    QT_API = 'PyQt5'
    HORIZONTAL = Qt.Horizontal
    VERTICAL = Qt.Vertical
    POINTING_HAND_CURSOR = Qt.PointingHandCursor
    MOVE_END = QTextCursor.End
    NO_WRAP = QTextEdit.NoWrap
    FONT_BOLD = QFont.Bold

    def app_exec(app: QApplication) -> int:
        return app.exec_()
