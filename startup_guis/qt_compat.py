from __future__ import annotations

import os

# Default to PyQt6 for modern systems. Set ASYNCROSCOPY_QT_API=pyqt5 on older
# Windows 10 machines where Qt6 cannot load because the OS build is too old.
QT_API_ENV = os.environ.get('ASYNCROSCOPY_QT_API', '').lower()

try:
    if QT_API_ENV == 'pyqt5':
        raise ImportError('PyQt5 requested by ASYNCROSCOPY_QT_API')
    from PyQt6.QtCore import QObject as QObject, QPointF as QPointF, Qt as Qt, pyqtSignal as pyqtSignal
    from PyQt6.QtGui import (
        QColor as QColor,
        QFont as QFont,
        QPainter as QPainter,
        QPainterPath as QPainterPath,
        QPalette as QPalette,
        QPen as QPen,
        QTextCharFormat as QTextCharFormat,
        QTextCursor as QTextCursor,
    )
    from PyQt6.QtWidgets import (
        QApplication as QApplication,
        QCheckBox as QCheckBox,
        QComboBox as QComboBox,
        QFileDialog as QFileDialog,
        QFormLayout as QFormLayout,
        QFrame as QFrame,
        QGridLayout as QGridLayout,
        QGroupBox as QGroupBox,
        QHBoxLayout as QHBoxLayout,
        QLabel as QLabel,
        QLineEdit as QLineEdit,
        QMainWindow as QMainWindow,
        QPushButton as QPushButton,
        QScrollArea as QScrollArea,
        QSizePolicy as QSizePolicy,
        QSplitter as QSplitter,
        QStyle as QStyle,
        QStyleOptionButton as QStyleOptionButton,
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
    FONT_MEDIUM = QFont.Weight.Medium
    MONOSPACE_HINT = QFont.StyleHint.Monospace
    NO_FRAME = QFrame.Shape.NoFrame
    SCROLLBAR_AS_NEEDED = Qt.ScrollBarPolicy.ScrollBarAsNeeded
    SCROLLBAR_OFF = Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    FIELDS_STAY_AT_SIZE_HINT = QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
    SE_CHECKBOX_INDICATOR = QStyle.SubElement.SE_CheckBoxIndicator
    PEN_CAP_ROUND = Qt.PenCapStyle.RoundCap
    PEN_JOIN_ROUND = Qt.PenJoinStyle.RoundJoin
    RENDER_HINT_ANTIALIASING = QPainter.RenderHint.Antialiasing
    PALETTE_WINDOW_ROLE = QPalette.ColorRole.Window

    def app_exec(app: QApplication) -> int:
        return app.exec()

    def window_palette_color(app: QApplication) -> QColor:
        return app.palette().color(PALETTE_WINDOW_ROLE)

except ImportError:
    from PyQt5.QtCore import QObject as QObject, QPointF as QPointF, Qt as Qt, pyqtSignal as pyqtSignal
    from PyQt5.QtGui import (
        QColor as QColor,
        QFont as QFont,
        QPainter as QPainter,
        QPainterPath as QPainterPath,
        QPalette as QPalette,
        QPen as QPen,
        QTextCharFormat as QTextCharFormat,
        QTextCursor as QTextCursor,
    )
    from PyQt5.QtWidgets import (
        QApplication as QApplication,
        QCheckBox as QCheckBox,
        QComboBox as QComboBox,
        QFileDialog as QFileDialog,
        QFormLayout as QFormLayout,
        QFrame as QFrame,
        QGridLayout as QGridLayout,
        QGroupBox as QGroupBox,
        QHBoxLayout as QHBoxLayout,
        QLabel as QLabel,
        QLineEdit as QLineEdit,
        QMainWindow as QMainWindow,
        QPushButton as QPushButton,
        QScrollArea as QScrollArea,
        QSizePolicy as QSizePolicy,
        QSplitter as QSplitter,
        QStyle as QStyle,
        QStyleOptionButton as QStyleOptionButton,
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
    FONT_MEDIUM = QFont.Medium
    MONOSPACE_HINT = QFont.Monospace
    NO_FRAME = QFrame.NoFrame
    SCROLLBAR_AS_NEEDED = Qt.ScrollBarAsNeeded
    SCROLLBAR_OFF = Qt.ScrollBarAlwaysOff
    FIELDS_STAY_AT_SIZE_HINT = QFormLayout.AllNonFixedFieldsGrow
    SE_CHECKBOX_INDICATOR = QStyle.SE_CheckBoxIndicator
    PEN_CAP_ROUND = Qt.RoundCap
    PEN_JOIN_ROUND = Qt.RoundJoin
    RENDER_HINT_ANTIALIASING = QPainter.Antialiasing
    PALETTE_WINDOW_ROLE = QPalette.Window

    def app_exec(app: QApplication) -> int:
        return app.exec_()

    def window_palette_color(app: QApplication) -> QColor:
        return app.palette().color(PALETTE_WINDOW_ROLE)
