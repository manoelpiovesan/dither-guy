from __future__ import annotations

# ── Palette (Photoshop CC Dark Theme) ──────────────────────────────────────────
_P0 = "#2d2d2d" # Darkest (Canvas bg)
_P1 = "#323232" # Main Background
_P2 = "#3a3a3a" # Slightly lighter (Panel bg)
_P3 = "#424242" # Hover bg
_P4 = "#4d4d4d" # Button bg
_P5 = "#252525" # Borders
_P6 = "#646464" # Lighter border
_P7 = "#888888" 

_G0 = "#2a8ceb" # PS Blue Accent
_G1 = "#1f75cb"
_G2 = "#145aa8"
_G3 = "#1a3d66" # Subdued blue for selections

_AE  = "#ff8c00"
_AE2 = "#3d2200"

_AM = "#f0a500"
_RE = "#e84040"

_FG  = "#d6d6d6" # Main text
_FG2 = "#a8a8a8" # Secondary text
_FG3 = "#808080" # Disabled text
_FG4 = "#303030"

_MONO_FONT = "'Consolas', 'JetBrains Mono', 'Cascadia Code', 'Courier New', monospace"
_SANS_FONT = "'Segoe UI', 'Helvetica Neue', 'Arial', sans-serif"

# ── Stylesheet ────────────────────────────────────────────────────────────────
_THEME = f"""
* {{ outline: none; }}

/* ── Global ──────────────────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {_P1};
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 11px;
}}

/* ── ToolBar (Top Nav Bar) ───────────────────────────────────────────────── */
QToolBar {{
    background: {_P1};
    border: none;
    border-bottom: 1px solid {_P5};
    padding: 2px 4px;
    spacing: 2px;
}}
QToolBar QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 2px;
    padding: 4px 8px;
    color: {_FG2};
    font-family: {_SANS_FONT};
    font-size: 11px;
    text-align: center;
}}
QToolBar QToolButton:hover {{
    color: {_FG};
    background: {_P3};
    border-color: {_P3};
}}
QToolBar QToolButton:pressed {{
    background: {_P0};
    color: {_FG};
}}
QToolBar QToolButton:checked {{
    background: {_P0};
    color: {_G0};
    border: 1px solid {_P5};
}}
QToolBar QLabel {{
    color: {_FG};
    font-weight: 600;
    font-size: 11px;
    padding: 4px;
}}
QToolBar::separator {{
    background: {_P5};
    height: 1px;
    margin: 4px 6px;
}}

/* ── Menu Bar ────────────────────────────────────────────────────────────── */
QMenuBar {{
    background: {_P1};
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
    border-bottom: 1px solid {_P5};
    padding: 2px 4px;
}}
QMenuBar::item {{
    padding: 4px 10px;
    background: transparent;
}}
QMenuBar::item:selected {{
    background: {_G1};
    color: #ffffff;
}}
QMenu {{
    background: {_P2};
    border: 1px solid {_P5};
    padding: 2px 0;
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
}}
QMenu::item {{
    padding: 4px 28px 4px 18px;
}}
QMenu::item:selected {{
    background: {_G1};
    color: #ffffff;
}}
QMenu::separator {{
    height: 1px;
    background: {_P5};
    margin: 4px 0;
}}

/* ── Tabs (Canvas Documents) ─────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {_P5};
    background: {_P0};
}}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: {_P2};
    border: 1px solid {_P5};
    border-bottom: none;
    padding: 6px 14px;
    font-family: {_SANS_FONT};
    font-size: 11px;
    color: {_FG2};
    margin-right: 1px;
    min-width: 60px;
}}
QTabBar::tab:selected {{
    background: {_P0};
    color: {_FG};
    border-top: 2px solid {_G0};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    color: {_FG};
    background: {_P3};
}}

/* ── GroupBox (Properties Panels) ────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {_P5};
    border-radius: 2px;
    margin-top: 18px;
    padding-top: 14px;
    font-family: {_SANS_FONT};
    font-size: 10px;
    font-weight: 600;
    color: {_FG3};
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    font-size: 10px;
    left: 8px;
    top: 0px;
    padding: 0 4px;
    color: {_FG2};
    background: transparent;
    text-transform: uppercase;
}}

/* ── Push Buttons ────────────────────────────────────────────────────────── */
QPushButton {{
    background: {_P4};
    border: 1px solid {_P5};
    border-radius: 3px;
    padding: 4px 12px;
    font-family: {_SANS_FONT};
    font-size: 11px;
    font-weight: 600;
    color: {_FG};
    min-height: 22px;
}}
QPushButton:hover {{
    background: {_P6};
    border-color: {_P5};
}}
QPushButton:pressed {{
    background: {_P2};
    color: {_G0};
    border-color: {_G0};
}}
QPushButton:disabled {{
    background: {_P2};
    color: {_FG3};
    border-color: {_P5};
}}
QPushButton#accent {{
    background: {_G1};
    border: 1px solid {_G2};
    color: #ffffff;
}}
QPushButton#accent:hover {{
    background: {_G0};
    border-color: {_G1};
}}
QPushButton#accent:pressed {{
    background: {_G2};
}}

/* ── ComboBox ────────────────────────────────────────────────────────────── */
QComboBox {{
    background: {_P4};
    border: 1px solid {_P5};
    border-radius: 3px;
    padding: 2px 6px;
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 11px;
    min-height: 20px;
}}
QComboBox:hover {{ border-color: {_P6}; background: {_P6}; }}
QComboBox:focus {{ border-color: {_G0}; }}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {_FG2};
}}
QComboBox QAbstractItemView {{
    background: {_P2};
    border: 1px solid {_P5};
    selection-background-color: {_G1};
    selection-color: #ffffff;
    font-size: 11px;
}}

/* ── Slider (Photoshop Style) ────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 3px;
    background: {_P0};
    border: 1px solid {_P5};
    border-radius: 1px;
    margin: 0 4px;
}}
QSlider::sub-page:horizontal {{
    background: {_G0};
    border-radius: 1px;
}}
QSlider::handle:horizontal {{
    background: {_FG};
    width: 10px;
    height: 14px;
    margin: -6px 0;
    border-radius: 2px;
    border: 1px solid {_P5};
}}
QSlider::handle:horizontal:hover {{
    background: #ffffff;
}}
QSlider::handle:horizontal:pressed {{
    background: {_G0};
}}

/* ── ScrollBar ───────────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {_P1};
    width: 12px;
    border-left: 1px solid {_P5};
    margin: 0;
}}
QScrollBar:horizontal {{
    background: {_P1};
    height: 12px;
    border-top: 1px solid {_P5};
    margin: 0;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {_P4};
    border-radius: 6px;
    min-height: 20px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover {{ background: {_P6}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}

/* ── Splitter ────────────────────────────────────────────────────────────── */
QSplitter::handle {{
    background: {_P5};
}}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}
QSplitter::handle:hover {{ background: {_P6}; }}

/* ── SpinBox / LineEdit ──────────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {_P0};
    border: 1px solid {_P5};
    border-radius: 2px;
    padding: 3px 5px;
    color: {_FG};
    font-size: 11px;
}}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {_G0};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 0px;
    border: none;
}}

/* ── StatusBar ───────────────────────────────────────────────────────────── */
QStatusBar {{
    background: {_P1};
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 11px;
    border-top: 1px solid {_P5};
    padding: 2px 6px;
}}

/* ── Dialog / ToolTip ────────────────────────────────────────────────────── */
QDialog {{ background: {_P1}; }}
QToolTip {{
    background: #ffffcc;
    color: #000000;
    border: 1px solid #000000;
    font-size: 11px;
}}
"""

THEME = _THEME
