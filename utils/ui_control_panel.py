from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QPushButton, QLabel, QSlider, QScrollArea, QLineEdit,
    QGridLayout, QColorDialog, QMessageBox,
)

from .constants import METHOD_GROUPS, METHODS, VERSION
from .palettes import PALETTES
from .presets import save_preset, load_preset, list_presets, delete_preset
from .theme import _P0, _P1, _P2, _P3, _P4, _P5, _P6, _G0, _G1, _G2, _G3, _AM, _FG, _FG2, _FG3, _MONO_FONT
from .ui_widgets import hsep, make_slider

try:
    from .dither_kernels import _NUMBA
except ImportError:
    _NUMBA = False


# ---------------------------------------------------------------------------
# Method picker
# ---------------------------------------------------------------------------

class MethodPicker(QWidget):
    method_selected = Signal(str)

    def __init__(self):
        super().__init__()
        self._current = "Floyd-Steinberg"
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(6)
        self._combo = QComboBox(); self._combo.setMinimumHeight(30)
        for group, members in METHOD_GROUPS.items():
            self._combo.addItem(f"── {group} ──")
            idx  = self._combo.count() - 1
            item = self._combo.model().item(idx)
            item.setEnabled(False)
            item.setForeground(QColor(_AM))
            for m in members:
                self._combo.addItem(f"  {m}")
        for i in range(self._combo.count()):
            if self._combo.itemText(i).strip() == "Floyd-Steinberg":
                self._combo.setCurrentIndex(i); break
        self._combo.currentIndexChanged.connect(self._on_changed)
        layout.addWidget(self._combo)

    def _on_changed(self, idx):
        item = self._combo.model().item(idx)
        if item and not item.isEnabled():
            self._combo.setCurrentIndex(idx + 1); return
        name = self._combo.currentText().strip()
        if name in METHODS:
            self._current = name
            self.method_selected.emit(name)

    def current_method(self) -> str:
        return self._current

    def set_method(self, name: str):
        for i in range(self._combo.count()):
            if self._combo.itemText(i).strip() == name:
                self._combo.blockSignals(True)
                self._combo.setCurrentIndex(i)
                self._combo.blockSignals(False)
                self._current = name; break


# ---------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------

class ControlPanel(QWidget):
    params_changed         = Signal()
    params_changed_preview = Signal()

    def __init__(self):
        super().__init__()
        self.current_color = (255, 255, 255)
        self._dragging_slider = False
        self._custom_palette: list[tuple] = []
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        inner  = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8); layout.setSpacing(10)
        scroll.setWidget(inner); outer.addWidget(scroll, stretch=1)

        # Algorithm
        ag = QGroupBox("Algorithm"); al = QVBoxLayout(ag); al.setSpacing(6)
        self.method_picker = MethodPicker()
        self.method_picker.method_selected.connect(lambda _: self.params_changed.emit())
        al.addWidget(self.method_picker); layout.addWidget(ag)

        # Dither
        dg = QGroupBox("Dither"); dl = QVBoxLayout(dg); dl.setSpacing(4)
        _, self._pix_val, self.pixel_sl  = make_slider(dl, "pixel size", 1, 20, 4)
        _, self._thr_val, self.thresh_sl = make_slider(dl, "threshold",  0, 255, 128)
        layout.addWidget(dg)

        # Adjustments
        adj = QGroupBox("Adjustments"); al2 = QVBoxLayout(adj); al2.setSpacing(4)
        _, self._br_val, self.bright_sl = make_slider(al2, "brightness", 0, 200, 100, "{v}%")
        _, self._co_val, self.contr_sl  = make_slider(al2, "contrast",   0, 200, 100, "{v}%")
        _, self._bl_val, self.blur_sl   = make_slider(al2, "blur",       0,  10,   0)
        _, self._sh_val, self.sharp_sl  = make_slider(al2, "sharpen",    0,   5,   0)
        layout.addWidget(adj)

        reset_btn = QPushButton("↺  Reset Adjustments")
        reset_btn.clicked.connect(self._reset); reset_btn.setMinimumHeight(26)
        layout.addWidget(reset_btn); layout.addWidget(hsep())

        # Glow
        gg = QGroupBox("Glow"); gl = QVBoxLayout(gg); gl.setSpacing(4)
        _, self._gr_val, self.glow_r_sl = make_slider(gl, "radius",    0,  40,  0)
        _, self._gi_val, self.glow_i_sl = make_slider(gl, "intensity", 0, 100,  0, "{v}%")
        layout.addWidget(gg); layout.addWidget(hsep())

        # Colour
        cg = QGroupBox("Colour"); cl = QVBoxLayout(cg); cl.setSpacing(6)
        self.swatch = QLabel(); self.swatch.setFixedHeight(34)
        self.swatch.setAlignment(Qt.AlignCenter); self._refresh_swatch()
        cl.addWidget(self.swatch)
        pick_btn = QPushButton("⬛  Pick Colour"); pick_btn.setMinimumHeight(26)
        pick_btn.clicked.connect(self._pick_color); cl.addWidget(pick_btn)
        layout.addWidget(cg)

        # Palette
        palg = QGroupBox("Palette"); pall = QVBoxLayout(palg); pall.setSpacing(6)
        self.palette_combo = QComboBox(); self.palette_combo.setMinimumHeight(28)
        for pal_name in PALETTES:
            self.palette_combo.addItem(pal_name)
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        pall.addWidget(self.palette_combo)

        self.pal_swatch_widget = QWidget()
        self.pal_swatch_layout = QGridLayout(self.pal_swatch_widget)
        self.pal_swatch_layout.setSpacing(2); self.pal_swatch_layout.setContentsMargins(0, 0, 0, 0)
        pall.addWidget(self.pal_swatch_widget)
        self._refresh_palette_swatches()

        custom_row = QHBoxLayout()
        self.custom_pal_btn   = QPushButton("+ Add Color");   self.custom_pal_btn.setMinimumHeight(24)
        self.clear_custom_btn = QPushButton("✕ Clear");       self.clear_custom_btn.setMinimumHeight(24)
        self.custom_pal_btn.clicked.connect(self._add_custom_color)
        self.clear_custom_btn.clicked.connect(self._clear_custom_palette)
        custom_row.addWidget(self.custom_pal_btn); custom_row.addWidget(self.clear_custom_btn)
        pall.addLayout(custom_row)
        layout.addWidget(palg)

        # Presets
        pg = QGroupBox("Presets"); pl = QVBoxLayout(pg); pl.setSpacing(6)
        row_p = QHBoxLayout()
        self.preset_name = QLineEdit(); self.preset_name.setPlaceholderText("preset name...")
        self.preset_name.setMinimumHeight(26); row_p.addWidget(self.preset_name)
        save_p = QPushButton("Save"); save_p.setMinimumHeight(26)
        save_p.clicked.connect(self._save_preset); row_p.addWidget(save_p)
        pl.addLayout(row_p)
        self.preset_combo = QComboBox(); self.preset_combo.setMinimumHeight(26)
        self._refresh_preset_combo()
        row_p2 = QHBoxLayout()
        load_p = QPushButton("Load"); load_p.setMinimumHeight(26)
        del_p  = QPushButton("Del");  del_p.setMinimumHeight(26); del_p.setObjectName("danger")
        load_p.clicked.connect(self._load_preset)
        del_p.clicked.connect(self._delete_preset)
        row_p2.addWidget(self.preset_combo); row_p2.addWidget(load_p); row_p2.addWidget(del_p)
        pl.addLayout(row_p2); layout.addWidget(pg)

        layout.addStretch()
        if _NUMBA:
            jit_lbl = QLabel("numba JIT active"); jit_lbl.setAlignment(Qt.AlignCenter)
            jit_lbl.setStyleSheet(
                f"font-family:{_MONO_FONT}; font-size:9px; color:{_FG}; padding:0 0 6px 0;")
            layout.addWidget(jit_lbl)
        ver = QLabel(f"DITHER GUY  v{VERSION}"); ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:9px; color:{_G3};"
            "letter-spacing:2px; padding:10px 0 4px 0;")
        layout.addWidget(ver)

        # Wire sliders
        # On Windows, emitting params_changed_preview on every valueChanged
        # tick while the mouse button is held floods the worker queue and
        # causes stale finished signals to snap the canvas back to an earlier
        # state once the drag ends (issue #8).  We therefore only emit the
        # preview signal when the user is NOT actively dragging — the label
        # still updates every tick for responsive number feedback, and a full
        # render is triggered on sliderReleased via params_changed.
        def _connect(sl, lbl, fmt):
            sl.sliderPressed.connect(lambda: setattr(self, '_dragging_slider', True))
            sl.sliderReleased.connect(self._on_slider_released)
            sl.valueChanged.connect(lambda v, l=lbl, f=fmt: l.setText(f.format(v=v)))
            sl.valueChanged.connect(self._on_slider_value_changed)

        _connect(self.pixel_sl,   self._pix_val, "{v}")
        _connect(self.thresh_sl,  self._thr_val, "{v}")
        _connect(self.bright_sl,  self._br_val,  "{v}%")
        _connect(self.contr_sl,   self._co_val,  "{v}%")
        _connect(self.blur_sl,    self._bl_val,  "{v}")
        _connect(self.sharp_sl,   self._sh_val,  "{v}")
        _connect(self.glow_r_sl,  self._gr_val,  "{v}")
        _connect(self.glow_i_sl,  self._gi_val,  "{v}%")

    def _on_slider_value_changed(self):
        # Only fire a live preview when the value changes via keyboard or
        # programmatic setValue — not while the mouse button is held down.
        # While dragging, the label already updated above; the full render
        # will fire on sliderReleased -> params_changed.
        if not self._dragging_slider:
            self.params_changed_preview.emit()

    def _on_slider_released(self):
        self._dragging_slider = False
        self.params_changed.emit()

    def _reset(self):
        for sl, v in [(self.bright_sl, 100), (self.contr_sl, 100), (self.blur_sl, 0),
                      (self.sharp_sl, 0), (self.glow_r_sl, 0), (self.glow_i_sl, 0)]:
            sl.blockSignals(True); sl.setValue(v); sl.blockSignals(False)
        self._refresh_value_labels()
        self.params_changed.emit()

    def _refresh_value_labels(self):
        self._br_val.setText(f"{self.bright_sl.value()}%")
        self._co_val.setText(f"{self.contr_sl.value()}%")
        self._bl_val.setText(str(self.blur_sl.value()))
        self._sh_val.setText(str(self.sharp_sl.value()))
        self._gr_val.setText(str(self.glow_r_sl.value()))
        self._gi_val.setText(f"{self.glow_i_sl.value()}%")

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(*self.current_color), self, "Pick Colour")
        if c.isValid():
            self.current_color = (c.red(), c.green(), c.blue())
            self._refresh_swatch(); self.params_changed.emit()

    def _refresh_swatch(self):
        r, g, b = self.current_color
        lum = (r * 299 + g * 587 + b * 114) // 1000
        txt = "#000" if lum > 128 else "#fff"
        self.swatch.setStyleSheet(
            f"background:rgb({r},{g},{b}); border:1px solid {_P6};"
            f"border-radius:3px; color:{txt};"
            f"font-family:{_MONO_FONT}; font-size:10px; font-weight:bold;")
        self.swatch.setText(f"#{r:02X}{g:02X}{b:02X}")

    def _on_palette_changed(self, name: str):
        self._refresh_palette_swatches()
        self.params_changed.emit()

    def _refresh_palette_swatches(self):
        while self.pal_swatch_layout.count():
            item = self.pal_swatch_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        name   = self.palette_combo.currentText()
        colors = self._custom_palette if name == "Custom" else PALETTES.get(name, [])
        cols   = 8
        for i, (r, g, b) in enumerate(colors):
            sq = QLabel(); sq.setFixedSize(16, 16)
            sq.setToolTip(f"#{r:02X}{g:02X}{b:02X}")
            sq.setStyleSheet(
                f"background:rgb({r},{g},{b}); border:1px solid {_P4}; border-radius:1px;")
            self.pal_swatch_layout.addWidget(sq, i // cols, i % cols)

    def _add_custom_color(self):
        c = QColorDialog.getColor(parent=self, title="Add Palette Color")
        if c.isValid():
            self._custom_palette.append((c.red(), c.green(), c.blue()))
            idx = self.palette_combo.findText("Custom")
            if idx < 0:
                self.palette_combo.addItem("Custom"); idx = self.palette_combo.count() - 1
            self.palette_combo.blockSignals(True)
            self.palette_combo.setCurrentIndex(idx)
            self.palette_combo.blockSignals(False)
            self._refresh_palette_swatches(); self.params_changed.emit()

    def _clear_custom_palette(self):
        self._custom_palette.clear()
        self._refresh_palette_swatches(); self.params_changed.emit()

    def _refresh_preset_combo(self):
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for name in list_presets(): self.preset_combo.addItem(name)
        self.preset_combo.blockSignals(False)

    def _save_preset(self):
        name = self.preset_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Preset", "Enter a preset name."); return
        save_preset(name, self.get_params())
        self._refresh_preset_combo()
        idx = self.preset_combo.findText(name)
        if idx >= 0: self.preset_combo.setCurrentIndex(idx)
        self.preset_name.clear()

    def _load_preset(self):
        name = self.preset_combo.currentText()
        if not name: return
        p = load_preset(name)
        if p is None:
            QMessageBox.warning(self, "Preset", f"Could not load '{name}'."); return
        self.method_picker.set_method(p.get('method', 'Floyd-Steinberg'))
        self.pixel_sl.setValue(p.get('pixel_size', 4))
        self.thresh_sl.setValue(p.get('threshold', 128))
        self.bright_sl.setValue(int(p.get('brightness', 1.0) * 100))
        self.contr_sl.setValue(int(p.get('contrast', 1.0) * 100))
        self.blur_sl.setValue(p.get('blur', 0))
        self.sharp_sl.setValue(p.get('sharpen', 0))
        self.glow_r_sl.setValue(p.get('glow_radius', 0))
        self.glow_i_sl.setValue(p.get('glow_intensity', 0))
        self.current_color = tuple(p.get('color', (0, 255, 65))); self._refresh_swatch()
        pal_name = p.get('palette_name', 'B&W')
        idx = self.palette_combo.findText(pal_name)
        if idx >= 0: self.palette_combo.setCurrentIndex(idx)
        self.params_changed.emit()

    def _delete_preset(self):
        name = self.preset_combo.currentText()
        if not name: return
        if delete_preset(name): self._refresh_preset_combo()

    def get_params(self) -> dict:
        pal_name = self.palette_combo.currentText()
        return {
            "method":         self.method_picker.current_method(),
            "pixel_size":     self.pixel_sl.value(),
            "threshold":      self.thresh_sl.value(),
            "brightness":     self.bright_sl.value() / 100.0,
            "contrast":       self.contr_sl.value()  / 100.0,
            "blur":           self.blur_sl.value(),
            "sharpen":        self.sharp_sl.value(),
            "color":          self.current_color,
            "glow_radius":    self.glow_r_sl.value(),
            "glow_intensity": self.glow_i_sl.value(),
            "palette_name":   pal_name,
            "custom_palette": self._custom_palette if pal_name == "Custom" else None,
        }
