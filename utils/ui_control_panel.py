from __future__ import annotations

import random

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QPushButton, QLabel, QScrollArea, QLineEdit,
    QGridLayout, QColorDialog, QMessageBox,
)

from .constants import METHOD_GROUPS, METHODS, VERSION
from .palettes import PALETTES
from .presets import save_preset, load_preset, list_presets, delete_preset
from .theme import _P6, _P4, _P1, _P2, _AM, _FG, _FG2, _G0, _G3, _MONO_FONT, _AE
from .ui_widgets import hsep, make_slider

try:
    from .dither_kernels import _NUMBA
except ImportError:
    _NUMBA = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ClickableSwatch(QLabel):
    clicked = Signal()
    rightClicked = Signal()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.clicked.emit()
        elif ev.button() == Qt.RightButton:
            self.rightClicked.emit()
        super().mousePressEvent(ev)


# ---------------------------------------------------------------------------
# Method picker
# ---------------------------------------------------------------------------

class MethodPicker(QWidget):
    method_selected = Signal(str)

    _DEFAULT = "Bayer 2x2"

    def __init__(self):
        super().__init__()
        self._current = self._DEFAULT
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._combo = QComboBox()
        self._combo.setMinimumHeight(28)
        self._combo.setToolTip("Dithering algorithm")

        for group, members in METHOD_GROUPS.items():
            self._combo.addItem(f"\u2500\u2500 {group} \u2500\u2500")
            header = self._combo.model().item(self._combo.count() - 1)
            header.setEnabled(False)
            header.setForeground(QColor(_AM))
            for m in members:
                self._combo.addItem(f"  {m}")

        for i in range(self._combo.count()):
            if self._combo.itemText(i).strip() == self._DEFAULT:
                self._combo.setCurrentIndex(i)
                break

        self._combo.currentIndexChanged.connect(self._on_changed)
        layout.addWidget(self._combo)

    def _on_changed(self, idx: int) -> None:
        item = self._combo.model().item(idx)
        if item and not item.isEnabled():
            self._combo.setCurrentIndex(idx + 1)
            return
        name = self._combo.currentText().strip()
        if name in METHODS:
            self._current = name
            self.method_selected.emit(name)

    def current_method(self) -> str:
        return self._current

    def set_method(self, name: str) -> None:
        for i in range(self._combo.count()):
            if self._combo.itemText(i).strip() == name:
                self._combo.blockSignals(True)
                self._combo.setCurrentIndex(i)
                self._combo.blockSignals(False)
                self._current = name
                break


# ---------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------

class ControlPanel(QWidget):
    params_changed         = Signal()
    params_changed_preview = Signal()

    def __init__(self):
        super().__init__()
        self.current_color: tuple[int, int, int] = (255, 255, 255)
        self._custom_palette: list[tuple[int, int, int]] = []
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        inner  = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)

        ag = self._group("Algorithm")
        self.method_picker = MethodPicker()
        self.method_picker.method_selected.connect(lambda _: self.params_changed.emit())
        ag.layout().addWidget(self.method_picker)
        layout.addWidget(ag)

        dg = self._group("Dither")
        _, self._pix_val,  self.pixel_sl  = make_slider(
            dg.layout(), "Pixel Size", 1, 20, 4,
            tooltip="Mosaic block size in pixels",
        )
        _, self._thr_val,  self.thresh_sl = make_slider(
            dg.layout(), "Threshold",  0, 255, 128,
            tooltip="Binarisation threshold (0\u2013255)",
        )
        layout.addWidget(dg)

        adj = self._group("Adjustments")
        _, self._br_val, self.bright_sl = make_slider(
            adj.layout(), "Brightness", 0, 200, 100, "{v}%",
            tooltip="Brightness multiplier (100 = no change)",
        )
        _, self._co_val, self.contr_sl  = make_slider(
            adj.layout(), "Contrast",   0, 200, 100, "{v}%",
            tooltip="Contrast multiplier (100 = no change)",
        )
        _, self._sa_val, self.sat_sl = make_slider(
            adj.layout(), "Saturation", 0, 200, 100, "{v}%",
            tooltip="Colour saturation multiplier (100 = no change)",
        )
        _, self._hu_val, self.hue_sl = make_slider(
            adj.layout(), "Hue", 0, 359, 0, "{v}\u00b0",
            tooltip="Hue rotation in degrees",
        )
        _, self._bl_val, self.blur_sl   = make_slider(
            adj.layout(), "Blur",       0, 10,  0,
            tooltip="Gaussian blur radius before dithering",
        )
        _, self._sh_val, self.sharp_sl  = make_slider(
            adj.layout(), "Sharpen",    0, 5,   0,
            tooltip="Unsharp-mask strength",
        )
        layout.addWidget(adj)

        pre = self._group("Pre-dither Filters")
        _, self._prd_val, self.pre_denoise_sl = make_slider(
            pre.layout(), "Denoise", 0, 10, 0,
            tooltip="Median denoise strength before dithering",
        )
        _, self._prs_val, self.pre_smooth_sl = make_slider(
            pre.layout(), "Smooth", 0, 8, 0,
            tooltip="Smooth filter passes before dithering",
        )
        layout.addWidget(pre)

        reset_layout = QHBoxLayout()
        reset_layout.setContentsMargins(0, 0, 0, 0)
        reset_layout.setSpacing(4)
        
        reset_btn = QPushButton("↺  Reset Adjustments")
        reset_btn.clicked.connect(self._reset)
        reset_btn.setMinimumHeight(28)
        reset_btn.setToolTip("Reset image adjustments and filters to defaults")
        reset_layout.addWidget(reset_btn)
        
        reset_all_btn = QPushButton("↺  Reset All")
        reset_all_btn.clicked.connect(self.reset_all)
        reset_all_btn.setMinimumHeight(28)
        reset_all_btn.setToolTip("Fully reset all settings to defaults")
        reset_layout.addWidget(reset_all_btn)
        
        layout.addLayout(reset_layout)
        layout.addWidget(hsep())

        gg = self._group("Glow")
        _, self._gr_val, self.glow_r_sl = make_slider(
            gg.layout(), "Radius",    0, 40,  0,
            tooltip="Bloom glow blur radius",
        )
        _, self._gi_val, self.glow_i_sl = make_slider(
            gg.layout(), "Intensity", 0, 100, 0, "{v}%",
            tooltip="Bloom glow blend intensity",
        )
        layout.addWidget(gg)

        post = self._group("Post-dither Filters")
        _, self._pod_val, self.post_denoise_sl = make_slider(
            post.layout(), "Denoise", 0, 10, 0,
            tooltip="Median denoise strength after dithering",
        )
        _, self._pos_val, self.post_smooth_sl = make_slider(
            post.layout(), "Smooth", 0, 8, 0,
            tooltip="Smooth filter passes after dithering",
        )
        layout.addWidget(post)
        layout.addWidget(hsep())

        cg = self._group("Colour")
        self.swatch = QLabel()
        self.swatch.setFixedHeight(36)
        self.swatch.setAlignment(Qt.AlignCenter)
        self._refresh_swatch()
        cg.layout().addWidget(self.swatch)
        pick_btn = QPushButton("🖌  Pick Colour")
        pick_btn.setMinimumHeight(28)
        pick_btn.setToolTip("Choose the foreground dither colour")
        pick_btn.clicked.connect(self._pick_color)
        cg.layout().addWidget(pick_btn)
        layout.addWidget(cg)

        palg = self._group("Palette")
        self.palette_combo = QComboBox()
        self.palette_combo.setMinimumHeight(28)
        self.palette_combo.setToolTip("Select a colour palette for quantisation")
        for pal_name in PALETTES:
            self.palette_combo.addItem(pal_name)
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        palg.layout().addWidget(self.palette_combo)

        self.pal_swatch_widget = QWidget()
        self.pal_swatch_layout = QGridLayout(self.pal_swatch_widget)
        self.pal_swatch_layout.setSpacing(2)
        self.pal_swatch_layout.setContentsMargins(0, 2, 0, 2)
        palg.layout().addWidget(self.pal_swatch_widget)
        self._refresh_palette_swatches()

        custom_row = QHBoxLayout()
        self.custom_pal_btn   = QPushButton("＋  Add")
        self.clear_custom_btn = QPushButton("✕  Clear")
        for b in (self.custom_pal_btn, self.clear_custom_btn):
            b.setMinimumHeight(26)
        self.custom_pal_btn.setToolTip("Add a colour to the custom palette")
        self.clear_custom_btn.setToolTip("Clear the custom palette")
        self.custom_pal_btn.clicked.connect(self._add_custom_color)
        self.clear_custom_btn.clicked.connect(self._clear_custom_palette)
        custom_row.addWidget(self.custom_pal_btn)
        custom_row.addWidget(self.clear_custom_btn)
        palg.layout().addLayout(custom_row)
        layout.addWidget(palg)

        # Presets UI moved to top menu bar
        layout.addStretch()

        if _NUMBA:
            jit_lbl = QLabel("numba JIT active")
            jit_lbl.setAlignment(Qt.AlignCenter)
            jit_lbl.setStyleSheet(
                f"font-family:{_MONO_FONT}; font-size:9px;"
                f"color:{_AE}; padding:0 0 4px 0; letter-spacing:1px;"
            )
            layout.addWidget(jit_lbl)

        ver = QLabel(f"DITHER MAN  v{VERSION}")
        ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:9px; color:{_FG2};"
            "letter-spacing:3px; padding:8px 0 4px 0;"
        )
        layout.addWidget(ver)

        self._wire_sliders()

    @staticmethod
    def _group(title: str) -> QGroupBox:
        gb = QGroupBox(title)
        lyt = QVBoxLayout(gb)
        lyt.setSpacing(5)
        lyt.setContentsMargins(8, 6, 8, 8)
        return gb

    def _wire_sliders(self) -> None:
        pairs = [
            (self.pixel_sl,  self._pix_val, "{v}"),
            (self.thresh_sl, self._thr_val, "{v}"),
            (self.bright_sl, self._br_val,  "{v}%"),
            (self.contr_sl,  self._co_val,  "{v}%"),
            (self.sat_sl,    self._sa_val,  "{v}%"),
            (self.hue_sl,    self._hu_val,  "{v}\u00b0"),
            (self.blur_sl,   self._bl_val,  "{v}"),
            (self.sharp_sl,  self._sh_val,  "{v}"),
            (self.pre_denoise_sl,  self._prd_val, "{v}"),
            (self.pre_smooth_sl,   self._prs_val, "{v}"),
            (self.glow_r_sl, self._gr_val,  "{v}"),
            (self.glow_i_sl, self._gi_val,  "{v}%"),
            (self.post_denoise_sl, self._pod_val, "{v}"),
            (self.post_smooth_sl,  self._pos_val, "{v}"),
        ]
        for sl, lbl, fmt in pairs:
            sl.valueChanged.connect(lambda v, l=lbl, f=fmt: l.setText(f.format(v=v)))
            sl.valueChanged.connect(lambda _: self.params_changed_preview.emit())
            sl.sliderReleased.connect(self.params_changed.emit)

    def _reset(self) -> None:
        defaults = [
            (self.bright_sl, 100), (self.contr_sl, 100),
            (self.sat_sl, 100),    (self.hue_sl, 0),
            (self.blur_sl, 0),     (self.sharp_sl, 0),
            (self.pre_denoise_sl, 0), (self.pre_smooth_sl, 0),
            (self.glow_r_sl, 0),   (self.glow_i_sl, 0),
            (self.post_denoise_sl, 0), (self.post_smooth_sl, 0),
        ]
        for sl, v in defaults:
            sl.blockSignals(True)
            sl.setValue(v)
            sl.blockSignals(False)
        self._refresh_value_labels()
        self.params_changed.emit()

    def reset_all(self) -> None:
        """Fully reset all sliders, methods, and palettes to their defaults."""
        self._reset()
        self.pixel_sl.blockSignals(True)
        self.pixel_sl.setValue(2)
        self.pixel_sl.blockSignals(False)
        
        self.thresh_sl.blockSignals(True)
        self.thresh_sl.setValue(128)
        self.thresh_sl.blockSignals(False)

        self.method_picker.set_method("Bayer 2x2")
        self.palette_combo.setCurrentText("B&W")
        self.current_color = (255, 255, 255)
        self._refresh_swatch()
        self._custom_palette.clear()
        self._refresh_palette_swatches()
        
        self._refresh_value_labels()
        self.params_changed.emit()

    def randomize(self) -> None:
        """Randomize all dither properties and emit params_changed."""
        ri = random.randint

        # Algorithm
        method = random.choice(METHODS)
        self.method_picker.set_method(method)

        # Dither
        sliders_block = [
            (self.pixel_sl,        ri(1, 20)),
            (self.thresh_sl,       ri(0, 255)),
            # Adjustments  (keep brightness/contrast/sat in a sane range)
            (self.bright_sl,       ri(60, 160)),
            (self.contr_sl,        ri(60, 160)),
            (self.sat_sl,          ri(0, 200)),
            (self.hue_sl,          ri(0, 359)),
            (self.blur_sl,         ri(0, 3)),
            (self.sharp_sl,        ri(0, 3)),
            # Pre-dither filters
            # (self.pre_denoise_sl,  ri(0, 5)),
            # (self.pre_smooth_sl,   ri(0, 4)),
            # Glow
            (self.glow_r_sl,       ri(0, 8)),
            (self.glow_i_sl,       ri(0, 50)),
            # Post-dither filters
            # (self.post_denoise_sl, ri(0, 5)),
            # (self.post_smooth_sl,  ri(0, 4)),
        ]
        for sl, val in sliders_block:
            sl.blockSignals(True)
            sl.setValue(val)
            sl.blockSignals(False)
        self._refresh_value_labels()

        # Foreground colour
        self.current_color = (ri(0, 255), ri(0, 255), ri(0, 255))
        self._refresh_swatch()

        # Palette  (exclude "Custom" which has no colours by default)
        named_palettes = [n for n in PALETTES if n != "Custom"]
        if named_palettes:
            pal_name = random.choice(named_palettes)
            idx = self.palette_combo.findText(pal_name)
            if idx >= 0:
                self.palette_combo.blockSignals(True)
                self.palette_combo.setCurrentIndex(idx)
                self.palette_combo.blockSignals(False)
                self._refresh_palette_swatches()

        self.params_changed.emit()

    def _refresh_value_labels(self) -> None:
        self._pix_val.setText(str(self.pixel_sl.value()))
        self._thr_val.setText(str(self.thresh_sl.value()))
        self._br_val.setText(f"{self.bright_sl.value()}%")
        self._co_val.setText(f"{self.contr_sl.value()}%")
        self._sa_val.setText(f"{self.sat_sl.value()}%")
        self._hu_val.setText(f"{self.hue_sl.value()}\u00b0")
        self._bl_val.setText(str(self.blur_sl.value()))
        self._sh_val.setText(str(self.sharp_sl.value()))
        self._prd_val.setText(str(self.pre_denoise_sl.value()))
        self._prs_val.setText(str(self.pre_smooth_sl.value()))
        self._gr_val.setText(str(self.glow_r_sl.value()))
        self._gi_val.setText(f"{self.glow_i_sl.value()}%")
        self._pod_val.setText(str(self.post_denoise_sl.value()))
        self._pos_val.setText(str(self.post_smooth_sl.value()))

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(*self.current_color), self, "Pick Colour")
        if c.isValid():
            self.current_color = (c.red(), c.green(), c.blue())
            self._refresh_swatch()
            self.params_changed.emit()

    def _refresh_swatch(self) -> None:
        r, g, b = self.current_color
        lum = (r * 299 + g * 587 + b * 114) // 1000
        txt = "#000" if lum > 128 else "#fff"
        self.swatch.setStyleSheet(
            f"background:rgb({r},{g},{b}); border:1px solid {_P6};"
            f"border-radius:3px; color:{txt};"
            f"font-family:{_MONO_FONT}; font-size:10px; font-weight:bold;"
        )
        self.swatch.setText(f"#{r:02X}{g:02X}{b:02X}")
        self.swatch.setToolTip(f"Current colour: #{r:02X}{g:02X}{b:02X}  rgb({r},{g},{b})")

    def _on_palette_changed(self, _name: str) -> None:
        self._refresh_palette_swatches()
        self.params_changed.emit()

    def _refresh_palette_swatches(self) -> None:
        while self.pal_swatch_layout.count():
            item = self.pal_swatch_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        name   = self.palette_combo.currentText()
        colors = self._custom_palette if name == "Custom" else PALETTES.get(name, [])
        cols   = 8
        for i, (r, g, b) in enumerate(colors):
            sq = ClickableSwatch()
            sq.setFixedSize(16, 16)
            tooltip = f"#{r:02X}{g:02X}{b:02X}"
            if name == "Custom":
                tooltip += "\nLeft-click to edit\nRight-click to remove"
                sq.setCursor(Qt.PointingHandCursor)
            sq.setToolTip(tooltip)
            sq.setStyleSheet(
                f"background:rgb({r},{g},{b}); border:1px solid {_P6}; border-radius:2px;"
            )
            
            if name == "Custom":
                sq.rightClicked.connect(lambda idx=i: self._remove_custom_color(idx))
                sq.clicked.connect(lambda idx=i: self._edit_custom_color(idx))
                
            self.pal_swatch_layout.addWidget(sq, i // cols, i % cols)

    def _add_custom_color(self) -> None:
        c = QColorDialog.getColor(parent=self, title="Add Palette Color")
        if not c.isValid():
            return
        self._custom_palette.append((c.red(), c.green(), c.blue()))
        idx = self.palette_combo.findText("Custom")
        if idx < 0:
            self.palette_combo.addItem("Custom")
            idx = self.palette_combo.count() - 1
        self.palette_combo.blockSignals(True)
        self.palette_combo.setCurrentIndex(idx)
        self.palette_combo.blockSignals(False)
        self._refresh_palette_swatches()
        self.params_changed.emit()

    def _remove_custom_color(self, idx: int) -> None:
        if 0 <= idx < len(self._custom_palette):
            self._custom_palette.pop(idx)
            self._refresh_palette_swatches()
            self.params_changed.emit()

    def _edit_custom_color(self, idx: int) -> None:
        if 0 <= idx < len(self._custom_palette):
            old_c = self._custom_palette[idx]
            c = QColorDialog.getColor(QColor(*old_c), parent=self, title="Edit Palette Color")
            if c.isValid():
                self._custom_palette[idx] = (c.red(), c.green(), c.blue())
                self._refresh_palette_swatches()
                self.params_changed.emit()

    def _clear_custom_palette(self) -> None:
        self._custom_palette.clear()
        self._refresh_palette_swatches()
        self.params_changed.emit()

    def set_params(self, p: dict) -> None:
        widgets = [
            self.method_picker, self.palette_combo, self.pixel_sl, self.thresh_sl,
            self.bright_sl, self.contr_sl, self.sat_sl, self.hue_sl, self.blur_sl,
            self.sharp_sl, self.pre_denoise_sl, self.pre_smooth_sl, self.glow_r_sl,
            self.glow_i_sl, self.post_denoise_sl, self.post_smooth_sl
        ]
        
        for w in widgets:
            w.blockSignals(True)
            
        try:
            self.method_picker.set_method(p.get("method", "Floyd-Steinberg"))
            self.pixel_sl.setValue(p.get("pixel_size", 4))
            self.thresh_sl.setValue(p.get("threshold", 128))
            self.bright_sl.setValue(int(p.get("brightness", 1.0) * 100))
            self.contr_sl.setValue(int(p.get("contrast", 1.0) * 100))
            self.sat_sl.setValue(int(p.get("saturation", 1.0) * 100))
            self.hue_sl.setValue(p.get("hue_rotate", 0))
            self.blur_sl.setValue(p.get("blur", 0))
            self.sharp_sl.setValue(p.get("sharpen", 0))
            self.pre_denoise_sl.setValue(p.get("pre_denoise", 0))
            self.pre_smooth_sl.setValue(p.get("pre_smooth", 0))
            self.glow_r_sl.setValue(p.get("glow_radius", 0))
            self.glow_i_sl.setValue(p.get("glow_intensity", 0))
            self.post_denoise_sl.setValue(p.get("post_denoise", 0))
            self.post_smooth_sl.setValue(p.get("post_smooth", 0))
            self.current_color = tuple(p.get("color", (0, 255, 65)))
            self._refresh_swatch()
            
            if "custom_palette" in p and p["custom_palette"] is not None:
                self._custom_palette = [tuple(c) for c in p["custom_palette"]]
                
            pal_name = p.get("palette_name", "B&W")
            idx = self.palette_combo.findText(pal_name)
            if idx >= 0:
                self.palette_combo.setCurrentIndex(idx)
            else:
                self.palette_combo.setCurrentText(pal_name)
                
            self._refresh_palette_swatches()
            self._refresh_value_labels()
        finally:
            for w in widgets:
                w.blockSignals(False)
                
        self.params_changed.emit()

    def get_params(self) -> dict:
        pal_name = self.palette_combo.currentText()
        return {
            "method":         self.method_picker.current_method(),
            "pixel_size":     self.pixel_sl.value(),
            "threshold":      self.thresh_sl.value(),
            "brightness":     self.bright_sl.value() / 100.0,
            "contrast":       self.contr_sl.value()  / 100.0,
            "saturation":     self.sat_sl.value() / 100.0,
            "hue_rotate":     self.hue_sl.value(),
            "blur":           self.blur_sl.value(),
            "sharpen":        self.sharp_sl.value(),
            "pre_denoise":    self.pre_denoise_sl.value(),
            "pre_smooth":     self.pre_smooth_sl.value(),
            "color":          self.current_color,
            "glow_radius":    self.glow_r_sl.value(),
            "glow_intensity": self.glow_i_sl.value(),
            "post_denoise":   self.post_denoise_sl.value(),
            "post_smooth":    self.post_smooth_sl.value(),
            "palette_name":   pal_name,
            "custom_palette": self._custom_palette if pal_name == "Custom" else None,
        }
