from __future__ import annotations
from typing import TYPE_CHECKING

from PySide6.QtGui import QAction

if TYPE_CHECKING:
    from ui.window import DitherGuy

class WindowToolbar:
    def _build_toolbar(self: 'DitherGuy') -> None:
        # 1. Top Menu Bar (Nav Bar)
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        view_menu = menubar.addMenu("View")
        edit_menu = menubar.addMenu("Edit")
        self.presets_menu = menubar.addMenu("Presets")
        help_menu = menubar.addMenu("Help")
        
        def create_action(label: str, shortcut: str, slot, tip: str = "") -> QAction:
            a = QAction(label, self)
            if shortcut:
                a.setShortcut(shortcut)
            if tip:
                a.setStatusTip(tip)
                a.setToolTip(tip)
            a.triggered.connect(slot)
            return a
            
        a_open  = create_action("Open", "Ctrl+O", self._open, "Open image")
        a_save  = create_action("Save", "Ctrl+S", self._save, "Save output")
        a_batch = create_action("Batch Process", "Ctrl+B", self._batch, "Batch process folder")
        file_menu.addAction(a_open)
        file_menu.addAction(a_save)
        file_menu.addSeparator()
        file_menu.addAction(a_batch)
        
        a_undo = create_action("Undo", "Ctrl+Z", lambda: self.image_tab.undo(), "Undo last image operation")
        a_reset = create_action("Reset Defaults", "Ctrl+D", self._reset_all, "Reset settings")
        a_rand = create_action("Randomize", "Ctrl+R", self._randomize, "Randomize properties")
        edit_menu.addAction(a_undo)
        edit_menu.addSeparator()
        edit_menu.addAction(a_reset)
        edit_menu.addAction(a_rand)
        
        a_zin  = create_action("Zoom In", "Ctrl+=", self._zoom_in, "Zoom in")
        a_zout = create_action("Zoom Out", "Ctrl+-", self._zoom_out, "Zoom out")
        a_fit  = create_action("Fit on Screen", "Ctrl+0", self._fit, "Fit to window")
        a_1to1 = create_action("100%", "Ctrl+1", self._actual, "Actual pixels")
        view_menu.addAction(a_zin)
        view_menu.addAction(a_zout)
        view_menu.addAction(a_fit)
        view_menu.addAction(a_1to1)

        view_menu.addSeparator()
        self.a_force_orig = QAction("Force Original Video", self)
        self.a_force_orig.setCheckable(True)
        self.a_force_orig.setChecked(False)

        def _on_force_orig_toggled(checked: bool):
            self.video_tab.force_original = checked
            if getattr(self.video_tab, 'video_path', None):
                from PySide6.QtWidgets import QMessageBox
                if QMessageBox.question(self, "Reload Video", "Reload the video now to apply this change?") == QMessageBox.StandardButton.Yes:
                    self.video_tab._load_video(self.video_tab.video_path)
        
        self.a_force_orig.toggled.connect(_on_force_orig_toggled)
        self.a_force_orig.setStatusTip("Always load full resolution video instead of generating a 720p proxy")
        view_menu.addAction(self.a_force_orig)

        a_shortcuts = create_action("Keyboard Shortcuts", "Ctrl+?", self._show_shortcuts, "View all keyboard shortcuts")
        help_menu.addAction(a_shortcuts)

        a_about = create_action("About This Project", "", self._show_about, "About this application")
        help_menu.addAction(a_about)

        self._build_presets_menu()

        from PySide6.QtWidgets import QLabel
        from PySide6.QtCore import Qt
        from utils.theme import _MONO_FONT, _G0
        
        self.zoom_lbl = QLabel("fit")
        self.zoom_lbl.setMinimumWidth(52)
        self.zoom_lbl.setAlignment(Qt.AlignCenter)
        self.zoom_lbl.setStyleSheet(f"font-family:{_MONO_FONT}; color:{_G0}; font-size:11px; padding:0 8px;")
        self.zoom_lbl.setToolTip("Current zoom level")
        self.statusBar().addPermanentWidget(self.zoom_lbl)

    def _build_presets_menu(self: 'DitherGuy') -> None:
        self.presets_menu.clear()
        
        a_save = QAction("Save Preset", self)
        a_save.triggered.connect(self._save_preset)
        self.presets_menu.addAction(a_save)
        
        a_del = QAction("Manage Presets", self)
        a_del.triggered.connect(self._manage_presets)
        self.presets_menu.addAction(a_del)

        a_dl = QAction("Download Presets from GitHub", self)
        a_dl.triggered.connect(self._download_presets)
        self.presets_menu.addAction(a_dl)
        
        from utils.presets import list_presets
        presets = list_presets()
        if presets:
            self.presets_menu.addSeparator()
            for p in presets:
                a = QAction(p, self)
                a.triggered.connect(lambda checked=False, name=p: self._load_preset(name))
                self.presets_menu.addAction(a)
