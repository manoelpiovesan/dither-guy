from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Optional
import itertools

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal, QTimer, QUrl, QThread
from PySide6.QtGui import QImage, QPixmap, QGuiApplication
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QCheckBox, QFileDialog, QMessageBox,
    QProgressBar, QSlider, QStyle, QSplitter, QProgressDialog,
)

from .constants import _MAX_PIXELS, _HISTORY_LIMIT, _DEBOUNCE_MS
from .dither_kernels import apply_dither
from .theme import _P0, _P1, _P2, _P3, _P4, _P5, _P6, _G0, _G1, _G2, _G3, _FG, _FG2, _FG3, _RE, _AE, _MONO_FONT
from .workers import DitherWorker, VideoExportWorker
from .ui_widgets import ZoomableLabel, HistogramWidget, pil_to_pixmap, hsep, vsep

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    _MULTIMEDIA = True
except ImportError:
    _MULTIMEDIA = False

_worker_id_counter = itertools.count(1)


# ---------------------------------------------------------------------------
# Image tab
# ---------------------------------------------------------------------------

class ImageTab(QWidget):
    status_message = Signal(str)

    _IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp);;All (*.*)"
    _DROP_EXTS    = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}

    def __init__(self, get_params):
        super().__init__()
        self.get_params = get_params
        self.original_img:  Optional[Image.Image] = None
        self.dithered_img:  Optional[Image.Image] = None
        self.last_dir       = str(Path.home())
        self.worker:        Optional[DitherWorker] = None
        self._worker_id     = 0
        self._orphaned_workers = []
        self.auto_update    = True
        self._history:      list[Image.Image] = []
        self._timer         = QTimer(singleShot=True)
        self._timer.timeout.connect(self.process)
        self._preview_timer = QTimer(singleShot=True)
        self._preview_timer.timeout.connect(self._process_preview)
        self._split_visible = True
        self._build()

    # ── Build ──────────────────────────────────────────────────────────────

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.info_lbl = QLabel("Drop an image here · or Ctrl+O")
        self.info_lbl.setAlignment(Qt.AlignCenter)
        self.info_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:10px; padding:4px 8px;"
            f"background:{_P0}; color:{_FG3};"
        )
        layout.addWidget(self.info_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.canvas = ZoomableLabel("▣ Drop image here · Ctrl+O")
        self.canvas.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:14px; color:{_P5}; background:{_P0};"
        )
        scroll.setWidget(self.canvas)
        layout.addWidget(scroll, stretch=1)

        self.histogram = HistogramWidget()
        self.histogram.setVisible(False)
        layout.addWidget(self.histogram)

        # ── Toolbar ────────────────────────────────────────────────────────────
        bar1 = QWidget()
        bar1.setStyleSheet(f"background:{_P0};")
        bl1 = QHBoxLayout(bar1)
        bl1.setContentsMargins(8, 5, 8, 5)
        bl1.setSpacing(5)

        def _btn(label: str, slot, accent: bool = False, tip: str = "") -> QPushButton:
            b = QPushButton(label)
            if accent:
                b.setObjectName("accent")
            b.clicked.connect(slot)
            b.setMinimumHeight(28)
            b.setToolTip(tip)
            bl1.addWidget(b)
            return b

        _btn("◑  Invert",  self.invert)
        _btn("↺  Left",       self.rotate_left,  tip="Rotate 90° CCW")
        _btn("↻  Right",       self.rotate_right, tip="Rotate 90° CW")
        _btn("↔  Flip H",       self.flip_h,       tip="Flip horizontal")
        _btn("↕  Flip V",       self.flip_v,       tip="Flip vertical")
        _btn("✂  Crop",    self.crop)

        bl1.addWidget(vsep())


        self.hist_cb = QCheckBox("Histogram")
        self.hist_cb.stateChanged.connect(lambda s: self.histogram.setVisible(bool(s)))
        bl1.addWidget(self.hist_cb)

        bl1.addWidget(vsep())
        self.auto_cb = QCheckBox("Live")
        self.auto_cb.setChecked(True)
        self.auto_cb.setToolTip("Auto-update on parameter change")
        self.auto_cb.stateChanged.connect(self._toggle_auto)
        bl1.addWidget(self.auto_cb)

        self.apply_btn = QPushButton("▶  Apply")
        self.apply_btn.setObjectName("accent")
        self.apply_btn.clicked.connect(self.process)
        self.apply_btn.setVisible(False)
        self.apply_btn.setMinimumHeight(28)
        bl1.addWidget(self.apply_btn)
        bl1.addStretch()
        layout.addWidget(bar1)



    # ── Auto-update ──────────────────────────────────────────────────

    def _toggle_auto(self, state: int) -> None:
        self.auto_update = bool(state)
        self.apply_btn.setVisible(not self.auto_update)

    def schedule(self, preview: bool = False) -> None:
        if not self.auto_update:
            return
        if preview:
            self._preview_timer.stop()
            self._preview_timer.start(80)
        else:
            self._timer.stop()
            self._timer.start(_DEBOUNCE_MS)

    # ── Processing ──────────────────────────────────────────────────

    def _build_worker(self, preview: bool) -> DitherWorker:
        p = self.get_params()
        return DitherWorker(
            self.original_img,
            p["pixel_size"], p["threshold"], p["color"], p["method"],
            p["brightness"], p["contrast"], p["blur"], p["sharpen"],
            p["glow_radius"], p["glow_intensity"],
            preview=preview,
            palette_name=p.get("palette_name", "B&W"),
            custom_palette=p.get("custom_palette"),
            saturation=p.get("saturation", 1.0),
            hue_rotate=p.get("hue_rotate", 0),
            pre_denoise=p.get("pre_denoise", 0),
            pre_smooth=p.get("pre_smooth", 0),
            post_denoise=p.get("post_denoise", 0),
            post_smooth=p.get("post_smooth", 0),
        )

    def _process_preview(self) -> None:
        if self.original_img is None:
            return
        self._stop_worker()
        wid = next(_worker_id_counter)
        self._worker_id = wid
        self.worker = self._build_worker(preview=True)
        self.worker.result_ready.connect(lambda pl, _id=wid: self._on_done(pl, _id))
        self.worker.error.connect(lambda _: None)
        self.worker.start()

    def process(self) -> None:
        if self.original_img is None:
            return
        self._stop_worker()
        self.status_message.emit("processing...")
        if hasattr(self, "_progress_dlg"):
            self._progress_dlg.show()
        wid = next(_worker_id_counter)
        self._worker_id = wid
        self.worker = self._build_worker(preview=False)
        self.worker.result_ready.connect(lambda pl, _id=wid: self._on_done(pl, _id))
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _stop_worker(self) -> None:
        if getattr(self, "worker", None) and self.worker.isRunning():
            self._worker_id = 0
            self.worker.stop()
            
            # Safely orphan the old worker so it dies naturally without blocking UI
            # We connect to QThread's finished signal safely. Since 'finished' is shadowed
            # by our custom signal, we must ensure it gets deleted.
            old_worker = self.worker
            
            # Workaround for shadowed 'finished' signal:
            # We must keep a strong reference to the old worker, otherwise Python
            # garbage collects it while the C++ thread is still running, which
            # causes PySide6 to trigger a SIGABRT crash.
            old_worker = self.worker
            self._orphaned_workers.append(old_worker)
            
            def _cleanup(*args, w=old_worker):
                if w in self._orphaned_workers:
                    self._orphaned_workers.remove(w)
                w.deleteLater()

            old_worker.finished.connect(_cleanup)
            old_worker.error.connect(_cleanup)
            
            self.worker = None

    # ── File I/O ──────────────────────────────────────────────────────

    def cleanup(self) -> None:
        self._stop_worker()
        if hasattr(self, '_orphaned_workers'):
            for w in self._orphaned_workers:
                if getattr(w, 'isRunning', lambda: False)():
                    w.stop()
                    w.wait()
                w.deleteLater()
            self._orphaned_workers.clear()
        self.original_img = None
        self.dithered_img = None
        self._history.clear()

    def _load(self, path: str) -> None:
        try:
            img = Image.open(path)
            if img.width * img.height > _MAX_PIXELS:
                ans = QMessageBox.question(
                    self, "Large Image",
                    f"Image is {img.width}×{img.height}. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if ans != QMessageBox.Yes:
                    return
            self.original_img = img.convert("RGB")
            self._history.clear()
            self.last_dir = str(Path(path).parent)
            # show source in left pane immediately
            pass
            self._refresh_info()
            self.status_message.emit(f"loaded {Path(path).name}")
            self.process()
        except Exception as exc:
            QMessageBox.critical(self, "Open Error", f"Failed:\n{exc}")

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", self.last_dir, self._IMAGE_FILTER
        )
        if path:
            self._load(path)

    def load_from_qimage(self, qimg: QImage) -> None:
        """Load a QImage (e.g. pasted from clipboard) as the source image."""
        # Normalise to RGB32 so the byte layout is predictable
        qimg = qimg.convertToFormat(QImage.Format.Format_RGB32)
        w, h = qimg.width(), qimg.height()
        ptr  = qimg.bits()
        # bits() returns a sip.voidptr; convert to bytes then numpy
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
        # QImage RGB32 stores pixels as 0xffRRGGBB in little-endian → BGRA byte order
        pil_img = Image.fromarray(arr[:, :, ::-1][:, :, 1:], mode="RGB")
        if pil_img.width * pil_img.height > _MAX_PIXELS:
            ans = QMessageBox.question(
                self, "Large Image",
                f"Image is {pil_img.width}×{pil_img.height}. Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
        self.original_img = pil_img
        self._history.clear()

        self._refresh_info()
        self.process()

    # ── Clipboard paste ──────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_V and (event.modifiers() & Qt.ControlModifier):
            self._paste_from_clipboard()
        else:
            super().keyPressEvent(event)

    def _paste_from_clipboard(self) -> None:
        """
        Load an image from the clipboard.

        QMimeData.hasImage() is unreliable on Linux/X11 and Windows — images
        copied from browsers, screenshot tools, or file managers are typically
        exposed as:
          1. Qt image (hasImage == True)          — ideal path
          2. Raw PNG/JPEG/BMP bytes under an
             explicit MIME type                   — common on Linux
          3. A file:// URI in text/uri-list       — common when copying files
             in a file manager
        We attempt all three in order.
        """
        cb   = QGuiApplication.clipboard()
        mime = cb.mimeData()

        # ── 1. Qt native image ─────────────────────────────────────────────
        if mime.hasImage():
            qimg = cb.image()
            if not qimg.isNull():
                self.load_from_qimage(qimg)
                self.status_message.emit("pasted from clipboard")
                return

        # ── 2. Raw encoded bytes under explicit image MIME types ───────────
        for fmt in ("image/png", "image/bmp", "image/jpeg",
                    "image/jpg", "image/tiff", "image/webp",
                    "image/x-bmp", "application/octet-stream"):
            if mime.hasFormat(fmt):
                data = mime.data(fmt)
                if data and not data.isEmpty():
                    qimg = QImage()
                    raw = bytes(data)
                    if qimg.loadFromData(raw):
                        if not qimg.isNull():
                            self.load_from_qimage(qimg)
                            self.status_message.emit("pasted from clipboard")
                            return

        # ── 3. File URL (e.g. copied file in file manager) ─────────────────
        if mime.hasUrls():
            for url in mime.urls():
                path = url.toLocalFile()
                if path and Path(path).suffix.lower() in self._DROP_EXTS:
                    self._load(path)
                    self.status_message.emit(f"pasted from {Path(path).name}")
                    return

        # ── 4. Text that looks like a local path ───────────────────────────
        if mime.hasText():
            text = mime.text().strip()
            p = Path(text)
            if p.exists() and p.suffix.lower() in self._DROP_EXTS:
                self._load(str(p))
                self.status_message.emit(f"pasted from {p.name}")
                return

        self.status_message.emit("nothing to paste — no image found in clipboard")

    def get_result_pixmap(self) -> Optional[QPixmap]:
        """Return the current dithered-output pixmap at full resolution."""
        pm = self.canvas.original_pixmap
        if pm is None or pm.isNull():
            return None
        return pm

    def save_file(self) -> None:
        if self.dithered_img is None:
            QMessageBox.warning(self, "Nothing to save", "Process an image first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image",
            str(Path(self.last_dir) / "dithered.png"),
            "PNG (*.png);;JPEG (*.jpg);;All (*.*)",
        )
        if path:
            try:
                self.dithered_img.save(path)
                self.last_dir = str(Path(path).parent)
                self.status_message.emit(f"saved {Path(path).name}")
            except Exception as exc:
                QMessageBox.critical(self, "Save Error", f"Failed:\n{exc}")

    # ── History ──────────────────────────────────────────────────────

    def _refresh_info(self) -> None:
        if self.original_img:
            w, h = self.original_img.size
            self.info_lbl.setText(f"{w}×{h} px · history: {len(self._history)}")

    def _push_history(self) -> None:
        if self.original_img is None:
            return
        self._history.append(self.original_img.copy())
        if len(self._history) > _HISTORY_LIMIT:
            self._history.pop(0)

    def _require_image(self, op: str = "do this") -> bool:
        if self.original_img is None:
            QMessageBox.warning(self, "No Image", f"Load an image to {op}.")
            return False
        return True

    # ── Image operations ─────────────────────────────────────────────────

    def _op(self, fn) -> None:
        if not self._require_image():
            return
        self._push_history()
        self.original_img = fn(self.original_img)

        self._refresh_info()
        self.process()

    def invert(self) -> None:
        if not self._require_image("invert"): return
        self._op(lambda img: Image.fromarray((255 - np.array(img)).astype(np.uint8)))
        self.status_message.emit("inverted")

    def rotate_left(self) -> None:
        if not self._require_image("rotate"): return
        self._op(lambda img: img.rotate(90, expand=True))
        self.status_message.emit("rotated 90° CCW")

    def rotate_right(self) -> None:
        if not self._require_image("rotate"): return
        self._op(lambda img: img.rotate(-90, expand=True))
        self.status_message.emit("rotated 90° CW")

    def flip_h(self) -> None:
        if not self._require_image("flip"): return
        self._op(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT))
        self.status_message.emit("flipped H")

    def flip_v(self) -> None:
        if not self._require_image("flip"): return
        self._op(lambda img: img.transpose(Image.FLIP_TOP_BOTTOM))
        self.status_message.emit("flipped V")

    def crop(self) -> None:
        if not self._require_image("crop"):
            return
        assert self.original_img is not None
        from .ui_dialogs import CropDialog
        from PySide6.QtWidgets import QDialog
        dlg = CropDialog(self.original_img.width, self.original_img.height, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        v  = dlg.values()
        l, t, r, b = v["left"], v["top"], v["right"], v["bottom"]
        x2 = self.original_img.width  - r
        y2 = self.original_img.height - b
        if l >= x2 or t >= y2:
            QMessageBox.warning(self, "Crop", "Nothing left after crop.")
            return
        self._push_history()
        self.original_img = self.original_img.crop((l, t, x2, y2))
        self.status_message.emit(
            f"cropped → {self.original_img.width}×{self.original_img.height}"
        )
        self._refresh_info()
        self.process()

    def undo(self) -> None:
        if not self._history:
            return
        self.original_img = self._history.pop()
        self.status_message.emit("undo")
        self._refresh_info()
        self.process()

    # ── Worker callbacks ────────────────────────────────────────────────

    def _on_done(self, payload, worker_id: int) -> None:
        if payload is None:
            return
        if worker_id != self._worker_id:
            return
        
        # Reset the popup and status message
        if hasattr(self, "_progress_dlg"):
            self._progress_dlg.reset()
        self.status_message.emit("ready")

        img, elapsed, is_preview = payload
        if not is_preview:
            self.dithered_img = img
        if self.hist_cb.isChecked():
            self.histogram.update_data(img)
        self.canvas.set_image(pil_to_pixmap(img))
        self.canvas.setStyleSheet(f"background:{_P0};")

    def _on_error(self, msg: str) -> None:
        if hasattr(self, "_progress_dlg"):
            self._progress_dlg.reset()
        self.status_message.emit(f"error: {msg}")
        QMessageBox.warning(self, "Processing Error", msg)

    # ── Zoom proxy ────────────────────────────────────────────────────

    def zoom_in(self)  -> None: self.canvas.zoom_in()
    def zoom_out(self) -> None: self.canvas.zoom_out()
    def fit(self)      -> None: self.canvas.fit()
    def actual(self)   -> None: self.canvas.actual()

    @property
    def zoom_level(self) -> float:
        return self.canvas.zoom_level


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ---------------------------------------------------------------------------
# Video tab
# ---------------------------------------------------------------------------

class ProxyGeneratorWorker(QThread):
    progress = Signal(int, int)
    finished_proxy = Signal(str)
    error = Signal(str)

    def __init__(self, input_path: str, target_width: int, target_height: int):
        super().__init__()
        self.input_path = input_path
        self.target_width = target_width
        self.target_height = target_height

    def run(self):
        try:
            import os
            cap = cv2.VideoCapture(self.input_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            
            proxy_path = os.path.join(tempfile.gettempdir(), f"proxy_{os.path.basename(self.input_path)}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(proxy_path, fourcc, fps, (self.target_width, self.target_height))
            
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                count += 1
                if count % 10 == 0 or count == total:
                    self.progress.emit(count, total)
                    
            cap.release()
            writer.release()
            self.finished_proxy.emit(proxy_path)
        except Exception as e:
            self.error.emit(str(e))

class VideoTab(QWidget):
    status_message = Signal(str)

    _VIDEO_FILTER = "Video (*.mp4 *.avi *.mov *.mkv *.webm);;All (*.*)"

    def __init__(self, get_params):
        super().__init__()
        self.get_params    = get_params
        self.video_cap     = None
        self.video_path    = None
        self.current_frame = None
        self.is_playing    = False
        self.loop          = False
        self.export_worker = None
        self.last_dir      = str(Path.home())
        self._fps          = 25.0
        self._total_frames = 0
        self._scrubbing    = False
        self._orphaned_workers = []
        self._frame_worker: Optional[FrameDitherWorker] = None
        self._proxy_worker: Optional[ProxyGeneratorWorker] = None
        self._proxy_dlg: Optional[QProgressDialog] = None
        self._play_timer   = QTimer()
        self._play_timer.timeout.connect(self._next_frame)

        # ── QMediaPlayer for audio ─────────────────────────────────────────
        if _MULTIMEDIA:
            self._audio_out = QAudioOutput()
            self._audio_out.setVolume(1.0)
            self._player = QMediaPlayer()
            self._player.setAudioOutput(self._audio_out)
            # no video output — audio only
        else:
            self._player    = None
            self._audio_out = None

        self._build()
        self.setFocusPolicy(Qt.StrongFocus)

    # ── Audio helpers ──────────────────────────────────────────────────────

    def _audio_enabled(self) -> bool:
        return _MULTIMEDIA and self._player is not None and self.audio_cb.isChecked()

    def _player_seek_ms(self, frame_idx: int) -> None:
        """Seek QMediaPlayer to match cv2 frame position."""
        if not self._audio_enabled():
            return
        ms = int(frame_idx / max(1.0, self._fps) * 1000)
        self._player.setPosition(ms)

    def _sync_audio_volume(self) -> None:
        if not _MULTIMEDIA or self._audio_out is None:
            return
        self._audio_out.setVolume(1.0 if self.audio_cb.isChecked() else 0.0)

    # ── Build ──────────────────────────────────────────────────────────────

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.info_lbl = QLabel("no video loaded")
        self.info_lbl.setAlignment(Qt.AlignCenter)
        self.info_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:10px; padding:4px;"
            f"background:{_P0}; color:{_FG3};"
        )
        layout.addWidget(self.info_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.canvas = ZoomableLabel("▶ Load a video · MP4 / AVI / MOV / MKV")
        self.canvas.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:14px; color:{_P5}; background:{_P0};"
        )
        scroll.setWidget(self.canvas)

        self.hq_warn_lbl = QLabel()
        self.hq_warn_lbl.setAlignment(Qt.AlignCenter)
        self.hq_warn_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:11px; padding:6px;"
            f"background:#4A3B12; color:#FFC107; border-radius:4px;"
        )
        self.hq_warn_lbl.setVisible(False)
        
        layout.addWidget(self.hq_warn_lbl)
        layout.addWidget(scroll, stretch=1)

        self.export_bar = QProgressBar()
        self.export_bar.setVisible(False)
        self.export_bar.setFixedHeight(4)
        self.export_bar.setTextVisible(False)
        self.export_bar.setStyleSheet(
            f"QProgressBar {{ background:{_P4}; border:none; }}"
            f"QProgressBar::chunk {{ background:{_AE}; }}"
        )
        layout.addWidget(self.export_bar)

        # ── Unified Player GUI ─────────────────────────────────────────────
        player_frame = QWidget()
        player_frame.setStyleSheet(
            f"QWidget {{ background:{_P1}; border-top:1px solid {_P5}; }}"
            f"QPushButton {{ background:{_P4}; border:1px solid {_P5}; border-radius:4px; color:{_FG}; font-size:13px; padding:4px 8px; }}"
            f"QPushButton:hover {{ background:{_P5}; border-color:{_P6}; }}"
            f"QPushButton:pressed {{ background:{_P3}; }}"
            f"QPushButton#accent {{ background:{_AE}; border:1px solid {_AE}; color:#000; font-weight:bold; }}"
            f"QPushButton#accent:hover {{ background:#8AEB8A; }}"
            f"QSlider::groove:horizontal {{ border:1px solid {_P5}; height:6px; background:{_P2}; border-radius:3px; }}"
            f"QSlider::sub-page:horizontal {{ background:{_AE}; border-radius:3px; }}"
            f"QSlider::handle:horizontal {{ background:#FFF; border:1px solid #777; width:12px; margin:-3px 0; border-radius:6px; }}"
        )
        vbl = QVBoxLayout(player_frame)
        vbl.setContentsMargins(12, 12, 12, 12)
        vbl.setSpacing(8)

        # Top row: Seek slider
        seek_layout = QHBoxLayout()
        seek_layout.setContentsMargins(0, 0, 0, 0)
        seek_layout.setSpacing(10)

        self._pos_lbl = QLabel("0:00")
        self._pos_lbl.setStyleSheet(f"font-family:{_MONO_FONT}; font-size:12px; color:{_FG2}; min-width:35px;")
        self._pos_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.seek_bar = QSlider(Qt.Horizontal)
        self.seek_bar.setMinimum(0)
        self.seek_bar.setMaximum(1000)
        self.seek_bar.setValue(0)
        self.seek_bar.setEnabled(False)
        self.seek_bar.setCursor(Qt.PointingHandCursor)
        self.seek_bar.setToolTip("Seek / scrub  (← / → keys for ±5 s)")
        self.seek_bar.sliderPressed.connect(self._on_seek_press)
        self.seek_bar.sliderMoved.connect(self._on_seek_move)
        self.seek_bar.sliderReleased.connect(self._on_seek_release)

        self._dur_lbl = QLabel("0:00")
        self._dur_lbl.setStyleSheet(f"font-family:{_MONO_FONT}; font-size:12px; color:{_FG3}; min-width:35px;")
        self._dur_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        seek_layout.addWidget(self._pos_lbl)
        seek_layout.addWidget(self.seek_bar, stretch=1)
        seek_layout.addWidget(self._dur_lbl)
        vbl.addLayout(seek_layout)

        # Bottom row: Controls
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(6)

        def _tbtn(label: str, slot, tip: str = "", oid: str = "") -> QPushButton:
            b = QPushButton(label)
            b.setMinimumHeight(28)
            b.setCursor(Qt.PointingHandCursor)
            if oid: b.setObjectName(oid)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            btn_layout.addWidget(b)
            return b

        self.jump_start_btn = _tbtn("⏮", self._rewind, "Jump to start  (Home)")
        self.step_b_btn = _tbtn("⏴", self._step_backward, "Step backward  (Left Arrow)")
        self.play_btn   = _tbtn("▶", self.toggle_play, "Play / Pause  (Space)", "accent")
        self.play_btn.setFixedWidth(64)
        self.stop_btn   = _tbtn("■", self.stop, "Stop  (S)")
        self.step_f_btn = _tbtn("⏵", self._step_forward, "Step forward  (Right Arrow)")
        self.jump_end_btn = _tbtn("⏭", self._end, "Jump to end  (End)")

        btn_layout.addSpacing(12)

        self.loop_btn = QPushButton("⟳  Loop")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setMinimumHeight(28)
        self.loop_btn.setCursor(Qt.PointingHandCursor)
        self.loop_btn.setToolTip("Toggle loop playback  (L)")
        self.loop_btn.setStyleSheet(
            f"QPushButton:checked {{ background:{_G3}; border-color:{_G2}; color:{_G0}; font-weight:bold; }}"
        )
        self.loop_btn.toggled.connect(self._on_loop_toggled)
        btn_layout.addWidget(self.loop_btn)

        btn_layout.addSpacing(12)

        self.audio_cb = QCheckBox("Audio")
        self.audio_cb.setChecked(True)
        self.audio_cb.setToolTip("Play audio in preview / Export with audio")
        self.audio_cb.stateChanged.connect(lambda _: self._sync_audio_volume())
        btn_layout.addWidget(self.audio_cb)

        btn_layout.addStretch()

        self._frame_badge = QLabel("-- / --")
        self._frame_badge.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:11px; color:{_FG2};"
            f"background:{_P2}; border:1px solid {_P5}; border-radius:3px; padding:3px 8px;"
        )
        btn_layout.addWidget(self._frame_badge)

        self._fps_badge = QLabel("-- fps")
        self._fps_badge.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:11px; color:{_FG3};"
            f"background:{_P2}; border:1px solid {_P5}; border-radius:3px; padding:3px 8px;"
        )
        btn_layout.addWidget(self._fps_badge)

        btn_layout.addSpacing(12)

        self.export_btn = QPushButton("⥅  Export")
        self.export_btn.setObjectName("accent")
        self.export_btn.setMinimumHeight(28)
        self.export_btn.setCursor(Qt.PointingHandCursor)
        self.export_btn.setToolTip("Export dithered video to MP4")
        self.export_btn.clicked.connect(self._on_export_click)
        btn_layout.addWidget(self.export_btn)

        vbl.addLayout(btn_layout)
        layout.addWidget(player_frame)

        if not _CV2:
            warn = QLabel("⚠ opencv-python not installed — video disabled")
            warn.setAlignment(Qt.AlignCenter)
            warn.setStyleSheet(
                f"color:{_RE}; font-family:{_MONO_FONT}; font-size:10px;"
                f"padding:4px; background:{_P1};"
            )
            layout.addWidget(warn)

        if not _MULTIMEDIA:
            warn2 = QLabel("⚠ PySide6-Multimedia unavailable — preview audio disabled")
            warn2.setAlignment(Qt.AlignCenter)
            warn2.setStyleSheet(
                f"color:{_FG3}; font-family:{_MONO_FONT}; font-size:10px;"
                f"padding:3px; background:{_P1};"
            )
            layout.addWidget(warn2)

        self._set_controls_enabled(False)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (self.jump_start_btn, self.step_b_btn, self.play_btn, self.stop_btn, self.step_f_btn, self.jump_end_btn,
                  self.loop_btn, self.audio_cb, self.export_btn, self.seek_bar):
            w.setEnabled(enabled)

    def open_file(self) -> None:
        if not _CV2:
            QMessageBox.warning(self, "Missing", "Install opencv-python.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", self.last_dir, self._VIDEO_FILTER
        )
        if not path:
            return
        self._load_video(path)

    def _load_video(self, path: str) -> None:
        if self.is_playing:
            self._play_timer.stop()
            self.is_playing = False
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None

        # stop any previous QMediaPlayer session
        if self._player is not None:
            self._player.stop()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            QMessageBox.critical(self, "Error", "Cannot open video.")
            return

        self.video_cap     = cap
        self.video_path    = path
        self._fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.last_dir      = str(Path(path).parent)

        # load file into QMediaPlayer (audio only — no video output assigned)
        if self._player is not None:
            self._player.setSource(QUrl.fromLocalFile(str(Path(path).resolve())))
            self._sync_audio_volume()

        duration = self._total_frames / self._fps
        name     = Path(path).name
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        max_dim = max(w, h)
        if max_dim > 720 and not getattr(self, 'force_original', False):
            scale = 720.0 / max_dim
            tw, th = int(w * scale), int(h * scale)
            tw -= tw % 2
            th -= th % 2
            
            self._proxy_dlg = QProgressDialog("Generating proxy video to prevent lag...", "", 0, self._total_frames, self)
            self._proxy_dlg.setWindowTitle("Compressing Video for Preview")
            self._proxy_dlg.setCancelButton(None)
            self._proxy_dlg.setWindowModality(Qt.WindowModal)
            self._proxy_dlg.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
            self._proxy_dlg.setMinimumDuration(0)
            self._proxy_dlg.setValue(0)
            self._proxy_dlg.show()
            
            cap.release()
            self.video_cap = None
            
            import time
            self._proxy_start_time = time.time()
            self._proxy_worker = ProxyGeneratorWorker(path, tw, th)
            
            def _on_proxy_progress(cur: int) -> None:
                if not hasattr(self, '_proxy_dlg') or self._proxy_dlg is None: return
                self._proxy_dlg.setValue(cur)
                elapsed = time.time() - getattr(self, '_proxy_start_time', time.time())
                fps = cur / elapsed if elapsed > 0 else 0
                total = self._total_frames
                if fps > 0:
                    eta_sec = (total - cur) / fps
                    m, s = divmod(int(eta_sec), 60)
                    h, m = divmod(m, 60)
                    eta_str = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s"
                else:
                    eta_str = "..."
                pct = int((cur / total) * 100) if total > 0 else 0
                self._proxy_dlg.setLabelText(f"Generating proxy video to prevent lag... {pct}%  •  {fps:.1f} fps  •  ETA {eta_str}")
            
            self._proxy_worker.progress.connect(_on_proxy_progress)
            self._proxy_worker.error.connect(lambda e: self._on_proxy_done(None, path, duration, name, w, h))
            self._proxy_worker.finished_proxy.connect(lambda p_path: self._on_proxy_done(p_path, path, duration, name, w, h))
            self._proxy_worker.start()
        else:
            self.hq_warn_lbl.setVisible(False)
            self._finish_load_video(path, path, duration, name, w, h)

    def _on_proxy_done(self, proxy_path: Optional[str], orig_path: str, duration, name, orig_w, orig_h):
        if hasattr(self, "_proxy_dlg") and self._proxy_dlg is not None:
            self._proxy_dlg.close()
            self._proxy_dlg = None
        if proxy_path and Path(proxy_path).exists():
            self.hq_warn_lbl.setText(
                f"⚠️({orig_w}×{orig_h}) Expect lag when using color_palette/Large Video  . Export will be full resolution."
            )
            self.hq_warn_lbl.setVisible(True)
            self._finish_load_video(proxy_path, orig_path, duration, name, orig_w, orig_h)
        else:
            self.hq_warn_lbl.setVisible(False)
            self._finish_load_video(orig_path, orig_path, duration, name, orig_w, orig_h)

    def _finish_load_video(self, load_path: str, orig_path: str, duration: float, name: str, w: int, h: int) -> None:
        self.video_cap  = cv2.VideoCapture(load_path)
        self.video_path = orig_path
        self.orig_w     = w
        self.orig_h     = h
        
        self.info_lbl.setText(
            f"{name}  ·  {w}×{h}  ·  {self._total_frames} fr  ·  {self._fps:.2f} fps  ·  {_fmt_time(duration)}"
        )
        self._fps_badge.setText(f"{self._fps:.2f} fps")
        self._dur_lbl.setText(_fmt_time(duration))
        self.seek_bar.setValue(0)
        self._set_controls_enabled(True)
        self.play_btn.setText("▶")
        self.status_message.emit(f"loaded {name}")
        self._seek_to_frame(0)

    def toggle_play(self) -> None:
        if self.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        if not self.video_cap:
            return
        self.is_playing = True
        self.play_btn.setText("⏸")
        interval = max(1, int(1000 / self._fps))
        self._play_timer.start(interval)
        if self._audio_enabled():
            self._player.play()
        self.status_message.emit("playing")

    def _pause(self) -> None:
        self.is_playing = False
        self._play_timer.stop()
        self.play_btn.setText("▶")
        if self._player is not None:
            self._player.pause()
        self.status_message.emit("paused")

    def stop(self) -> None:
        self._pause()
        self._seek_to_frame(0)
        self.status_message.emit("stopped")

    def _rewind(self) -> None:
        was_playing = self.is_playing
        if was_playing: self._pause()
        self._seek_to_frame(0)
        if was_playing: self._play()

    def _end(self) -> None:
        was_playing = self.is_playing
        if was_playing: self._pause()
        self._seek_to_frame(max(0, self._total_frames - 1))

    def _step_backward(self) -> None:
        if self.is_playing: self._pause()
        if not self.video_cap or not self.video_cap.isOpened(): return
        current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._seek_to_frame(max(0, current_frame - 1))

    def _step_forward(self) -> None:
        if self.is_playing: self._pause()
        if not self.video_cap or not self.video_cap.isOpened(): return
        current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._seek_to_frame(min(self._total_frames - 1, current_frame + 1))

    def _on_loop_toggled(self, checked: bool) -> None:
        self.loop = checked
        if self._player is not None:
            from PySide6.QtMultimedia import QMediaPlayer as _QMP
            self._player.setLoops(_QMP.Infinite if checked else 1)

    def _on_seek_press(self) -> None:
        self._scrubbing = True
        if self.is_playing:
            self._play_timer.stop()
            if self._player is not None:
                self._player.pause()

    def _on_seek_move(self, value: int) -> None:
        frame_idx = int(value / 1000 * (self._total_frames - 1))
        self._seek_to_frame(frame_idx, update_bar=False)

    def _on_seek_release(self) -> None:
        self._scrubbing = False
        frame_idx = int(self.seek_bar.value() / 1000 * (self._total_frames - 1))
        self._seek_to_frame(frame_idx)
        if self.is_playing:
            self._play_timer.start(max(1, int(1000 / self._fps)))
            if self._audio_enabled():
                self._player.play()

    def _seek_to_frame(self, frame_idx: int, update_bar: bool = True) -> None:
        if not self.video_cap:
            return
        frame_idx = max(0, min(frame_idx, self._total_frames - 1))
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()
        if not ret:
            return
            
        self.current_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._show(self.current_frame)
        self._update_position_ui(frame_idx, update_bar)
        self._player_seek_ms(frame_idx)

    def _update_position_ui(self, frame_idx: int, update_bar: bool = True) -> None:
        elapsed = frame_idx / max(1, self._fps)
        self._pos_lbl.setText(_fmt_time(elapsed))
        self._frame_badge.setText(f"{frame_idx + 1} / {self._total_frames}")
        if update_bar and self._total_frames > 1:
            pos = int(frame_idx / (self._total_frames - 1) * 1000)
            self.seek_bar.blockSignals(True)
            self.seek_bar.setValue(pos)
            self.seek_bar.blockSignals(False)

    def _next_frame(self) -> None:
        if not self.video_cap or not self.video_cap.isOpened():
            return
            
        frame_idx = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Audio sync: drop frames if lagging, stall if ahead
        if self.is_playing and self._audio_enabled() and self._player is not None:
            pos_ms = self._player.position()
            target_frame = int((pos_ms / 1000.0) * self._fps)
            diff = target_frame - frame_idx
            
            # If audio looped or jumped backwards significantly, sync video backwards
            if diff < -max(10, int(self._fps / 2)):
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                frame_idx = target_frame
            elif diff > 2:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                frame_idx = target_frame
            elif diff < -2:
                return

        ret, frame = self.video_cap.read()
        if not ret:
            if self.loop:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # QMediaPlayer loop handled via setLoops; just seek to keep sync
                if self._audio_enabled():
                    self._player.setPosition(0)
                self._next_frame()
            else:
                self._pause()
                self._seek_to_frame(0)
            return
            
        self.current_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._show(self.current_frame)
        self._update_position_ui(frame_idx)

    def _show(self, img: Image.Image) -> None:
        p = self.get_params()
        if getattr(self, '_frame_worker', None) and self._frame_worker.isRunning():
            if self.is_playing:
                return  # Drop frame instead of cancelling to ensure frames actually render
            else:
                self._frame_worker.stop()
                
                # Safely orphan the worker in case it doesn't stop immediately
                old_w = self._frame_worker
                if not hasattr(self, '_orphaned_workers'):
                    self._orphaned_workers = []
                self._orphaned_workers.append(old_w)
                
                def _cleanup(*args, w=old_w):
                    if w in self._orphaned_workers:
                        self._orphaned_workers.remove(w)
                    w.deleteLater()
                    
                old_w.finished.connect(_cleanup)
                self._frame_worker = None

        from .workers import FrameDitherWorker
        self._frame_worker = FrameDitherWorker(img, p, preview=True, is_video=True)
        self._frame_worker.result_ready.connect(self._on_frame_dithered)
        self._frame_worker.start()

    def _on_frame_dithered(self, dith: Image.Image) -> None:
        if dith is None:
            return
        try:
            self.canvas.set_image(pil_to_pixmap(dith))
            self.canvas.setStyleSheet(f"background:{_P0};")
        except Exception as exc:
            self.status_message.emit(f"frame error: {exc}")

    def cleanup(self) -> None:
        self.is_playing = False
        if getattr(self, '_play_timer', None) and self._play_timer.isActive():
            self._play_timer.stop()
        if getattr(self, '_frame_worker', None) and self._frame_worker.isRunning():
            self._frame_worker.stop()
            self._frame_worker.wait()
        if getattr(self, 'export_worker', None) and self.export_worker.isRunning():
            self.export_worker.stop()
            self.export_worker.wait()
        if getattr(self, '_proxy_worker', None) and getattr(self._proxy_worker, 'isRunning', lambda: False)():
            self._proxy_worker.stop()
            self._proxy_worker.wait()
        if hasattr(self, '_orphaned_workers'):
            for w in self._orphaned_workers:
                if getattr(w, 'isRunning', lambda: False)():
                    w.stop()
                    w.wait()
                w.deleteLater()
            self._orphaned_workers.clear()
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        if self._player is not None:
            self._player.stop()
        self.current_frame = None

    def keyPressEvent(self, event) -> None:
        if not self.video_cap:
            return super().keyPressEvent(event)
        key = event.key()
        if key == Qt.Key_Space:
            self.toggle_play()
        elif key == Qt.Key_S:
            self.stop()
        elif key == Qt.Key_L:
            self.loop_btn.setChecked(not self.loop_btn.isChecked())
        elif key == Qt.Key_Home:
            self._rewind()
        elif key == Qt.Key_End:
            self._end()
        elif key in (Qt.Key_Right, Qt.Key_Period):
            cur = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._seek_to_frame(cur + int(self._fps * 5))
        elif key in (Qt.Key_Left, Qt.Key_Comma):
            cur = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._seek_to_frame(cur - int(self._fps * 5))
        elif key == Qt.Key_BracketRight:
            cur = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._seek_to_frame(cur + 1)
        elif key == Qt.Key_BracketLeft:
            cur = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._seek_to_frame(max(0, cur - 2))
        else:
            super().keyPressEvent(event)

    def export_video(self) -> None:
        if not _CV2:
            QMessageBox.warning(self, "Missing", "Install opencv-python.")
            return
        if not self.video_path:
            QMessageBox.warning(self, "No Video", "Load a video first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Video",
            str(Path(self.last_dir) / "dithered.mp4"),
            "MP4 (*.mp4)",
        )
        if not path:
            return
        if self.is_playing:
            self._pause()
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        p = self.get_params()
        
        # Dynamic scaling is now handled internally by apply_dither for videos

        self.export_worker = VideoExportWorker(
            self.video_path, path,
            p["pixel_size"], p["threshold"], p["color"], p["method"],
            p["brightness"], p["contrast"], p["blur"], p["sharpen"],
            p.get("glow_radius", 0), p.get("glow_intensity", 0),
            palette_name=p.get("palette_name", "B&W"),
            custom_palette=p.get("custom_palette"),
            saturation=p.get("saturation", 1.0),
            hue_rotate=p.get("hue_rotate", 0),
            pre_denoise=p.get("pre_denoise", 0),
            pre_smooth=p.get("pre_smooth", 0),
            post_denoise=p.get("post_denoise", 0),
            post_smooth=p.get("post_smooth", 0),
            include_audio=self.audio_cb.isChecked(),
        )
        self.export_worker.frame_ready.connect(
            lambda img: self.canvas.set_image(pil_to_pixmap(img))
        )
        self.export_worker.progress.connect(self._on_export_progress)
        self.export_worker.finished.connect(self._on_export_done)
        self.export_worker.error.connect(
            lambda msg: QMessageBox.critical(self, "Export Error", msg)
        )
        self.export_bar.setVisible(True)
        self.export_bar.setValue(0)
        self.status_message.emit("exporting...")
        
        self.export_btn.setText("⨉  Cancel")
        self.export_btn.setObjectName("")
        self.export_btn.setStyleSheet(f"background: {_RE}; color: #FFF; font-weight: bold;")
        
        import time
        self._export_start_time = time.time()
        self.export_worker.start()

    def _on_export_click(self) -> None:
        if self.export_worker and self.export_worker.isRunning():
            self._cancel_export()
        else:
            self.export_video()

    def _cancel_export(self) -> None:
        if self.export_worker is not None and self.export_worker.isRunning():
            worker = self.export_worker
            self.export_worker = None
            worker.stop()
            worker.wait()
            self._orphaned_workers.append(worker)
            worker.deleteLater()
            self.export_bar.setVisible(False)
            self._reset_export_btn()
            self._set_controls_enabled(True)
            self.status_message.emit("export cancelled")

    def _reset_export_btn(self) -> None:
        self.export_btn.setText("⥅  Export")
        self.export_btn.setObjectName("accent")
        self.export_btn.setStyleSheet("")

    def _on_export_progress(self, cur: int, total: int) -> None:
        self.export_bar.setMaximum(total)
        self.export_bar.setValue(cur)
        import time
        elapsed = time.time() - getattr(self, '_export_start_time', time.time())
        fps = cur / elapsed if elapsed > 0 else 0
        if fps > 0:
            eta_sec = (total - cur) / fps
            m, s = divmod(int(eta_sec), 60)
            h, m = divmod(m, 60)
            eta_str = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s"
        else:
            eta_str = "..."
        pct = int((cur / total) * 100) if total > 0 else 0
        self.status_message.emit(f"Exporting {pct}%  •  {cur}/{total} frames  •  {fps:.1f} fps  •  ETA {eta_str}")

    def _on_export_done(self) -> None:
        if self.export_worker is None:
            return
        worker = self.export_worker
        self.export_worker = None
        self._orphaned_workers.append(worker)
        worker.deleteLater()
        
        self.export_bar.setVisible(False)
        self._reset_export_btn()
        self._set_controls_enabled(True)
        self.status_message.emit("export complete")
        QMessageBox.information(self, "Done", "Video exported.")
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if hasattr(self, 'frame_slider'):
                self.frame_slider.blockSignals(True)
                self.frame_slider.setValue(0)
                self.frame_slider.blockSignals(False)
            self._seek_to_frame(0)

    def zoom_in(self)  -> None: self.canvas.zoom_in()
    def zoom_out(self) -> None: self.canvas.zoom_out()
    def fit(self)      -> None: self.canvas.fit()
    def actual(self)   -> None: self.canvas.actual()

    @property
    def zoom_level(self) -> float:
        return self.canvas.zoom_level

    def cleanup(self) -> None:
        self._play_timer.stop()
        if self._player is not None:
            self._player.stop()
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        if self._frame_worker is not None and self._frame_worker.isRunning():
            self._frame_worker.stop()
            self._frame_worker.wait(500)
        if self._proxy_worker is not None and self._proxy_worker.isRunning():
            self._proxy_worker.requestInterruption()
            self._proxy_worker.wait(500)
        if self.export_worker is not None and self.export_worker.isRunning():
            self.export_worker.stop()
            self.export_worker.wait(2000)