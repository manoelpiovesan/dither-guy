from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from PySide6.QtCore import Signal, QThread, QMutex

from .dither_kernels import apply_dither
from .constants import _VIDEO_WORKERS

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


class DitherWorker(QThread):
    finished = Signal(object)
    error    = Signal(str)

    def __init__(self, img, pixel_size, threshold, replace_color, method,
                 brightness, contrast, blur, sharpen,
                 glow_radius=0, glow_intensity=0, preview=False,
                 palette_name="B&W", custom_palette=None):
        super().__init__()
        self._img  = img; self._ps = pixel_size; self._t = threshold
        self._rc   = replace_color; self._m = method
        self._br   = brightness; self._co = contrast
        self._bl   = blur; self._sh = sharpen
        self._gr   = glow_radius; self._gi = glow_intensity
        self._prev = preview
        self._pal  = palette_name
        self._cpal = custom_palette
        self._stop = False; self._mutex = QMutex()
        # Lower thread priority so UI never stutters
        self.setPriority(QThread.Priority.LowPriority)

    def run(self):
        try:
            t0 = time.perf_counter()
            result = apply_dither(
                self._img, self._ps, self._t, self._rc, self._m,
                self._br, self._co, self._bl, self._sh,
                self._gr, self._gi, self._prev,
                self._pal, self._cpal)
            elapsed = time.perf_counter() - t0
            self._mutex.lock(); ok = not self._stop; self._mutex.unlock()
            if ok:
                self.finished.emit((result, elapsed, self._prev))
        except MemoryError:
            self.error.emit("Out of memory — try smaller image or larger pixel size.")
        except Exception as exc:
            self.error.emit(f"Processing error: {exc}")

    def stop(self):
        self._mutex.lock(); self._stop = True; self._mutex.unlock()


def _process_frame_worker(args):
    """Top-level worker function for parallel frame dithering."""
    frame_bytes, mode, size, ps, t, rc, m, br, co, bl, sh, gr, gi = args
    img = Image.frombytes(mode, size, frame_bytes)
    out = apply_dither(img, ps, t, rc, m, br, co, bl, sh, gr, gi)
    return out.tobytes(), out.mode, out.size


class VideoExportWorker(QThread):
    frame_ready = Signal(object)
    progress    = Signal(int, int)
    finished    = Signal()
    error       = Signal(str)

    def __init__(self, video_path, save_path, pixel_size, threshold,
                 replace_color, method, brightness, contrast, blur, sharpen,
                 glow_radius=0, glow_intensity=0):
        super().__init__()
        self._vp = video_path; self._sp = save_path; self._ps = pixel_size
        self._t  = threshold;  self._rc = replace_color; self._m = method
        self._br = brightness; self._co = contrast; self._bl = blur; self._sh = sharpen
        self._gr = glow_radius; self._gi = glow_intensity
        self._stop = False; self._mutex = QMutex()

    def _make_args(self, frames: list[Image.Image]):
        ps, t, rc, m   = self._ps, self._t, self._rc, self._m
        br, co, bl, sh = self._br, self._co, self._bl, self._sh
        gr, gi         = self._gr, self._gi
        return [
            (f.tobytes(), f.mode, f.size, ps, t, rc, m, br, co, bl, sh, gr, gi)
            for f in frames
        ]

    def run(self):
        if not _CV2:
            self.error.emit("opencv-python not installed."); return
        cap = out = None
        try:
            cap = cv2.VideoCapture(self._vp)
            if not cap.isOpened():
                self.error.emit("Failed to open video."); return

            fps   = cap.get(cv2.CAP_PROP_FPS) or 25.
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Use ThreadPoolExecutor (not ProcessPoolExecutor).
            # ProcessPoolExecutor uses the 'spawn' start method on Windows,
            # which re-runs the app entry point in each worker process and
            # causes new GUI windows to open during export (issue #7).
            # ThreadPoolExecutor is safe inside a QThread and avoids this.
            CHUNK = max(1, _VIDEO_WORKERS * 2)
            count = 0
            last_dithered = None
            out_size = None  # determined from first dithered frame
            frames_buf: list[Image.Image] = []

            with ThreadPoolExecutor(max_workers=_VIDEO_WORKERS) as executor:
                while True:
                    self._mutex.lock(); ok = not self._stop; self._mutex.unlock()
                    if not ok: break

                    frames_buf.clear()
                    for _ in range(CHUNK):
                        ret, frame = cap.read()
                        if not ret: break
                        frames_buf.append(
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    if not frames_buf: break

                    args = self._make_args(frames_buf)
                    raw_results = list(executor.map(_process_frame_worker, args))

                    dithered = [
                        Image.frombytes(mode, size, data)
                        for data, mode, size in raw_results
                    ]

                    for dith in dithered:
                        self._mutex.lock(); ok = not self._stop; self._mutex.unlock()
                        if not ok: break

                        # Initialise VideoWriter on the first dithered frame so
                        # that the output dimensions match the actual output size
                        # (which may differ from the source when pixel_size > 1).
                        if out is None:
                            out_size = (dith.width, dith.height)
                            out = cv2.VideoWriter(
                                self._sp,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps,
                                out_size)

                        out.write(cv2.cvtColor(np.array(dith), cv2.COLOR_RGB2BGR))
                        count += 1
                        self.progress.emit(count, total)
                        last_dithered = dith

                    if last_dithered is not None and count % max(1, CHUNK) == 0:
                        self.frame_ready.emit(last_dithered)

            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if cap: cap.release()
            if out: out.release()

    def stop(self):
        self._mutex.lock(); self._stop = True; self._mutex.unlock()
