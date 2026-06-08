from __future__ import annotations

import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Signal, QThread, QMutex

from .dither_kernels import apply_dither
from .constants import _VIDEO_WORKERS
from .gpu_kernels import GPU_BACKEND, to_gpu, gpu_palette_batch
from .palettes import PALETTES

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

_GPU_EXPORT_BATCH = 8


def _ffmpeg_mux_audio(video_only_path: str, source_video_path: str, output_path: str) -> bool:
    """Mux audio from source_video_path into video_only_path, write to output_path.
    Returns True on success, False if ffmpeg unavailable or source has no audio."""
    try:
        import imageio_ffmpeg
        ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_cmd = "ffmpeg"
        
    try:
        subprocess.run(
            [ffmpeg_cmd, "-version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

    cmd = [
        ffmpeg_cmd,
        "-y",
        "-i", video_only_path,
        "-i", source_video_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        output_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except Exception:
        return False


class DitherWorker(QThread):
    result_ready = Signal(object)
    error    = Signal(str)

    def __init__(self, img, pixel_size, threshold, replace_color, method,
                 brightness, contrast, blur, sharpen,
                 glow_radius=0, glow_intensity=0, preview=False,
                 palette_name="B&W", custom_palette=None,
                 saturation=1.0, hue_rotate=0,
                 pre_denoise=0, pre_smooth=0,
                 post_denoise=0, post_smooth=0):
        super().__init__()
        self._img  = img; self._ps = pixel_size; self._t = threshold
        self._rc   = replace_color; self._m = method
        self._br   = brightness; self._co = contrast
        self._bl   = blur; self._sh = sharpen
        self._gr   = glow_radius; self._gi = glow_intensity
        self._prev = preview
        self._pal  = palette_name
        self._cpal = custom_palette
        self._sa   = saturation
        self._hu   = hue_rotate
        self._prd  = pre_denoise
        self._prs  = pre_smooth
        self._pod  = post_denoise
        self._pos  = post_smooth
        self._stop = False; self._mutex = QMutex()

    def run(self):
        self.setPriority(QThread.Priority.LowPriority)
        try:
            t0     = time.perf_counter()
            result = apply_dither(
                self._img, self._ps, self._t, self._rc, self._m,
                self._br, self._co, self._bl, self._sh,
                self._gr, self._gi, self._prev,
                self._pal, self._cpal,
                saturation=self._sa,
                hue_rotate=self._hu,
                pre_denoise=self._prd,
                pre_smooth=self._prs,
                post_denoise=self._pod,
                post_smooth=self._pos,
            )
            elapsed = time.perf_counter() - t0
            self._mutex.lock()
            ok = not self._stop
            self._mutex.unlock()
            if ok:
                self.result_ready.emit((result, elapsed, self._prev))
            else:
                self.result_ready.emit(None)
        except MemoryError:
            self.error.emit("Out of memory — try smaller image or larger pixel size.")
        except Exception as exc:
            self.error.emit(f"Processing error: {exc}")

    def stop(self):
        self._mutex.lock(); self._stop = True; self._mutex.unlock()


class FrameDitherWorker(QThread):
    result_ready = Signal(object)
    
    def __init__(self, img: Image.Image, params: dict, preview: bool = True, is_video: bool = False):
        super().__init__()
        self._img = img
        self._p = params
        self._preview = preview
        self._is_video = is_video
        self._stop = False
        self._mutex = QMutex()

    def run(self) -> None:
        self.setPriority(QThread.Priority.LowPriority)
        p = self._p
        try:
            result = apply_dither(
                self._img,
                p["pixel_size"], p["threshold"], p["color"], p["method"],
                p["brightness"], p["contrast"], p["blur"], p["sharpen"],
                p.get("glow_radius", 0), p.get("glow_intensity", 0),
                preview=self._preview,
                palette_name=p.get("palette_name", "B&W"),
                custom_palette=p.get("custom_palette"),
                saturation=p.get("saturation", 1.0),
                hue_rotate=p.get("hue_rotate", 0),
                pre_denoise=p.get("pre_denoise", 0),
                pre_smooth=p.get("pre_smooth", 0),
                post_denoise=p.get("post_denoise", 0),
                post_smooth=p.get("post_smooth", 0),
                is_video=self._is_video
            )
            self._mutex.lock()
            ok = not self._stop
            self._mutex.unlock()
            if ok:
                self.result_ready.emit(result)
            else:
                self.result_ready.emit(None)
        except Exception:
            import traceback
            traceback.print_exc()

    def stop(self) -> None:
        self._mutex.lock(); self._stop = True; self._mutex.unlock()


def _process_frame_worker(args):
    img, ps, t, rc, m, br, co, bl, sh, gr, gi, pal, cpal, sa, hu, prd, prs, pod, pos = args
    out = apply_dither(img, ps, t, rc, m, br, co, bl, sh, gr, gi,
                       palette_name=pal, custom_palette=cpal,
                       saturation=sa, hue_rotate=hu,
                       pre_denoise=prd, pre_smooth=prs,
                       post_denoise=pod, post_smooth=pos,
                       is_video=True)
    return out


def _resolve_palette_rgb(palette_name: str, custom_palette) -> np.ndarray | None:
    if custom_palette is not None:
        arr = np.array(custom_palette, dtype=np.uint8)
        return arr if arr.ndim == 2 and arr.shape[1] == 3 else None
    pal = PALETTES.get(palette_name)
    if pal is None or palette_name == "B&W":
        return None
    try:
        arr = np.array(pal, dtype=np.uint8)
        return arr if arr.ndim == 2 and arr.shape[1] == 3 else None
    except Exception:
        return None


class _VideoExportBase(QThread):
    error = Signal(str)

    def __init__(self, video_path, save_path, pixel_size, threshold,
                 replace_color, method, brightness, contrast, blur, sharpen,
                 glow_radius=0, glow_intensity=0,
                 palette_name="B&W", custom_palette=None,
                 saturation=1.0, hue_rotate=0,
                 pre_denoise=0, pre_smooth=0,
                 post_denoise=0, post_smooth=0,
                 include_audio=True):
        super().__init__()
        self._vp   = video_path;  self._sp  = save_path
        self._ps   = pixel_size;  self._t   = threshold
        self._rc   = replace_color; self._m = method
        self._br   = brightness;  self._co  = contrast
        self._bl   = blur;        self._sh  = sharpen
        self._gr   = glow_radius; self._gi  = glow_intensity
        self._pal  = palette_name
        self._cpal = custom_palette
        self._sa   = saturation
        self._hu   = hue_rotate
        self._prd  = pre_denoise
        self._prs  = pre_smooth
        self._pod  = post_denoise
        self._pos  = post_smooth
        self._include_audio = include_audio
        self._stop = False; self._mutex = QMutex()

    def _make_args(self, frames: list[Image.Image]) -> list:
        ps, t, rc, m   = self._ps, self._t, self._rc, self._m
        br, co, bl, sh = self._br, self._co, self._bl, self._sh
        gr, gi         = self._gr, self._gi
        pal, cpal      = self._pal, self._cpal
        sa, hu         = self._sa, self._hu
        prd, prs       = self._prd, self._prs
        pod, pos       = self._pod, self._pos
        return [
            (f, ps, t, rc, m, br, co, bl, sh, gr, gi,
             pal, cpal, sa, hu, prd, prs, pod, pos)
            for f in frames
        ]

    def _is_running(self) -> bool:
        self._mutex.lock()
        ok = not self._stop
        self._mutex.unlock()
        return ok

    def stop(self):
        self._mutex.lock(); self._stop = True; self._mutex.unlock()


class VideoExportWorker(_VideoExportBase):
    frame_ready = Signal(object)
    progress    = Signal(int, int)
    finished    = Signal()

    def run(self):
        if not _CV2:
            self.error.emit("opencv-python not installed."); return
        cap = out = None

        # If audio requested, write video to a temp file first, then mux.
        # Otherwise write directly to final path.
        if self._include_audio:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            import os; os.close(tmp_fd)
            write_path = tmp_path
        else:
            tmp_path   = None
            write_path = self._sp

        try:
            cap = cv2.VideoCapture(self._vp)
            if not cap.isOpened():
                self.error.emit("Failed to open video."); return

            fps   = cap.get(cv2.CAP_PROP_FPS) or 25.
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            BATCH = max(1, _VIDEO_WORKERS * 2)
            count = 0
            last_dithered = None
            frames_buf: list[Image.Image] = []

            with ThreadPoolExecutor(max_workers=_VIDEO_WORKERS) as executor:
                while self._is_running():
                    frames_buf.clear()
                    for _ in range(BATCH):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames_buf.append(
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    if not frames_buf:
                        break

                    dithered = list(executor.map(
                        _process_frame_worker, self._make_args(frames_buf)))

                    for dith in dithered:
                        if not self._is_running():
                            break
                        if out is None:
                            out = cv2.VideoWriter(
                                write_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps,
                                (dith.width, dith.height))
                        out.write(cv2.cvtColor(np.array(dith), cv2.COLOR_RGB2BGR))
                        count += 1
                        self.progress.emit(count, total)
                        last_dithered = dith

                    if last_dithered is not None and count % max(1, BATCH) == 0:
                        self.frame_ready.emit(last_dithered)

        except Exception as exc:
            self.error.emit(str(exc))
            return
        finally:
            if cap: cap.release()
            if out: out.release()

        # Mux audio from source into final output.
        if self._include_audio and tmp_path and self._is_running():
            muxed = _ffmpeg_mux_audio(tmp_path, self._vp, self._sp)
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
            if not muxed:
                # ffmpeg unavailable or no audio track — move temp to final path.
                try:
                    import shutil
                    shutil.move(tmp_path, self._sp)
                except Exception:
                    pass
        elif tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

        self.finished.emit()


class GifExportWorker(_VideoExportBase):
    frame_ready = Signal(object)
    progress    = Signal(int, int)
    finished    = Signal()

    def run(self):
        if not _CV2:
            self.error.emit("opencv-python not installed."); return
        cap = None
        try:
            cap = cv2.VideoCapture(self._vp)
            if not cap.isOpened():
                self.error.emit("Failed to open video."); return

            fps         = cap.get(cv2.CAP_PROP_FPS) or 25.
            total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_ms = max(20, int(1000 / fps))

            BATCH = max(1, _VIDEO_WORKERS * 2)

            frames: list[Image.Image] = []
            count = 0
            frames_buf: list[Image.Image] = []

            with ThreadPoolExecutor(max_workers=_VIDEO_WORKERS) as executor:
                while self._is_running():
                    frames_buf.clear()
                    for _ in range(BATCH):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames_buf.append(
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    if not frames_buf:
                        break

                    dithered = list(executor.map(
                        _process_frame_worker, self._make_args(frames_buf)))

                    for dith in dithered:
                        if not self._is_running():
                            break
                        frames.append(dith.convert("P", palette=Image.ADAPTIVE, colors=256))
                        count += 1
                        self.progress.emit(count, total)
                        if count % max(1, BATCH) == 0:
                            self.frame_ready.emit(dith)

            if frames and self._is_running():
                frames[0].save(
                    self._sp,
                    save_all=True,
                    append_images=frames[1:],
                    loop=0,
                    duration=duration_ms,
                    optimize=False,
                )
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if cap: cap.release()
