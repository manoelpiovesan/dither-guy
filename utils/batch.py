from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from .dither_kernels import apply_dither
from .constants import _VIDEO_WORKERS

_BATCH_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def batch_process(
    input_folder: str,
    output_folder: str,
    params: dict,
    progress_cb=None,
    cancel_flag=None,
) -> tuple[int, int]:
    in_dir  = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    files   = [f for f in in_dir.iterdir() if f.suffix.lower() in _BATCH_IMAGE_EXTS]
    total   = len(files)
    success = 0
    errors  = 0

    def process_one(fp: Path):
        try:
            img = Image.open(fp).convert("RGB")
            result = apply_dither(
                img,
                params.get("pixel_size", 4),
                params.get("threshold", 128),
                tuple(params.get("color", (0, 255, 65))),
                params.get("method", "Floyd-Steinberg"),
                params.get("brightness", 1.0),
                params.get("contrast", 1.0),
                params.get("blur", 0),
                params.get("sharpen", 0),
                params.get("glow_radius", 0),
                params.get("glow_intensity", 0),
                preview=False,
                palette_name=params.get("palette_name", "B&W"),
                custom_palette=params.get("custom_palette"),
                saturation=params.get("saturation", 1.0),
                hue_rotate=params.get("hue_rotate", 0),
                pre_denoise=params.get("pre_denoise", 0),
                pre_smooth=params.get("pre_smooth", 0),
                post_denoise=params.get("post_denoise", 0),
                post_smooth=params.get("post_smooth", 0),
            )
            out_path = out_dir / (fp.stem + ".png")
            result.save(out_path)
            return True, fp.name
        except Exception as exc:
            return False, f"{fp.name}: {exc}"

    workers = min(_VIDEO_WORKERS, max(1, total))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, fp): fp for fp in files}
        for fut in as_completed(futures):
            if cancel_flag and cancel_flag[0]:
                ex.shutdown(wait=False, cancel_futures=True)
                break
            ok, msg = fut.result()
            done += 1
            if ok:
                success += 1
            else:
                errors += 1
            if progress_cb:
                progress_cb(done, total, msg)

    return success, errors
