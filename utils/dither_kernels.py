from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from .matrices import tile, ORDERED_MATRICES, _DOT_CLASS, _BAYER_4x4
from .palettes import PALETTES
from .gpu_kernels import (
    GPU_BACKEND,
    gpu_ordered_dither,
    gpu_palette_nearest,
    from_gpu,
    to_gpu,
)

try:
    import cv2 as _cv2
    _CV2 = True
except ImportError:
    _CV2 = False

# ---------------------------------------------------------------------------
# Optional JIT via numba
# ---------------------------------------------------------------------------

try:
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False

    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range

# ---------------------------------------------------------------------------
# Pre/post-processing helpers
# ---------------------------------------------------------------------------

def adjust(img: Image.Image, brightness: float, contrast: float,
           blur: float, sharpen: float) -> Image.Image:
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    for _ in range(int(sharpen)):
        img = img.filter(ImageFilter.SHARPEN)
    return img


def _apply_saturation(img: Image.Image, factor: float) -> Image.Image:
    if factor == 1.0:
        return img
    return ImageEnhance.Color(img.convert("RGB")).enhance(factor)


def _apply_hue_rotate(img: Image.Image, degrees: int) -> Image.Image:
    """Vectorised HSV hue rotation — no Python pixel loop."""
    if degrees == 0:
        return img
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    # RGB -> HSV (all NumPy)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(r)
    mask_r = (cmax == r) & (delta != 0)
    mask_g = (cmax == g) & (delta != 0)
    mask_b = (cmax == b) & (delta != 0)
    h[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360.0
    h[mask_g] = (60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120.0) % 360.0
    h[mask_b] = (60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240.0) % 360.0

    h = (h + degrees) % 360.0

    # Saturation + Value
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(cmax == 0, 0.0, delta / cmax)
    v = cmax

    # HSV -> RGB
    hi = (h / 60.0).astype(np.int32) % 6
    f  = h / 60.0 - np.floor(h / 60.0)
    p  = v * (1.0 - s)
    q  = v * (1.0 - f * s)
    t  = v * (1.0 - (1.0 - f) * s)

    out = np.empty_like(arr)
    for sector, (cr, cg, cb) in enumerate([
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ]):
        m = hi == sector
        out[..., 0][m] = cr[m]
        out[..., 1][m] = cg[m]
        out[..., 2][m] = cb[m]

    return Image.fromarray((out * 255.0).clip(0, 255).astype(np.uint8), "RGB")


def _apply_denoise(img: Image.Image, strength: int) -> Image.Image:
    """strength 1-5. Uses cv2.medianBlur when available (~10x faster)."""
    if strength <= 0:
        return img
    size = max(3, min(11, 2 * strength + 1))
    if size % 2 == 0:
        size += 1
    if _CV2:
        arr = np.array(img.convert("RGB"))
        arr = _cv2.medianBlur(arr, size)
        return Image.fromarray(arr, "RGB")
    return img.filter(ImageFilter.MedianFilter(size=size))


def _apply_smooth(img: Image.Image, strength: int) -> Image.Image:
    """strength 1-5 -> GaussianBlur radius 0.5-2.5."""
    if strength <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=strength * 0.5))


def apply_glow(img: Image.Image, radius: float, intensity: float) -> Image.Image:
    if radius <= 0 or intensity <= 0:
        return img
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    # In-place screen blend to prevent OOM on 4K: A + B - (A * B) / 255
    base = np.asarray(img, dtype=np.float32)
    glow = np.asarray(blurred, dtype=np.float32)
    glow *= (intensity / 100.0)
    
    out = np.copy(base)
    out += glow
    
    temp = base * glow
    temp *= (1.0 / 255.0)
    
    out -= temp
    np.clip(out, 0, 255, out=out)
    return Image.fromarray(out.astype(np.uint8))

# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

_SRGB_TO_XYZ = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
], dtype=np.float32)
_XYZ_SCALE = np.array([1.0 / 0.9505, 1.0, 1.0 / 1.089], dtype=np.float32)
_LAB_EPS   = np.float32(0.008856)
_LAB_KAPPA = np.float32(903.3)


def _rgb_to_lab_batch(rgb: np.ndarray) -> np.ndarray:
    r    = rgb.astype(np.float32) / np.float32(255.0)
    mask = r > np.float32(0.04045)
    r    = np.where(mask,
                    ((r + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4),
                    r / np.float32(12.92))
    orig_shape = r.shape
    flat = r.reshape(-1, 3) if r.ndim > 2 else r
    xyz  = (flat @ _SRGB_TO_XYZ.T) * _XYZ_SCALE
    xyz  = xyz.reshape(orig_shape)
    f    = np.where(xyz > _LAB_EPS,
                    np.cbrt(xyz),
                    (_LAB_KAPPA * xyz + np.float32(16.0)) / np.float32(116.0))
    L = np.float32(116.0) * f[..., 1] - np.float32(16.0)
    a = np.float32(500.0) * (f[..., 0] - f[..., 1])
    b = np.float32(200.0) * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


_rgb_to_lab = _rgb_to_lab_batch

_PAL_LAB_CACHE: dict[int, np.ndarray] = {}


def _get_pal_lab(pal: np.ndarray) -> np.ndarray:
    key = hash(pal.tobytes())
    if key not in _PAL_LAB_CACHE:
        _PAL_LAB_CACHE[key] = _rgb_to_lab_batch(pal)
    return _PAL_LAB_CACHE[key]


USE_GPU_THRESHOLD = 100_000


def _nearest_palette_indices(pixels, pal: np.ndarray,
                              pal_lab: np.ndarray,
                              on_device: bool = False) -> np.ndarray:
    if on_device:
        return gpu_palette_nearest(pixels, pal_lab)
    pix_lab = _rgb_to_lab_batch(np.asarray(pixels))
    pix_sq = np.sum(pix_lab**2, axis=-1, keepdims=True)
    pal_sq = np.sum(pal_lab**2, axis=-1)
    dot    = pix_lab @ pal_lab.T
    dists  = pix_sq + pal_sq - 2 * dot
    return np.argmin(dists, axis=1)


def _nearest_palette_indices_rgb(pixels: np.ndarray, pal: np.ndarray) -> np.ndarray:
    pix_sq = np.sum(pixels**2, axis=-1, keepdims=True)
    pal_sq = np.sum(pal**2, axis=-1)
    dot    = pixels @ pal.T
    dists  = pix_sq + pal_sq - 2 * dot
    return np.argmin(dists, axis=1)


# ---------------------------------------------------------------------------
# Error-diffusion coefficient tables
# ---------------------------------------------------------------------------

_COEFF_TABLES: dict[str, list[tuple]] = {
    "Floyd-Steinberg":     [(0, 1, 7/16), (1,-1, 3/16), (1, 0, 5/16), (1, 1, 1/16)],
    "Atkinson":            [(0, 1, 1/8),  (0, 2, 1/8),  (1,-1, 1/8),  (1, 0, 1/8),
                            (1, 1, 1/8),  (2, 0, 1/8)],
    "Sierra":              [(0, 1, 5/32), (0, 2, 3/32), (1,-2, 2/32), (1,-1, 4/32),
                            (1, 0, 5/32), (1, 1, 4/32), (1, 2, 2/32), (2,-1, 2/32),
                            (2, 0, 3/32), (2, 1, 2/32)],
    "Burkes":              [(0, 1, 8/32), (0, 2, 4/32), (1,-2, 2/32), (1,-1, 4/32),
                            (1, 0, 8/32), (1, 1, 4/32), (1, 2, 2/32)],
    "Stucki":              [(0, 1, 8/42), (0, 2, 4/42), (1,-2, 2/42), (1,-1, 4/42),
                            (1, 0, 8/42), (1, 1, 4/42), (1, 2, 2/42), (2,-2, 1/42),
                            (2,-1, 2/42), (2, 0, 4/42), (2, 1, 2/42), (2, 2, 1/42)],
    "Jarvis-Judice-Ninke": [(0, 1, 7/48), (0, 2, 5/48), (1,-2, 3/48), (1,-1, 5/48),
                            (1, 0, 7/48), (1, 1, 5/48), (1, 2, 3/48), (2,-2, 1/48),
                            (2,-1, 3/48), (2, 0, 5/48), (2, 1, 3/48), (2, 2, 1/48)],
    "Sierra-Lite":         [(0, 1, 2/4),  (1,-1, 1/4),  (1, 0, 1/4)],
    "Nakano":              [(0, 1, 8/24), (1,-1, 4/24),  (1, 0, 4/24),  (1, 1, 4/24),
                            (2,-2, 1/24), (2,-1, 2/24),  (2, 0, 1/24)],
}


# ---------------------------------------------------------------------------
# Core colour error-diffusion
# Optimised: Lab conversion done once for the whole image up-front,
# then per-row nearest lookup uses the pre-computed pixel Lab values.
# ---------------------------------------------------------------------------

def _palette_ed_vectorised(
    arr: np.ndarray,
    pal: np.ndarray,
    pal_lab: np.ndarray,
    coeffs: list[tuple],
    on_device: bool = False,
) -> np.ndarray:
    on_device = False  # Disable row-by-row GPU transfers (PCI-E bottleneck)
    h, w, _ = arr.shape
    out = arr.copy()

    for y in range(h):
        row  = out[y]                      # (W, 3) float32, already modified by prev rows
        if on_device:
            idxs = gpu_palette_nearest(to_gpu(row), pal_lab)
        else:
            pix_lab = _rgb_to_lab_batch(row)   # (W, 3)
            pix_sq  = np.sum(pix_lab**2, axis=-1, keepdims=True)
            pal_sq  = np.sum(pal_lab**2, axis=-1)
            dot     = pix_lab @ pal_lab.T
            dists   = pix_sq + pal_sq - 2 * dot
            idxs    = np.argmin(dists, axis=1)                               # (W,)
        snapped = pal[idxs]                # (W, 3)
        err     = row - snapped            # (W, 3)
        out[y]  = snapped

        for dy, dx, wt in coeffs:
            ny = y + dy
            if ny >= h:
                continue
            if dx > 0:
                if w > dx:
                    out[ny, dx:] = np.clip(out[ny, dx:] + err[:w - dx] * wt, 0, 255)
            elif dx < 0:
                adx = -dx
                if w > adx:
                    out[ny, :w - adx] = np.clip(out[ny, :w - adx] + err[adx:] * wt, 0, 255)
            else:
                out[ny] = np.clip(out[ny] + err * wt, 0, 255)

    return np.clip(np.asarray(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public colour dither entry points
# ---------------------------------------------------------------------------

def palette_dither(image: Image.Image, palette: list[tuple],
                   method: str = "Floyd-Steinberg",
                   threshold: float = 128.0,
                   use_gpu: bool = False) -> Image.Image:
    arr     = np.array(image.convert("RGB"), dtype=np.float32)
    pal     = np.asarray(palette, dtype=np.float32)
    pal_lab = _get_pal_lab(pal)
    coeffs  = _COEFF_TABLES.get(method, _COEFF_TABLES["Floyd-Steinberg"])
    result  = _palette_ed_vectorised(arr, pal, pal_lab, coeffs, on_device=use_gpu)
    return Image.fromarray(result, mode="RGB")


def palette_dither_ordered(image: Image.Image, palette: list[tuple], method: str, use_gpu: bool = False) -> Image.Image:
    arr  = np.array(image.convert("RGB"), dtype=np.float32)
    pal  = np.asarray(palette, dtype=np.float32)
    h, w = arr.shape[:2]

    from .matrices import ORDERED_MATRICES, _BAYER_4x4
    mat = ORDERED_MATRICES.get(method, _BAYER_4x4)
    bayer = tile(mat, h, w)
    noise = (bayer - 128.0) * 0.3
    noisy = np.clip(arr + noise[:, :, np.newaxis], 0, 255)

    if use_gpu:
        try:
            from .gpu_kernels import gpu_palette_batch, to_gpu
            # Add batch dimension and process
            result = gpu_palette_batch(to_gpu(noisy[np.newaxis, ...]), pal)[0]
            return Image.fromarray(result, mode="RGB")
        except Exception as e:
            import traceback
            traceback.print_exc()

    flat  = noisy.reshape(-1, 3)
    idxs  = _nearest_palette_indices_rgb(flat, pal)
    result = pal[idxs].reshape(h, w, 3)
    return Image.fromarray(result.astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# B&W error-diffusion kernels (JIT / NumPy fallbacks)
# ---------------------------------------------------------------------------

if _NUMBA:
    @njit(cache=True, fastmath=True, nogil=True)
    def _fs_core(a, t):
        h, w = a.shape
        for y in range(h):
            for x in range(w):
                old = a[y, x]; new = 255.0 if old > t else 0.0; e = old - new
                a[y, x] = new
                if x+1 < w: a[y, x+1] += e * 0.4375
                if y+1 < h:
                    if x > 0: a[y+1, x-1] += e * 0.1875
                    a[y+1, x] += e * 0.3125
                    if x+1 < w: a[y+1, x+1] += e * 0.0625
        return a

    @njit(cache=True, fastmath=True, nogil=True)
    def _atkinson_core(a, t):
        h, w = a.shape
        for y in range(h):
            for x in range(w):
                old = a[y, x]; new = 255.0 if old > t else 0.0; e = (old - new) * 0.125
                a[y, x] = new
                if x+1 < w: a[y, x+1] += e
                if x+2 < w: a[y, x+2] += e
                if y+1 < h:
                    if x > 0: a[y+1, x-1] += e
                    a[y+1, x] += e
                    if x+1 < w: a[y+1, x+1] += e
                if y+2 < h: a[y+2, x] += e
        return a

    @njit(cache=True, fastmath=True, nogil=True)
    def _jjn_core(a, t):
        h, w = a.shape; d = 48.0
        for y in range(h):
            for x in range(w):
                old = a[y, x]; new = 255.0 if old > t else 0.0; e = old - new
                a[y, x] = new
                if x+1<w: a[y,x+1]+=e*7/d
                if x+2<w: a[y,x+2]+=e*5/d
                if y+1<h:
                    if x>1: a[y+1,x-2]+=e*3/d
                    if x>0: a[y+1,x-1]+=e*5/d
                    a[y+1,x]+=e*7/d
                    if x+1<w: a[y+1,x+1]+=e*5/d
                    if x+2<w: a[y+1,x+2]+=e*3/d
                if y+2<h:
                    if x>1: a[y+2,x-2]+=e*1/d
                    if x>0: a[y+2,x-1]+=e*3/d
                    a[y+2,x]+=e*5/d
                    if x+1<w: a[y+2,x+1]+=e*3/d
                    if x+2<w: a[y+2,x+2]+=e*1/d
        return a

    @njit(cache=True, fastmath=True, nogil=True)
    def _stucki_core(a, t):
        h, w = a.shape; d = 42.0
        for y in range(h):
            for x in range(w):
                old = a[y, x]; new = 255.0 if old > t else 0.0; e = old - new
                a[y, x] = new
                if x+1<w: a[y,x+1]+=e*8/d
                if x+2<w: a[y,x+2]+=e*4/d
                if y+1<h:
                    if x>1: a[y+1,x-2]+=e*2/d
                    if x>0: a[y+1,x-1]+=e*4/d
                    a[y+1,x]+=e*8/d
                    if x+1<w: a[y+1,x+1]+=e*4/d
                    if x+2<w: a[y+1,x+2]+=e*2/d
                if y+2<h:
                    if x>1: a[y+2,x-2]+=e*1/d
                    if x>0: a[y+2,x-1]+=e*2/d
                    a[y+2,x]+=e*4/d
                    if x+1<w: a[y+2,x+1]+=e*2/d
                    if x+2<w: a[y+2,x+2]+=e*1/d
        return a

    @njit(cache=True, fastmath=True, nogil=True)
    def _sierra_core(a, t):
        h, w = a.shape; d = 32.0
        for y in range(h):
            for x in range(w):
                old = a[y, x]; new = 255.0 if old > t else 0.0; e = old - new
                a[y, x] = new
                if x+1<w: a[y,x+1]+=e*5/d
                if x+2<w: a[y,x+2]+=e*3/d
                if y+1<h:
                    if x>1: a[y+1,x-2]+=e*2/d
                    if x>0: a[y+1,x-1]+=e*4/d
                    a[y+1,x]+=e*5/d
                    if x+1<w: a[y+1,x+1]+=e*4/d
                    if x+2<w: a[y+1,x+2]+=e*2/d
                if y+2<h:
                    if x>0: a[y+2,x-1]+=e*2/d
                    a[y+2,x]+=e*3/d
                    if x+1<w: a[y+2,x+1]+=e*2/d
        return a

    @njit(cache=True, fastmath=True, nogil=True)
    def _sierra_lite_core(a, t):
        h, w = a.shape
        for y in range(h):
            for x in range(w):
                old = a[y, x]; new = 255.0 if old > t else 0.0; e = old - new
                a[y, x] = new
                if x+1<w: a[y,x+1]+=e*0.5
                if y+1<h:
                    if x>0: a[y+1,x-1]+=e*0.25
                    a[y+1,x]+=e*0.25
        return a

    @njit(cache=True, fastmath=True, nogil=True)
    def _nakano_core(a, t):
        h, w = a.shape
        for y in range(h):
            for x in range(w):
                old = a[y, x]; new = 255.0 if old > t else 0.0; e = old - new
                a[y, x] = new
                if x+1<w: a[y,x+1]+=e*(8/24)
                if y+1<h:
                    if x>0: a[y+1,x-1]+=e*(4/24)
                    a[y+1,x]+=e*(4/24)
                    if x+1<w: a[y+1,x+1]+=e*(4/24)
                if y+2<h:
                    if x>1: a[y+2,x-2]+=e*(1/24)
                    if x>0: a[y+2,x-1]+=e*(2/24)
                    a[y+2,x]+=e*(1/24)
        return a

    def _wrap_jit(fn):
        def inner(a: np.ndarray, t: float) -> np.ndarray:
            a = a.astype(np.float64)
            np.clip(a, 0, 255, out=a)
            result = fn(a, float(t))
            return np.clip(result, 0, 255).astype(np.uint8)
        return inner

    _fs_vectorised          = _wrap_jit(_fs_core)
    _atkinson_vectorised    = _wrap_jit(_atkinson_core)
    _sierra_vectorised      = _wrap_jit(_sierra_core)
    _sierra_lite_vectorised = _wrap_jit(_sierra_lite_core)
    _nakano_vectorised      = _wrap_jit(_nakano_core)
    _jjn_vectorised         = _wrap_jit(_jjn_core)
    _stucki_vectorised      = _wrap_jit(_stucki_core)

else:
    def _fs_vectorised(a: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape
        for y in range(h):
            row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
            a[y, 1:] = np.clip(a[y, 1:] + e[:-1] * 0.4375, 0, 255)
            if y+1 < h:
                a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:] * 0.1875, 0, 255)
                a[y+1]      = np.clip(a[y+1]       + e    * 0.3125,  0, 255)
                if w > 1: a[y+1, 1:] = np.clip(a[y+1, 1:] + e[:-1] * 0.0625, 0, 255)
            a[y] = nw
        return np.clip(a, 0, 255).astype(np.uint8)

    def _atkinson_vectorised(a: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape
        for y in range(h):
            row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = (row - nw) * 0.125
            a[y, 1:] = np.clip(a[y, 1:] + e[:-1], 0, 255)
            if w > 2: a[y, 2:] = np.clip(a[y, 2:] + e[:-2], 0, 255)
            if y+1 < h:
                a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:], 0, 255)
                a[y+1]      = np.clip(a[y+1]       + e,    0, 255)
                if w > 1: a[y+1, 1:] = np.clip(a[y+1, 1:] + e[:-1], 0, 255)
            if y+2 < h: a[y+2] = np.clip(a[y+2] + e, 0, 255)
            a[y] = nw
        return np.clip(a, 0, 255).astype(np.uint8)

    def _sierra_vectorised(a: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape; d = 32.
        for y in range(h):
            row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
            if w > 1: a[y, 1:]    = np.clip(a[y, 1:]    + e[:-1] * (5/d), 0, 255)
            if w > 2: a[y, 2:]    = np.clip(a[y, 2:]    + e[:-2] * (3/d), 0, 255)
            if y+1 < h:
                if w > 2: a[y+1, :-2] = np.clip(a[y+1, :-2] + e[2:] * (2/d), 0, 255)
                if w > 1: a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:] * (4/d), 0, 255)
                a[y+1]             = np.clip(a[y+1]           + e     * (5/d), 0, 255)
                if w > 1: a[y+1, 1:]  = np.clip(a[y+1, 1:]  + e[:-1] * (4/d), 0, 255)
                if w > 2: a[y+1, 2:]  = np.clip(a[y+1, 2:]  + e[:-2] * (2/d), 0, 255)
            if y+2 < h:
                if w > 1: a[y+2, :-1] = np.clip(a[y+2, :-1] + e[1:] * (2/d), 0, 255)
                a[y+2]             = np.clip(a[y+2]           + e     * (3/d), 0, 255)
                if w > 1: a[y+2, 1:]  = np.clip(a[y+2, 1:]  + e[:-1] * (2/d), 0, 255)
            a[y] = nw
        return np.clip(a, 0, 255).astype(np.uint8)

    def _sierra_lite_vectorised(a: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape
        for y in range(h):
            row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
            a[y, 1:] = np.clip(a[y, 1:] + e[:-1] * 0.5, 0, 255)
            if y+1 < h:
                if w > 1: a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:] * 0.25, 0, 255)
                a[y+1] = np.clip(a[y+1] + e * 0.25, 0, 255)
            a[y] = nw
        return np.clip(a, 0, 255).astype(np.uint8)

    def _nakano_vectorised(a: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape
        _w83 = 8/24; _w43 = 4/24; _w13 = 1/24; _w23 = 2/24
        for y in range(h):
            row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
            a[y, 1:] = np.clip(a[y, 1:] + e[:-1] * _w83, 0, 255)
            if y+1 < h:
                if w > 1: a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:] * _w43, 0, 255)
                a[y+1]             = np.clip(a[y+1]           + e     * _w43, 0, 255)
                if w > 1: a[y+1, 1:]  = np.clip(a[y+1, 1:]  + e[:-1] * _w43, 0, 255)
            if y+2 < h:
                if w > 2: a[y+2, :-2] = np.clip(a[y+2, :-2] + e[2:] * _w13, 0, 255)
                if w > 1: a[y+2, :-1] = np.clip(a[y+2, :-1] + e[1:] * _w23, 0, 255)
                a[y+2]             = np.clip(a[y+2]           + e     * _w13, 0, 255)
            a[y] = nw
        return np.clip(a, 0, 255).astype(np.uint8)

    def _jjn_vectorised(a: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape; d = 48.
        _w7 = 7/d; _w5 = 5/d; _w3 = 3/d; _w1 = 1/d
        for y in range(h):
            row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
            if w > 1: a[y, 1:]    = np.clip(a[y, 1:]    + e[:-1] * _w7, 0, 255)
            if w > 2: a[y, 2:]    = np.clip(a[y, 2:]    + e[:-2] * _w5, 0, 255)
            if y+1 < h:
                if w > 2: a[y+1, :-2] = np.clip(a[y+1, :-2] + e[2:] * _w3, 0, 255)
                if w > 1: a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:] * _w5, 0, 255)
                a[y+1]             = np.clip(a[y+1]           + e     * _w7, 0, 255)
                if w > 1: a[y+1, 1:]  = np.clip(a[y+1, 1:]  + e[:-1] * _w5, 0, 255)
                if w > 2: a[y+1, 2:]  = np.clip(a[y+1, 2:]  + e[:-2] * _w3, 0, 255)
            if y+2 < h:
                if w > 2: a[y+2, :-2] = np.clip(a[y+2, :-2] + e[2:] * _w1, 0, 255)
                if w > 1: a[y+2, :-1] = np.clip(a[y+2, :-1] + e[1:] * _w3, 0, 255)
                a[y+2]             = np.clip(a[y+2]           + e     * _w5, 0, 255)
                if w > 1: a[y+2, 1:]  = np.clip(a[y+2, 1:]  + e[:-1] * _w3, 0, 255)
                if w > 2: a[y+2, 2:]  = np.clip(a[y+2, 2:]  + e[:-2] * _w1, 0, 255)
            a[y] = nw
        return np.clip(a, 0, 255).astype(np.uint8)

    def _stucki_vectorised(a: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape; d = 42.
        _w8 = 8/d; _w4 = 4/d; _w2 = 2/d; _w1 = 1/d
        for y in range(h):
            row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
            if w > 1: a[y, 1:]    = np.clip(a[y, 1:]    + e[:-1] * _w8, 0, 255)
            if w > 2: a[y, 2:]    = np.clip(a[y, 2:]    + e[:-2] * _w4, 0, 255)
            if y+1 < h:
                if w > 2: a[y+1, :-2] = np.clip(a[y+1, :-2] + e[2:] * _w2, 0, 255)
                if w > 1: a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:] * _w4, 0, 255)
                a[y+1]             = np.clip(a[y+1]           + e     * _w8, 0, 255)
                if w > 1: a[y+1, 1:]  = np.clip(a[y+1, 1:]  + e[:-1] * _w4, 0, 255)
                if w > 2: a[y+1, 2:]  = np.clip(a[y+1, 2:]  + e[:-2] * _w2, 0, 255)
            a[y] = nw
        return np.clip(a, 0, 255).astype(np.uint8)


def _variable_error_vectorised(a: np.ndarray, t: float) -> np.ndarray:
    h, w = a.shape
    for y in range(h - 1):
        row = a[y, 1:w-1].copy(); nw = np.where(row > t, 255., 0.)
        e = row - nw; f = row / 255.
        a[y, 2:w]     = np.clip(a[y, 2:w]     + e * 7. * f        / 16., 0, 255)
        a[y+1, 0:w-2] = np.clip(a[y+1, 0:w-2] + e * 3. * (1. - f) / 16., 0, 255)
        a[y+1, 1:w-1] = np.clip(a[y+1, 1:w-1] + e * 5.             / 16., 0, 255)
        a[y+1, 2:w]   = np.clip(a[y+1, 2:w]   + e                  / 16., 0, 255)
        a[y, 1:w-1] = nw
    return np.clip(a, 0, 255).astype(np.uint8)


def _burkes_vectorised(a: np.ndarray, t: float) -> np.ndarray:
    h, w = a.shape; d = 32.0
    _w8 = 8/d; _w4 = 4/d; _w2 = 2/d
    for y in range(h):
        row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
        if w > 1: a[y, 1:]    = np.clip(a[y, 1:]    + e[:-1] * _w8, 0, 255)
        if w > 2: a[y, 2:]    = np.clip(a[y, 2:]    + e[:-2] * _w4, 0, 255)
        if y+1 < h:
            if w > 2: a[y+1, :-2] = np.clip(a[y+1, :-2] + e[2:]  * _w2, 0, 255)
            if w > 1: a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:]  * _w4, 0, 255)
            a[y+1]             = np.clip(a[y+1]           + e      * _w8, 0, 255)
            if w > 1: a[y+1, 1:]  = np.clip(a[y+1, 1:]  + e[:-1] * _w4, 0, 255)
            if w > 2: a[y+1, 2:]  = np.clip(a[y+1, 2:]  + e[:-2] * _w2, 0, 255)
        a[y] = nw
    return np.clip(a, 0, 255).astype(np.uint8)


def _stevenson_arce_vectorised(a: np.ndarray, t: float) -> np.ndarray:
    h, w = a.shape; d = 200.0
    coeffs = [
        (0,  2, 32/d),
        (1, -3, 12/d), (1, -1, 26/d), (1,  1, 30/d), (1, 3, 16/d),
        (2, -2, 12/d), (2,  0, 26/d), (2,  2, 12/d),
        (3, -3,  5/d), (3, -1, 12/d), (3,  1, 12/d), (3, 3,  5/d),
    ]
    for y in range(h):
        row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
        for dy, dx, wt in coeffs:
            ny = y + dy
            if ny >= h:
                continue
            if dx > 0:
                if w > dx:
                    a[ny, dx:] = np.clip(a[ny, dx:] + e[:w - dx] * wt, 0, 255)
            elif dx < 0:
                adx = -dx
                if w > adx:
                    a[ny, :w - adx] = np.clip(a[ny, :w - adx] + e[adx:] * wt, 0, 255)
            else:
                a[ny] = np.clip(a[ny] + e * wt, 0, 255)
        a[y] = nw
    return np.clip(a, 0, 255).astype(np.uint8)


_OSTROMOUKHOV_TABLE = np.array([
    (13, 0,  5, 18), (13, 0,  5, 18), (21, 0, 10, 31), ( 7, 0,  4, 11),
    ( 8, 0,  5, 13), (47, 3, 28, 78), (23, 3, 13, 39), (15, 3,  8, 26),
    (22, 5, 10, 37), (56,14, 21, 91), (28, 8,  9, 45), (19, 6,  5, 30),
    (14, 5,  3, 22), ( 7, 3,  1, 11), (65,32,  7,104), (23,12,  2, 37),
    (23,12,  2, 37), (65,32,  7,104), ( 7, 3,  1, 11), (14, 5,  3, 22),
    (19, 6,  5, 30), (28, 8,  9, 45), (56,14, 21, 91), (22, 5, 10, 37),
    (15, 3,  8, 26), (23, 3, 13, 39), (47, 3, 28, 78), ( 8, 0,  5, 13),
    ( 7, 0,  4, 11), (21, 0, 10, 31), (13, 0,  5, 18), (13, 0,  5, 18),
], dtype=np.float32)


def _ostromoukhov_vectorised(a: np.ndarray, t: float) -> np.ndarray:
    h, w = a.shape
    tbl  = _OSTROMOUKHOV_TABLE
    for y in range(h):
        for x in range(w):
            old = a[y, x]; new = 255.0 if old > t else 0.0; err = old - new
            a[y, x] = new
            band = int(np.clip(old * (1.0 / 8.0), 0, 31))
            c0, c1, c2, dn = tbl[band]
            if dn == 0:
                continue
            inv_dn = 1.0 / dn
            if x+1 < w:   a[y,   x+1] = np.clip(a[y,   x+1] + err * c0 * inv_dn, 0, 255)
            if y+1 < h:
                if x > 0: a[y+1, x-1] = np.clip(a[y+1, x-1] + err * c1 * inv_dn, 0, 255)
                a[y+1, x]             = np.clip(a[y+1, x]     + err * c2 * inv_dn, 0, 255)
    return np.clip(a, 0, 255).astype(np.uint8)


_BLUE_NOISE_MASK_64 = None


def _get_blue_noise_mask(h: int, w: int) -> np.ndarray:
    global _BLUE_NOISE_MASK_64
    if _BLUE_NOISE_MASK_64 is None:
        rng  = np.random.default_rng(0xD1740)
        base = rng.integers(0, 256, (64, 64), dtype=np.uint8).astype(np.float32)
        blurred = np.array(
            Image.fromarray(base.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=4)),
            dtype=np.float32)
        _BLUE_NOISE_MASK_64 = np.clip(base - blurred + 128, 0, 255)
    return tile(_BLUE_NOISE_MASK_64, h, w)


def _blue_noise_mask_vectorised(a: np.ndarray, t: float) -> np.ndarray:
    h, w = a.shape
    mask = _get_blue_noise_mask(h, w)
    return np.where(a + (mask - 128.0) * (1.0 - t / 255.0) > t, 255, 0).astype(np.uint8)


def _dot_diffusion_vectorised(a: np.ndarray, t: float) -> np.ndarray:
    h, w = a.shape
    dc   = tile(_DOT_CLASS, h, w)
    for y in range(h):
        row = a[y].copy(); nw = np.where(row > t, 255., 0.)
        err = row - nw; cm = dc[y]
        if w > 1:
            a[y, 1:] = np.clip(a[y, 1:] + err[:-1] / (cm[:-1] + 1.), 0, 255)
        a[y] = nw
    return np.clip(a, 0, 255).astype(np.uint8)


def _riemersma_vectorised(a: np.ndarray, t: float) -> np.ndarray:
    h, w = a.shape
    buf  = [0.] * 16
    decay = 0.0625
    for y in range(h):
        xs = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        for x in xs:
            old = float(a[y, x]) + buf[0]
            new = 255. if old > t else 0.
            a[y, x] = new
            buf = buf[1:] + [(old - new) * decay]
    return np.clip(a, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Replace-colour: vectorised
# ---------------------------------------------------------------------------

_WHITE = np.array([255, 255, 255], dtype=np.uint8)


def _apply_replace_color(data: np.ndarray, replace_color: tuple) -> np.ndarray:
    rc = np.array(replace_color, dtype=np.uint8)
    if np.array_equal(rc, _WHITE):
        return data
    mask = np.all(data == _WHITE, axis=-1)
    data[mask] = rc
    return data


# ---------------------------------------------------------------------------
# Main dither pipeline
# ---------------------------------------------------------------------------

_BW_DISPATCH: dict[str, object] = {}


def _build_dispatch():
    global _BW_DISPATCH
    _BW_DISPATCH = {
        "Floyd-Steinberg":     _fs_vectorised,
        "Atkinson":            _atkinson_vectorised,
        "Sierra":              _sierra_vectorised,
        "Sierra-Lite":         _sierra_lite_vectorised,
        "Nakano":              _nakano_vectorised,
        "Jarvis-Judice-Ninke": _jjn_vectorised,
        "Stucki":              _stucki_vectorised,
        "Burkes":              _burkes_vectorised,
        "Stevenson-Arce":      _stevenson_arce_vectorised,
        "Ostromoukhov":        _ostromoukhov_vectorised,
        "Variable-Error":      _variable_error_vectorised,
        "Dot-Diffusion":       _dot_diffusion_vectorised,
        "Riemersma":           _riemersma_vectorised,
    }


_build_dispatch()


def apply_dither(
    img: Image.Image,
    pixel_size: int,
    threshold: float,
    replace_color: tuple,
    method: str,
    brightness: float = 1.0,
    contrast:   float = 1.0,
    blur:       float = 0,
    sharpen:    float = 0,
    glow_radius:    float = 0,
    glow_intensity: float = 0,
    preview: bool = False,
    palette_name: str = "B&W",
    custom_palette: Optional[list] = None,
    saturation:   float = 1.0,
    hue_rotate:   int   = 0,
    pre_denoise:  int   = 0,
    pre_smooth:   int   = 0,
    post_denoise: int   = 0,
    post_smooth:  int   = 0,
    is_video: bool = False,
) -> Image.Image:
    # --- pre-dither adjustments ---
    img = adjust(img, brightness, contrast, blur, sharpen)
    if saturation != 1.0:
        img = _apply_saturation(img, saturation)
    if hue_rotate:
        img = _apply_hue_rotate(img, hue_rotate)
    if pre_denoise:
        img = _apply_denoise(img, pre_denoise)
    if pre_smooth:
        img = _apply_smooth(img, pre_smooth)

    palette  = custom_palette if (custom_palette and len(custom_palette) >= 2) \
               else PALETTES.get(palette_name, PALETTES["B&W"])
    is_bw    = (palette == PALETTES["B&W"])
    if is_video:
        video_scale = max(img.width, img.height) / 720.0
        effective_pixel = max(1, round(pixel_size * video_scale))
        blur *= video_scale
        sharpen *= video_scale
        glow_radius = round(glow_radius * video_scale)
        pre_smooth *= video_scale
        post_smooth *= video_scale
    else:
        effective_pixel = max(1, pixel_size * (2 if preview else 1))

    use_gpu = (
        GPU_BACKEND == "cuda" and
        img.width * img.height > USE_GPU_THRESHOLD
    )

    if not is_bw:
        rgb = img.convert("RGB")
        sw  = max(1, rgb.width  // effective_pixel)
        sh  = max(1, rgb.height // effective_pixel)
        rgb = rgb.resize((sw, sh), Image.NEAREST)

        from .constants import METHOD_GROUPS
        is_ordered = method in METHOD_GROUPS.get("Ordered", [])
        
        if is_ordered:
            result = palette_dither_ordered(rgb, palette, method=method, use_gpu=use_gpu)
        else:
            result = palette_dither(rgb, palette, method=method,
                                    threshold=threshold, use_gpu=use_gpu)

        result = result.resize((sw * effective_pixel, sh * effective_pixel), Image.NEAREST)
        data   = np.array(result)
        data   = _apply_replace_color(data, replace_color)
        result = Image.fromarray(data)
        if glow_radius > 0 and glow_intensity > 0:
            result = apply_glow(result, glow_radius, glow_intensity)
        if post_denoise:
            result = _apply_denoise(result, post_denoise)
        if post_smooth:
            result = _apply_smooth(result, post_smooth)
        return result

    # -- B&W path --
    img = img.convert('L')
    sw  = max(1, img.width  // effective_pixel)
    sh  = max(1, img.height // effective_pixel)
    img = img.resize((sw, sh), Image.NEAREST)
    a   = np.array(img, dtype=np.float32)
    h, w = a.shape
    t    = float(threshold)

    if use_gpu:
        a = to_gpu(a)

    if method in ORDERED_MATRICES:
        tiled = tile(ORDERED_MATRICES[method], h, w)
        if use_gpu:
            tiled = to_gpu(tiled.astype(np.float32))
            a = gpu_ordered_dither(a, tiled, t)
            a = from_gpu(a)
        else:
            a = np.where(a.astype(np.float32) + (tiled.astype(np.float32) - 128.0) * (1.0 - t / 255.0) > t, 255, 0).astype(np.uint8)
    elif method == "Crosshatch":
        xs  = np.arange(w, dtype=np.float32)
        ys  = np.arange(h, dtype=np.float32)
        ch  = (np.sin(xs[None, :] * 0.5) + np.sin(ys[:, None] * 0.5)) * 64. + 128.
        if use_gpu:
            ch = to_gpu(ch)
            a = gpu_ordered_dither(a, ch, t)
            a = from_gpu(a)
        else:
            a = np.where(a.astype(np.float32) + (ch.astype(np.float32) - 128.0) * (1.0 - t / 255.0) > t, 255, 0).astype(np.uint8)
    elif method == "Blue-Noise Mask":
        mask = _get_blue_noise_mask(h, w)
        if use_gpu:
            mask = to_gpu(mask)
            a = gpu_ordered_dither(a, mask, t)
            a = from_gpu(a)
        else:
            a = np.where(a.astype(np.float32) + (mask.astype(np.float32) - 128.0) * (1.0 - t / 255.0) > t, 255, 0).astype(np.uint8)
    elif method in _BW_DISPATCH:
        if use_gpu:
            a = from_gpu(a)
        a = _BW_DISPATCH[method](a, t)
    else:
        if use_gpu:
            a = from_gpu(a)
        a = np.where(a > t, 255, 0).astype(np.uint8)

    a = np.asarray(from_gpu(a))

    img  = Image.fromarray(a, mode='L')
    img  = img.resize((sw * effective_pixel, sh * effective_pixel), Image.NEAREST)
    img  = img.convert("RGB")
    data = np.array(img)
    data = _apply_replace_color(data, replace_color)
    img  = Image.fromarray(data)

    if glow_radius > 0 and glow_intensity > 0:
        img = apply_glow(img, glow_radius, glow_intensity)
    if post_denoise:
        img = _apply_denoise(img, post_denoise)
    if post_smooth:
        img = _apply_smooth(img, post_smooth)

    return img
