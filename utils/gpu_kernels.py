"""GPU acceleration layer for Dither Guy.

Backend priority:
  1. CuPy  – NVIDIA CUDA (fastest)
  2. NumPy – pure-CPU fallback (always available)

OpenCL is intentionally disabled: cl.create_some_context() / Context() picks
devices non-deterministically across platforms and has caused instability in
practice.  Re-enable by restoring _try_opencl() if a stable device-selection
strategy is available.

Public API
----------
GPU_BACKEND : str           – "cuda" | "cpu"
to_gpu(arr)  -> array       – move ndarray to active device
from_gpu(arr) -> np.ndarray – bring result back to host
gpu_ordered_dither(a, tiled, t) -> array
gpu_palette_nearest(flat, pal_lab) -> np.ndarray  (index array)
gpu_palette_batch(frames_gpu, pal_lab) -> np.ndarray  (N,H,W,3 uint8)
gpu_rgb_to_lab(r_device)    -> array
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _try_cuda():
    try:
        import cupy as cp
        _ = cp.cuda.runtime.getDeviceCount()
        return cp
    except Exception:
        return None


_CL_PRG = None
_CL_CTX = None
_CL_QUEUE = None

def _try_opencl():
    try:
        import pyopencl as cl
        import pyopencl.array
        # Deterministic device selection prioritizing NVIDIA, AMD, Intel GPUs
        best_device = None
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                if device.type == cl.device_type.GPU:
                    vendor = device.vendor.lower()
                    if "nvidia" in vendor or "amd" in vendor or "intel" in vendor or "advanced micro devices" in vendor:
                        best_device = device
                        break
            if best_device:
                break
        if not best_device:
            return None, None, None
            
        ctx = cl.Context([best_device])
        queue = cl.CommandQueue(ctx)
        
        src = """
        __kernel void ordered_dither(__global const float* a, __global const float* tiled, float t, __global unsigned char* out, int size) {
            int i = get_global_id(0);
            if (i < size) {
                float val = a[i] + (tiled[i] - 128.0f) * (1.0f - t / 255.0f);
                out[i] = (val > t) ? 255 : 0;
            }
        }
        __kernel void rgb_to_lab_nearest(__global const unsigned char* rgb, __global const float* pal_lab, __global int* out, int num_pixels, int num_colors) {
            int i = get_global_id(0);
            if (i < num_pixels) {
                float r = rgb[i*3] / 255.0f;
                float g = rgb[i*3+1] / 255.0f;
                float b = rgb[i*3+2] / 255.0f;
                r = (r > 0.04045f) ? pow((r + 0.055f) / 1.055f, 2.4f) : (r / 12.92f);
                g = (g > 0.04045f) ? pow((g + 0.055f) / 1.055f, 2.4f) : (g / 12.92f);
                b = (b > 0.04045f) ? pow((b + 0.055f) / 1.055f, 2.4f) : (b / 12.92f);
                float x = (r * 0.4124f + g * 0.3576f + b * 0.1805f) / 0.9505f;
                float y = (r * 0.2126f + g * 0.7152f + b * 0.0722f);
                float z = (r * 0.0193f + g * 0.1192f + b * 0.9505f) / 1.089f;
                float fx = (x > 0.008856f) ? cbrt(x) : ((903.3f * x + 16.0f) / 116.0f);
                float fy = (y > 0.008856f) ? cbrt(y) : ((903.3f * y + 16.0f) / 116.0f);
                float fz = (z > 0.008856f) ? cbrt(z) : ((903.3f * z + 16.0f) / 116.0f);
                float L1 = 116.0f * fy - 16.0f;
                float a1 = 500.0f * (fx - fy);
                float b1 = 200.0f * (fy - fz);
                float min_dist = 1e30f;
                int min_idx = 0;
                for (int k = 0; k < num_colors; k++) {
                    float dL = L1 - pal_lab[k*3];
                    float da = a1 - pal_lab[k*3+1];
                    float db = b1 - pal_lab[k*3+2];
                    float dist = dL*dL + da*da + db*db;
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = k;
                    }
                }
                out[i] = min_idx;
            }
        }
        """
        prg = cl.Program(ctx, src).build()
        return cl, ctx, queue, prg
    except Exception:
        return None, None, None, None

cp = _try_cuda()

if cp:
    GPU_BACKEND = "cuda"
    _xp = cp
else:
    cl, _CL_CTX, _CL_QUEUE, _CL_PRG = _try_opencl()
    if cl is not None:
        GPU_BACKEND = "opencl"
        _xp = np
    else:
        GPU_BACKEND = "cpu"
        _xp = np

# ---------------------------------------------------------------------------
# Transfer helpers
# ---------------------------------------------------------------------------

def to_gpu(arr: np.ndarray):
    """Upload a NumPy array to the active device; no-op on CPU backend."""
    if GPU_BACKEND == "cuda" and cp is not None:
        return cp.asarray(arr)
    if GPU_BACKEND == "opencl" and cl is not None:
        import pyopencl.array as cl_array
        if isinstance(arr, cl_array.Array): return arr
        return cl_array.to_device(_CL_QUEUE, np.ascontiguousarray(arr))
    return arr


def from_gpu(arr) -> np.ndarray:
    """Download a device array back to NumPy; no-op on CPU backend."""
    if GPU_BACKEND == "cuda" and cp is not None:
        return cp.asnumpy(arr)
    if GPU_BACKEND == "opencl" and cl is not None:
        import pyopencl.array as cl_array
        if isinstance(arr, cl_array.Array): return arr.get()
    return np.asarray(arr)


def _active_np():
    return _xp

# ---------------------------------------------------------------------------
# Precomputed L*a*b* conversion constants (host side -- uploaded on demand)
# ---------------------------------------------------------------------------
_SRGB_TO_XYZ = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
], dtype=np.float32)
_XYZ_SCALE = np.array([1.0 / 0.9505, 1.0, 1.0 / 1.089], dtype=np.float32)
_LAB_EPS   = np.float32(0.008856)
_LAB_KAPPA = np.float32(903.3)

# Palette LAB is small -- cache it on device to avoid re-uploading each batch
_GPAL_CACHE: dict[int, object] = {}


def _get_gpal(pal_lab: np.ndarray):
    """Return device copy of pal_lab, uploading and caching on first call."""
    key = hash(pal_lab.tobytes())
    if key not in _GPAL_CACHE:
        _GPAL_CACHE[key] = to_gpu(pal_lab.astype(np.float32))
    return _GPAL_CACHE[key]


# ---------------------------------------------------------------------------
# GPU-accelerated ordered dither
# Assumes `a` and `tiled` are already on device when GPU_BACKEND != "cpu".
# Returns a device array -- caller must call from_gpu().
# ---------------------------------------------------------------------------

def gpu_ordered_dither(a, tiled, t: float):
    """Threshold + ordered matrix on the active device."""
    if GPU_BACKEND == "opencl":
        import pyopencl.array as cl_array
        size = a.size
        a_f32 = a.astype(np.float32) if a.dtype != np.float32 else a
        tiled_f32 = tiled.astype(np.float32) if tiled.dtype != np.float32 else tiled
        out = cl_array.empty(_CL_QUEUE, a.shape, dtype=np.uint8)
        _CL_PRG.ordered_dither(_CL_QUEUE, (size,), None, a_f32.data, tiled_f32.data, np.float32(t), out.data, np.int32(size))
        return out

    xp = _active_np()
    _t      = xp.float32(t)
    _128    = xp.float32(128.0)
    _1      = xp.float32(1.0)
    _255    = xp.float32(255.0)
    _inv255 = xp.float32(1.0 / 255.0)
    out = xp.where(
        a.astype(xp.float32) + (tiled.astype(xp.float32) - _128) * (_1 - _t * _inv255) > _t,
        _255, xp.float32(0.0))
    return out.astype(xp.uint8)


# ---------------------------------------------------------------------------
# GPU-accelerated palette nearest-neighbour in L*a*b* space
# ---------------------------------------------------------------------------

def gpu_rgb_to_lab(r_device):
    """Vectorised sRGB -> L*a*b* on the active device.

    Expects `r_device` to already be on device (no internal transfer).
    Returns a device array.
    """
    xp = _active_np()

    r    = r_device.astype(xp.float32) / xp.float32(255.0)
    mask = r > xp.float32(0.04045)
    r    = xp.where(
        mask,
        ((r + xp.float32(0.055)) / xp.float32(1.055)) ** xp.float32(2.4),
        r / xp.float32(12.92)
    )

    M  = to_gpu(_SRGB_TO_XYZ)
    sc = to_gpu(_XYZ_SCALE)

    xyz = (r @ M.T) * sc

    f = xp.where(
        xyz > xp.float32(_LAB_EPS),
        xp.cbrt(xyz),
        (xp.float32(_LAB_KAPPA) * xyz + xp.float32(16.0)) / xp.float32(116.0)
    )

    L = xp.float32(116.0) * f[..., 1] - xp.float32(16.0)
    a = xp.float32(500.0) * (f[..., 0] - f[..., 1])
    b = xp.float32(200.0) * (f[..., 1] - f[..., 2])
    return xp.stack([L, a, b], axis=-1)


_PALETTE_NEAREST_BATCH = 1_048_576  # pixels per chunk -- caps peak VRAM allocation


import threading
_GPU_LOCK = threading.Lock()

def gpu_palette_nearest(flat, pal_lab: np.ndarray) -> np.ndarray:
    """Return argmin palette index for each row of `flat` (N,3) RGB."""
    with _GPU_LOCK:
        if GPU_BACKEND == "cpu":
            pix_lab = _cpu_rgb_to_lab(np.asarray(flat))
            pix_sq  = np.sum(pix_lab**2, axis=-1, keepdims=True)
            pal_sq  = np.sum(pal_lab**2, axis=-1)
            dists   = pix_sq + pal_sq - 2 * (pix_lab @ pal_lab.T)
            return np.argmin(dists, axis=1)
            
        if GPU_BACKEND == "opencl":
            import pyopencl.array as cl_array
            gpal = _get_gpal(pal_lab)
            results = []
            n = flat.shape[0]
            num_colors = pal_lab.shape[0]
            for i in range(0, n, _PALETTE_NEAREST_BATCH):
                chunk = flat[i : i + _PALETTE_NEAREST_BATCH]
                size = chunk.shape[0]
                out = cl_array.empty(_CL_QUEUE, (size,), dtype=np.int32)
                _CL_PRG.rgb_to_lab_nearest(_CL_QUEUE, (size,), None, chunk.data, gpal.data, out.data, np.int32(size), np.int32(num_colors))
                results.append(out.get())
            return np.concatenate(results)

        xp      = _active_np()
        gpal    = _get_gpal(pal_lab)
        results = []
        n       = flat.shape[0]

        for i in range(0, n, _PALETTE_NEAREST_BATCH):
            chunk    = flat[i : i + _PALETTE_NEAREST_BATCH]
            gpix_lab = gpu_rgb_to_lab(chunk)
            
            # O(NK) memory instead of O(NKC) via a^2 + b^2 - 2ab expansion
            gpix_sq = xp.sum(gpix_lab**2, axis=-1, keepdims=True)
            gpal_sq = xp.sum(gpal**2, axis=-1)
            dot     = gpix_lab @ gpal.T
            dists   = gpix_sq + gpal_sq - 2 * dot
            
            results.append(from_gpu(xp.argmin(dists, axis=1)))

        return np.concatenate(results)


def gpu_palette_batch(frames_gpu, pal_rgb: np.ndarray) -> np.ndarray:
    """Map every pixel in a batch of frames to its nearest palette colour.

    Parameters
    ----------
    frames_gpu : device array, shape (B, H, W, 3) uint8
        Already on device.  CPU backend accepts plain np.ndarray.
    pal_rgb : np.ndarray, shape (K, 3) float32
        RGB palette to map pixels into.

    Returns
    -------
    np.ndarray, shape (B, H, W, 3) uint8  -- always a HOST array.

    Notes
    -----
    Internally the (B, H, W, 3) volume is flattened to (B*H*W, 3), processed
    via gpu_palette_nearest in pixel-chunks, then reshaped back.  One PCIe
    upload (frames) + one download (result) for the whole batch.
    """
    xp = _active_np()
    B, H, W, C = frames_gpu.shape

    # Fix: convert the RGB palette to LAB before passing it to distance calculation
    pal_lab = _cpu_rgb_to_lab(np.asarray(pal_rgb, dtype=np.float32))

    # Flatten to pixel list
    flat = frames_gpu.reshape(B * H * W, C)          # device array

    # Nearest-palette index for every pixel (returns HOST array)
    indices = gpu_palette_nearest(flat, pal_lab)      # (B*H*W,) int host

    # Map indices -> RGB colours using the host palette colours
    out_flat = pal_rgb[indices]                        # (B*H*W, 3) host
    out = out_flat.reshape(B, H, W, C).astype(np.uint8)
    return out


def _cpu_rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """CPU-only L*a*b* conversion using a fused matrix multiply."""
    r    = rgb.astype(np.float32) / 255.0
    mask = r > 0.04045
    r    = np.where(mask, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
    xyz  = (r @ _SRGB_TO_XYZ.T) * _XYZ_SCALE
    f    = np.where(xyz > _LAB_EPS, np.cbrt(xyz), (_LAB_KAPPA * xyz + 16.0) / 116.0)
    L    = 116.0 * f[:, 1] - 16.0
    a    = 500.0 * (f[:, 0] - f[:, 1])
    b    = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=-1)
