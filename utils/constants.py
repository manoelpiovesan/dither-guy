from __future__ import annotations

import os

_MAX_PIXELS    = 8_000_000
_HISTORY_LIMIT = 20
_DEBOUNCE_MS   = 250
_PREVIEW_SCALE = 0.5
_VIDEO_WORKERS = max(4, (os.cpu_count() - 1 or 4))
VERSION        = "5.5.0"

### Removed Dither Algo

# Error Diffusion

# Nakano
# Ostromoukhov
# jarvis-Judice-Ninke


METHOD_GROUPS = {
    "Ordered": [
        "Bayer 2x2", "Bayer 4x4", "Bayer 8x8",
        "Clustered-Dot", "Halftone", "Blue-Noise",
        "Void-and-Cluster", "Pattern", "Crosshatch",
    ],
    "Error Diffusion": [
        "Floyd-Steinberg", "Atkinson", "Sierra", "Sierra-Lite",
        "Jarvis-Judice-Ninke", "Stucki", "Nakano",
        "Burkes", "Stevenson-Arce", "Ostromoukhov",
        "Variable-Error", "Dot-Diffusion", "Riemersma",
    ],
    "Stochastic": [
        "Blue-Noise Mask",
    ],
}
METHODS: list[str] = [m for ms in METHOD_GROUPS.values() for m in ms]

# Populated at runtime by gpu_kernels.py; read by status bar in app.py
from .gpu_kernels import GPU_BACKEND   # noqa: E402
