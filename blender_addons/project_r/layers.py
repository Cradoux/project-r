"""Single source of truth for per-layer classification + resampling heuristics.

These rules used to be copy-pasted (and had already drifted) across section_ops,
erode_ops and reassemble_ops: which crop filenames are heightmaps vs masks, which
interpolation/colour-management they want, and the two near-identical bilinear
resamplers. Centralising them here means the classification contract is defined
once and the operators can't disagree about what "heightmap" means.

Pure module (no bpy) so it stays unit-testable outside Blender.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# Filename keywords that identify a layer's role. Matched case-insensitively as
# substrings of the crop filename (e.g. "world_heightmap.exr" -> height).
HEIGHT_KEYWORDS = ("height", "elev", "dem")
MASK_KEYWORDS = ("mask", "land", "plates", "labels")

_COLOR_EXTS = (".png", ".jpg", ".jpeg")


def is_height_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in HEIGHT_KEYWORDS)


def is_mask_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in MASK_KEYWORDS)


def interp_for_layer(name: str) -> str:
    """Reprojection interpolation: categorical masks must stay 'nearest' (no new
    label values invented at edges); everything else is 'linear'."""
    return "nearest" if is_mask_name(name) else "linear"


def treat_as_color(name: str) -> bool:
    """Whether a layer is an sRGB colour image (vs linear data like a heightmap or
    a categorical mask). Drives 8-bit-sRGB vs 16-bit-linear encode decisions."""
    if is_mask_name(name) or is_height_name(name):
        return False
    return Path(name).suffix.lower() in _COLOR_EXTS


def resample_2d(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Bilinear resample a 2D float array to (out_h, out_w). Uses Pillow 'F' mode
    when available; falls back to nearest-ish index sampling if Pillow is absent."""
    h, w = arr.shape[:2]
    if (w, h) == (out_w, out_h):
        return arr.astype(np.float32, copy=False)
    try:
        from PIL import Image as PILImage  # type: ignore

        im = PILImage.fromarray(arr.astype(np.float32), mode="F")
        im = im.resize((int(out_w), int(out_h)), resample=PILImage.BILINEAR)
        return np.asarray(im, dtype=np.float32)
    except Exception:
        ys = np.clip((np.arange(out_h) * h / out_h).astype(int), 0, h - 1)
        xs = np.clip((np.arange(out_w) * w / out_w).astype(int), 0, w - 1)
        return arr[ys][:, xs].astype(np.float32)


def resample_by_scale(pixels: np.ndarray, scale_factor: float) -> np.ndarray:
    """Resample a (H, W[, C]) array by a scale factor with explicit bilinear
    interpolation. scale_factor > 1 upscales, < 1 downscales. Channel-aware so it
    works for both single-channel masks/heightmaps and multi-channel colour."""
    if abs(scale_factor - 1.0) < 0.001:
        return pixels

    h, w = pixels.shape[:2]
    new_h = max(1, int(round(h * scale_factor)))
    new_w = max(1, int(round(w * scale_factor)))

    if pixels.ndim == 2:
        pixels = pixels[..., None]
    channels = pixels.shape[2]

    y_new = np.linspace(0, h - 1, new_h)
    x_new = np.linspace(0, w - 1, new_w)
    xx, yy = np.meshgrid(x_new, y_new)

    x0 = np.floor(xx).astype(int)
    y0 = np.floor(yy).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    dx = xx - x0
    dy = yy - y0

    output = np.zeros((new_h, new_w, channels), dtype=pixels.dtype)
    for c in range(channels):
        p = pixels[:, :, c]
        v00 = p[y0, x0]
        v01 = p[y0, x1]
        v10 = p[y1, x0]
        v11 = p[y1, x1]
        output[:, :, c] = (
            v00 * (1 - dx) * (1 - dy)
            + v01 * dx * (1 - dy)
            + v10 * (1 - dx) * dy
            + v11 * dx * dy
        )

    return output
