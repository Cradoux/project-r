"""Decode Gleba-style map exports back into usable fields. Pure numpy (no bpy), so
it stays unit-testable outside Blender and is shared by the input operators.

Two encodings need decoding before Project-R can consume them:

* **Colormap ramps** (viridis): the scalar is recovered from the colormap INDEX, not
  the luminance. ``decode_colormap_to_scalar`` inverts a baked 256-entry LUT
  (Blender's Python has no matplotlib) by nearest-colour lookup.
* **Categorical palettes** (Biome / Koppen / RockType / plates / ...): each distinct
  colour is a class. ``split_categorical`` finds the dominant palette, snaps
  anti-aliased boundary pixels to the nearest palette colour, and returns one
  integer class raster plus per-class metadata (so callers can emit B&W masks).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # packaged inside the addon
    from ._viridis_lut import VIRIDIS_LUT_U8
except ImportError:  # importable standalone for tests
    from _viridis_lut import VIRIDIS_LUT_U8  # type: ignore

VIRIDIS_LUT = np.asarray(VIRIDIS_LUT_U8, dtype=np.float32)  # (256,3) in 0..255


# Standard Köppen-Geiger colour table (the widely used Peel/Wikipedia palette).
# Gleba was confirmed to use this exact palette, so categorical Koppen exports can be
# auto-labelled by nearest colour. A large distance means a non-standard palette.
KOPPEN: Dict[str, Tuple[int, int, int]] = {
    "Af": (0, 0, 254), "Am": (0, 119, 255), "Aw": (70, 169, 250), "As": (70, 169, 250),
    "BWh": (254, 0, 0), "BWk": (254, 150, 149), "BSh": (245, 163, 1), "BSk": (255, 219, 99),
    "Csa": (255, 255, 0), "Csb": (198, 199, 0), "Csc": (150, 150, 0),
    "Cwa": (150, 255, 150), "Cwb": (99, 199, 100), "Cwc": (50, 150, 51),
    "Cfa": (198, 255, 78), "Cfb": (102, 255, 51), "Cfc": (50, 199, 0),
    "Dsa": (255, 0, 254), "Dsb": (198, 0, 199), "Dsc": (150, 50, 149), "Dsd": (150, 100, 149),
    "Dwa": (171, 177, 255), "Dwb": (90, 119, 219), "Dwc": (76, 81, 181), "Dwd": (50, 0, 135),
    "Dfa": (0, 255, 255), "Dfb": (56, 199, 255), "Dfc": (0, 126, 125), "Dfd": (0, 69, 94),
    "ET": (178, 178, 178), "EF": (104, 104, 104),
}


def koppen_guess(rgb: Tuple[int, int, int]) -> Tuple[str, float]:
    """Nearest standard Köppen code to an RGB colour + the L2 distance to it."""
    arr = np.array(rgb, dtype=np.float32)
    best, bestd = "", 1e18
    for code, c in KOPPEN.items():
        d = float(np.linalg.norm(arr - np.array(c, dtype=np.float32)))
        if d < bestd:
            best, bestd = code, d
    return best, round(bestd, 1)


# ---------------------------------------------------------------------------
# Colormap ramp -> scalar
# ---------------------------------------------------------------------------

def decode_luminance(rgb01: np.ndarray) -> np.ndarray:
    """Perceptual luminance [0,1] of an (H,W,3+) image. A dependency-free, monotonic
    decode for dark->bright intensity ramps whose exact colormap is unknown (e.g. Gleba's
    OrogenyStrength / soil maps), where brightness tracks the quantity. Returns (H,W)."""
    rgb = np.asarray(rgb01, dtype=np.float32)[..., :3]
    return np.clip(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2], 0.0, 1.0)


def decode_colormap_to_scalar(
    rgb01: np.ndarray,
    *,
    lut: np.ndarray = VIRIDIS_LUT,
    dark_sentinel: int = 35,
    sentinel_value: float = 0.0,
) -> np.ndarray:
    """Invert a colormap-encoded image back to a [0,1] scalar via nearest-LUT lookup.

    ``rgb01`` is (H,W,3+) in [0,1] (as ``imaging.load_image`` returns). Pixels darker
    than ``dark_sentinel`` (the near-black ocean/no-data sentinel many Gleba ramps use,
    distinct from viridis[0]=(68,1,84)) are set to ``sentinel_value`` instead of being
    matched to the bottom of the ramp. Returns (H,W) float32 in [0,1].
    """
    rgb = np.asarray(rgb01, dtype=np.float32)[..., :3] * 255.0
    H, W = rgb.shape[:2]
    flat = rgb.reshape(-1, 3)
    n = flat.shape[0]

    lut = np.asarray(lut, dtype=np.float32)  # (256,3)
    lut2 = (lut * lut).sum(axis=1)           # (256,)
    out = np.empty(n, dtype=np.float32)

    # Chunk the nearest-LUT search to bound the (chunk,256) distance matrix.
    chunk = 262144
    for i in range(0, n, chunk):
        p = flat[i:i + chunk]                       # (c,3)
        p2 = (p * p).sum(axis=1, keepdims=True)     # (c,1)
        d2 = p2 - 2.0 * (p @ lut.T) + lut2[None, :]  # (c,256)
        out[i:i + chunk] = d2.argmin(axis=1).astype(np.float32) / 255.0

    scalar = out.reshape(H, W)
    if dark_sentinel > 0:
        ocean = np.all(rgb < float(dark_sentinel), axis=-1)
        scalar = np.where(ocean, np.float32(sentinel_value), scalar)
    return scalar.astype(np.float32)


# ---------------------------------------------------------------------------
# Categorical palette -> class raster + metadata
# ---------------------------------------------------------------------------

def _pack(rgb_u8: np.ndarray) -> np.ndarray:
    r = rgb_u8[:, 0].astype(np.uint32)
    return (r << 16) | (rgb_u8[:, 1].astype(np.uint32) << 8) | rgb_u8[:, 2].astype(np.uint32)


def detect_palette(
    rgb_u8: np.ndarray,
    *,
    max_classes: int = 40,
    coverage: float = 0.997,
    min_frac: float = 0.0008,
) -> Tuple[np.ndarray, int]:
    """Pick the dominant palette of a categorical image by colour frequency.

    Returns ``(palette, n_unique_raw)`` where palette is (K,3) float32 RGB (0..255),
    ordered most-common first. Colours past the coverage target whose frequency is
    below ``min_frac`` (anti-alias fringe) are dropped; ``max_classes`` is a hard cap.
    """
    flat = np.asarray(rgb_u8, dtype=np.uint8).reshape(-1, 3)
    n = flat.shape[0]
    packed = _pack(flat)
    uniq, counts = np.unique(packed, return_counts=True)
    order = np.argsort(counts)[::-1]
    uniq, counts = uniq[order], counts[order]
    fracs = counts / float(n)
    cum = np.cumsum(fracs)

    keep: List[int] = []
    for i in range(len(uniq)):
        if len(keep) >= max_classes:
            break
        if fracs[i] < min_frac and cum[i] > coverage:
            break
        keep.append(i)
        if cum[i] >= coverage and fracs[i] < min_frac:
            break
    kp = uniq[keep]
    palette = np.stack([(kp >> 16) & 255, (kp >> 8) & 255, kp & 255], axis=1).astype(np.float32)
    return palette, int(uniq.size)


def assign_nearest(rgb_u8: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Assign every pixel to the nearest palette entry (L2 in RGB). Snaps the
    anti-aliased boundary fringe into a clean class. Returns (H,W) int32 indices."""
    flat = np.asarray(rgb_u8, dtype=np.float32).reshape(-1, 3)
    H, W = rgb_u8.shape[:2]
    best_idx = np.zeros(flat.shape[0], dtype=np.int32)
    best_d = np.full(flat.shape[0], 1e18, dtype=np.float32)
    for k in range(palette.shape[0]):
        d = ((flat - palette[k]) ** 2).sum(axis=1)
        upd = d < best_d
        best_d[upd] = d[upd]
        best_idx[upd] = k
    return best_idx.reshape(H, W)


def split_categorical(
    rgb_u8: np.ndarray,
    *,
    max_classes: int = 40,
    coverage: float = 0.997,
    min_frac: float = 0.0008,
    koppen: bool = False,
) -> Dict:
    """Decompose a categorical map into classes.

    Returns a dict with ``classes`` (per-class metadata: index, rgb, hex,
    coverage_frac, is_white_bg, and koppen_guess/koppen_dist when ``koppen``), the
    integer ``assignment`` raster (H,W), and palette stats. Callers build B&W masks
    from ``assignment == index``.
    """
    palette, n_raw = detect_palette(rgb_u8, max_classes=max_classes,
                                    coverage=coverage, min_frac=min_frac)
    assign = assign_nearest(rgb_u8, palette)
    K = palette.shape[0]
    classes: List[Dict] = []
    for k in range(K):
        r, g, b = int(palette[k, 0]), int(palette[k, 1]), int(palette[k, 2])
        entry: Dict = {
            "index": k,
            "rgb": [r, g, b],
            "hex": f"{r:02X}{g:02X}{b:02X}",
            "coverage_frac": round(float((assign == k).mean()), 5),
            "is_white_bg": (r, g, b) == (255, 255, 255),
        }
        if koppen:
            code, dist = koppen_guess((r, g, b))
            entry["koppen_guess"] = code
            entry["koppen_dist"] = dist
        classes.append(entry)
    return {
        "n_classes": K,
        "n_unique_colors_raw": n_raw,
        "assignment": assign,
        "classes": classes,
    }
