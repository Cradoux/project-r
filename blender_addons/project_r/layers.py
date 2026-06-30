"""Single source of truth for per-layer classification + resampling heuristics.

These rules used to be copy-pasted (and had already drifted) across section_ops,
erode_ops and reassemble_ops: which crop filenames are heightmaps vs masks, which
interpolation/colour-management they want, and the two near-identical bilinear
resamplers. Centralising them here means the classification contract is defined
once and the operators can't disagree about what "heightmap" means.

Pure module (no bpy) so it stays unit-testable outside Blender.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


# Filename keywords that identify a layer's role. Matched case-insensitively as
# substrings of the crop filename (e.g. "world_heightmap.exr" -> height).
HEIGHT_KEYWORDS = ("height", "elev", "dem")
MASK_KEYWORDS = ("mask", "land", "plates", "labels")
RAINFALL_KEYWORDS = ("rain", "precip")

# Roles for the optional Map Inputs panel. Used for folder auto-detection of the
# Gleba export set (consistent filenames) and to drive decode/interp choices.
WORLD_KEYWORDS = ("truecolor", "colorsmooth", "coloursmooth")
BATHY_KEYWORDS = ("bathy",)
LANDSEA_KEYWORDS = ("landvssea", "landsea", "crustmap", "island")
UPLIFT_KEYWORDS = ("orogeny", "uplift")
ERODIBILITY_KEYWORDS = ("rocktype", "geolog", "litholog")

_COLOR_EXTS = (".png", ".jpg", ".jpeg")


def is_height_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in HEIGHT_KEYWORDS)


def is_mask_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in MASK_KEYWORDS)


def _tokens(name: str):
    """Split a filename stem into lowercase tokens, breaking on BOTH non-alphanumeric
    separators AND camelCase/PascalCase boundaries -- so 'AverageRainfall' (the Gleba
    naming, no delimiters) tokenizes to ['average', 'rainfall'] while 'terrain' stays a
    single token (so 'rain' as a token-start still doesn't match it)."""
    stem = Path(name).stem
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", stem)
    return re.split(r"[^a-z0-9]+", spaced.lower())


def is_rainfall_name(name: str) -> bool:
    # Token-based (not raw substring): 'rain' must START a name token, so 'rainfall'
    # and 'RainShadow' match but 'terrain' does NOT (it would silently feed a terrain
    # heightmap in as the rainfall field). Also never classify a height/mask layer
    # as rainfall.
    if is_height_name(name) or is_mask_name(name):
        return False
    tokens = _tokens(name)
    return any(t.startswith(k) for t in tokens for k in RAINFALL_KEYWORDS)


def _has_kw(name: str, keywords) -> bool:
    n = name.lower()
    return any(k in n for k in keywords)


def is_world_name(name: str) -> bool:
    """An sRGB display image for the preview sphere (TrueColor / colour hypsometric)."""
    return _has_kw(name, WORLD_KEYWORDS)


def is_bathy_name(name: str) -> bool:
    """Carries sub-sea-level depth (the combined elevation+bathymetry export)."""
    return _has_kw(name, BATHY_KEYWORDS)


def is_landsea_name(name: str) -> bool:
    """A purpose-built land/sea or coastline mask."""
    return _has_kw(name, LANDSEA_KEYWORDS)


def is_uplift_name(name: str) -> bool:
    """A tectonic-uplift / orogeny intensity field."""
    return _has_kw(name, UPLIFT_KEYWORDS)


def is_erodibility_name(name: str) -> bool:
    """A lithology / geology map usable as a spatial erodibility (K_sp) field."""
    return _has_kw(name, ERODIBILITY_KEYWORDS)


def classify_source_folder(filenames):
    """Map a folder's image filenames to Project-R input slots, resolving the
    documented competitions between sibling Gleba exports. Returns a dict
    ``{slot: best_filename or ""}`` for every slot. Pure: takes plain filenames.

    Resolution rules (see docs/gleba_map_integration.md):
      heightmap   prefer *Land (ocean hard-zeroed = sea level, the encoding Project-R's
                  height=brightness*max_elev expects) > plain greyscale; EXCLUDE the
                  *Bathymetry variant (its land is clamped white -- no land relief) and
                  colour images (they match 'elev' but are 8-bit display).
      bathymetry  the *Bathymetry variant (ocean-depth gradient); decoded 1-norm on load.
      rainfall    prefer Average* > January/July; never the RainShadow dryness map.
      world_map   prefer TrueColor > colour hypsometric.
    """
    names = [str(n) for n in filenames]

    def best(include, *, prefer=(), exclude=()):
        cands = [n for n in names
                 if include(n) and not any(e in n.lower() for e in exclude)]
        if not cands:
            return ""

        def score(n):
            low = n.lower()
            for i, p in enumerate(prefer):
                if p in low:
                    return i
            return len(prefer)

        cands.sort(key=lambda n: (score(n), n.lower()))
        return cands[0]

    return {
        "heightmap": best(lambda n: is_height_name(n) and not treat_as_color(n),
                          prefer=("land", "greyscale", "grayscale"),
                          exclude=("color", "colour", "bathymetry")),
        "bathymetry": best(is_bathy_name, prefer=("bathymetry",)),
        "rainfall": best(is_rainfall_name,
                         prefer=("averagerainfall", "rainfall", "precip"),
                         exclude=("shadow",)),
        "world_map": best(is_world_name, prefer=("truecolor",)),
        "landsea_mask": best(is_landsea_name, prefer=("landvssea",)),
        "uplift": best(is_uplift_name, prefer=("orogeny",)),
        "erodibility": best(is_erodibility_name, prefer=("rocktype", "geolog")),
    }


def interp_for_layer(name: str) -> str:
    """Reprojection interpolation: categorical masks must stay 'nearest' (no new
    label values invented at edges); everything else is 'linear'."""
    return "nearest" if is_mask_name(name) else "linear"


def treat_as_color(name: str) -> bool:
    """Whether a layer is an sRGB colour image (vs linear data like a heightmap, a
    rainfall/bathymetry map, or a categorical mask). Drives 8-bit-sRGB vs 16-bit-linear
    encode -- so single-channel data crops stay 16-bit, not collapsed to 8-bit colour."""
    n = name.lower()
    # Our decoded single-channel caches (rainfall/uplift/erodibility/bathymetry) are 16-bit
    # data, never sRGB colour -- recognise them by the cache marker regardless of stem.
    if "__decoded" in n or "__bathy" in n:
        return False
    if is_mask_name(name) or is_height_name(name) or is_rainfall_name(name) or is_bathy_name(name):
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
