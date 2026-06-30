"""Profile a Gleba map-export PNG: encoding, channels, grayscale-ness,
categorical-vs-continuous, and best-fit matplotlib colormap (viridis etc.).

Usage:  python analyze_map.py <path-to-png>
Emits a single JSON object to stdout.
"""
from __future__ import annotations

import json
import sys
import numpy as np
from PIL import Image

# Candidate colormaps Gleba-style exports commonly use.
CMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    "jet", "rainbow", "gist_rainbow", "hsv", "coolwarm", "RdBu",
    "RdYlBu", "RdYlGn", "Spectral", "terrain", "gist_earth", "ocean",
    "Blues", "Greens", "Reds", "YlGnBu", "YlOrRd", "BrBG", "PuOr",
    "twilight", "cubehelix", "gnuplot", "gnuplot2", "nipy_spectral",
]


def _cmap_lut(name: str, n: int = 256) -> np.ndarray:
    import matplotlib
    cm = matplotlib.colormaps[name]
    xs = np.linspace(0.0, 1.0, n)
    return np.asarray(cm(xs))[:, :3].astype(np.float32)  # (n,3) in 0..1


def main(path: str) -> dict:
    im = Image.open(path)
    info: dict = {
        "file": path,
        "pil_mode": im.mode,
        "size": list(im.size),  # (w,h)
        "has_palette": im.mode == "P",
        "n_palette_colors": None,
        "bit_depth_guess": None,
    }

    # Bit-depth: PIL exposes 16-bit grayscale as I;16 / I.
    raw = np.array(im)
    info["raw_dtype"] = str(raw.dtype)
    info["raw_shape"] = list(raw.shape)
    if raw.dtype == np.uint16 or im.mode in ("I;16", "I;16B", "I"):
        info["bit_depth_guess"] = 16
    elif raw.dtype == np.uint8:
        info["bit_depth_guess"] = 8

    has_alpha = im.mode in ("LA", "RGBA", "PA") or (raw.ndim == 3 and raw.shape[2] == 4)
    info["has_alpha"] = bool(has_alpha)

    if im.mode == "P":
        pal = im.getpalette() or []
        # Count palette indices actually used.
        idx = np.array(im)
        used = np.unique(idx)
        info["n_palette_colors"] = int(used.size)

    # Work in RGB float for color analysis; keep a grayscale view too.
    rgb_im = im.convert("RGB")
    rgb = np.asarray(rgb_im, dtype=np.float32) / 255.0  # (H,W,3)
    H, W = rgb.shape[:2]
    info["megapixels"] = round(H * W / 1e6, 2)

    # --- grayscale-ness: are R==G==B (ignoring alpha)? ---
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    chan_spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    gray_frac = float((chan_spread < (1.5 / 255.0)).mean())
    info["grayscale_fraction"] = round(gray_frac, 4)
    info["is_effectively_grayscale"] = bool(gray_frac > 0.995)

    # --- value stats: prefer the true 16-bit data if present, else luminance ---
    if raw.ndim == 2 and raw.dtype == np.uint16:
        vals = raw.astype(np.float64)
        denom = 65535.0
    elif raw.ndim == 2:
        vals = raw.astype(np.float64)
        denom = 255.0
    else:
        vals = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float64) * 255.0
        denom = 255.0
    info["value_min"] = float(vals.min())
    info["value_max"] = float(vals.max())
    info["value_mean"] = round(float(vals.mean()), 3)
    norm = vals / denom
    info["frac_at_min"] = round(float((vals <= vals.min() + 1e-6).mean()), 4)
    info["frac_zero"] = round(float((vals <= (0.5 if denom == 255 else 1.5)).mean()), 4)
    # coarse histogram of normalized luminance
    hist, _ = np.histogram(norm, bins=16, range=(0, 1))
    info["lum_hist16"] = [int(x) for x in hist]

    # --- unique colors / categorical detection (sample-capped) ---
    flat = (np.asarray(rgb_im, dtype=np.uint8)).reshape(-1, 3)
    # sample for speed on big images
    rng = np.random.default_rng(0)
    if flat.shape[0] > 400000:
        sel = rng.choice(flat.shape[0], 400000, replace=False)
        sample = flat[sel]
    else:
        sample = flat
    # exact unique on the full image but via void-view (fast)
    packed = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2].astype(np.uint32)
    uniq, counts = np.unique(packed, return_counts=True)
    n_unique = int(uniq.size)
    info["n_unique_colors"] = n_unique
    order = np.argsort(counts)[::-1]
    top = []
    total = float(flat.shape[0])
    for i in order[:12]:
        c = int(uniq[i])
        top.append({
            "rgb": [(c >> 16) & 255, (c >> 8) & 255, c & 255],
            "frac": round(counts[i] / total, 4),
        })
    info["top_colors"] = top
    # For grayscale (esp. 16-bit) the 8-bit RGB unique count is a projection
    # artifact, so report the true distinct level count for those.
    if info["is_effectively_grayscale"]:
        info["n_unique_levels"] = int(np.unique(vals).size)
    else:
        info["n_unique_levels"] = n_unique
    # Categorical heuristic: few distinct values covering most of the image,
    # and NOT a wide-range continuous grayscale ramp.
    wide_gray = info["is_effectively_grayscale"] and (info["value_max"] - info["value_min"]) > (0.1 * denom)
    info["categorical_candidate"] = bool(info["n_unique_levels"] <= 64 and not wide_gray)

    # --- colormap fit (only meaningful for non-grayscale continuous images) ---
    cmap_fit = None
    if not info["is_effectively_grayscale"] and n_unique > 64:
        s_rgb = sample.astype(np.float32) / 255.0  # (M,3)
        # cap further for the M x 256 distance matrix
        if s_rgb.shape[0] > 40000:
            s_rgb = s_rgb[rng.choice(s_rgb.shape[0], 40000, replace=False)]
        best = []
        for name in CMAPS:
            try:
                lut = _cmap_lut(name, 256)
            except Exception:
                continue
            # nearest LUT colour for each sample pixel
            d = np.linalg.norm(s_rgb[:, None, :] - lut[None, :, :], axis=2)  # (M,256)
            nn = d.min(axis=1)
            nn_idx = d.argmin(axis=1)
            mean_err = float(nn.mean())
            p95_err = float(np.percentile(nn, 95))
            coverage = float(np.unique((nn_idx // 8)).size / 32.0)  # how much of the ramp is spanned
            best.append({
                "cmap": name,
                "mean_err": round(mean_err, 4),
                "p95_err": round(p95_err, 4),
                "ramp_coverage": round(coverage, 3),
            })
        best.sort(key=lambda x: x["mean_err"])
        cmap_fit = best[:5]
    info["cmap_fit_top5"] = cmap_fit
    if cmap_fit:
        top1 = cmap_fit[0]
        # A good fit: low mean error AND the data spans much of the ramp.
        info["likely_colormap"] = (
            top1["cmap"] if (top1["mean_err"] < 0.06 and top1["ramp_coverage"] > 0.4) else None
        )
    else:
        info["likely_colormap"] = None

    # --- final encoding verdict ---
    if info["is_effectively_grayscale"]:
        enc = "grayscale_continuous"
    elif info["categorical_candidate"]:
        enc = "categorical_palette"
    elif info["likely_colormap"]:
        enc = f"colormap_ramp:{info['likely_colormap']}"
    else:
        enc = "truecolor_or_unknown_rgb"
    info["encoding_verdict"] = enc
    return info


if __name__ == "__main__":
    out = main(sys.argv[1])
    print(json.dumps(out, indent=2))
