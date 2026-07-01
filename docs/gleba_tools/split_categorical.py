"""Split a categorical map (Biome / Koppen / RockType / plates / etc.) into one
black-and-white mask per class, for use as Gaea (or any DCC) selection masks.

White (255) = "this class", black (0) = everything else. Output is full-resolution,
8-bit, hard-edged. Anti-aliased / speckled boundary pixels are snapped to the
nearest dominant palette colour so each class is a clean union (no fringe leftovers).

Usage:
    python split_categorical.py <input.png> <out_dir> [--max-classes N]
        [--coverage 0.997] [--min-frac 0.0008] [--skip-white] [--koppen]

Writes <out_dir>/<stem>_mask_<idx>_<hex>.png per class + <stem>_palette.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Standard Köppen-Geiger colour table (the widely used Peel/Wikipedia palette),
# only consulted with --koppen. Used to GUESS a label per class by nearest colour;
# a large distance means the generator used a different palette (reported, not trusted).
KOPPEN = {
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


def _pack(rgb: np.ndarray) -> np.ndarray:
    return (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32)


def koppen_guess(rgb: tuple[int, int, int]):
    arr = np.array(rgb, dtype=np.float32)
    best, bestd = None, 1e9
    for code, c in KOPPEN.items():
        d = float(np.linalg.norm(arr - np.array(c, dtype=np.float32)))
        if d < bestd:
            best, bestd = code, d
    return best, round(bestd, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("out_dir")
    ap.add_argument("--max-classes", type=int, default=40)
    ap.add_argument("--coverage", type=float, default=0.997)
    ap.add_argument("--min-frac", type=float, default=0.0008)
    ap.add_argument("--skip-white", action="store_true",
                    help="don't emit a mask for pure-white (ocean/background)")
    ap.add_argument("--koppen", action="store_true",
                    help="guess a Köppen code per class from the standard palette")
    args = ap.parse_args()

    stem = Path(args.input).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    im = Image.open(args.input).convert("RGB")
    rgb = np.asarray(im, dtype=np.uint8)
    H, W = rgb.shape[:2]
    flat = rgb.reshape(-1, 3)
    n = flat.shape[0]

    packed = _pack(flat)
    uniq, counts = np.unique(packed, return_counts=True)
    order = np.argsort(counts)[::-1]
    uniq, counts = uniq[order], counts[order]

    # Choose the dominant palette: by cumulative coverage, min fraction, and a hard cap.
    fracs = counts / float(n)
    cum = np.cumsum(fracs)
    keep = []
    for i in range(len(uniq)):
        if len(keep) >= args.max_classes:
            break
        if fracs[i] < args.min_frac and cum[i] > args.coverage:
            break
        keep.append(i)
        if cum[i] >= args.coverage and fracs[i] < args.min_frac:
            break
    keep_packed = uniq[keep]
    palette = np.stack([(keep_packed >> 16) & 255, (keep_packed >> 8) & 255, keep_packed & 255], axis=1).astype(np.float32)
    K = palette.shape[0]

    # Nearest-palette assignment (chunked over K to bound memory): snaps AA fringe.
    best_idx = np.zeros(n, dtype=np.int32)
    best_dist = np.full(n, 1e18, dtype=np.float32)
    flatf = flat.astype(np.float32)
    for k in range(K):
        d = np.sum((flatf - palette[k]) ** 2, axis=1)
        upd = d < best_dist
        best_dist[upd] = d[upd]
        best_idx[upd] = k
    assign = best_idx.reshape(H, W)

    pal_meta = []
    written = 0
    for k in range(K):
        r, g, b = (int(palette[k, 0]), int(palette[k, 1]), int(palette[k, 2]))
        hexc = f"{r:02X}{g:02X}{b:02X}"
        is_white = (r, g, b) == (255, 255, 255)
        coverage = float((assign == k).mean())
        entry = {"index": k, "rgb": [r, g, b], "hex": hexc,
                 "coverage_frac": round(coverage, 5), "is_white_bg": is_white}
        if args.koppen:
            code, dist = koppen_guess((r, g, b))
            entry["koppen_guess"] = code
            entry["koppen_dist"] = dist
        pal_meta.append(entry)
        if is_white and args.skip_white:
            continue
        mask = np.where(assign == k, np.uint8(255), np.uint8(0))
        label = f"_{entry['koppen_guess']}" if args.koppen else ""
        name = f"{stem}_mask_{k:02d}_{hexc}{label}.png"
        Image.fromarray(mask, mode="L").save(out_dir / name)
        written += 1

    meta = {
        "source": args.input, "size": [W, H], "n_classes": K,
        "n_unique_colors_raw": int(uniq.size),
        "coverage_of_palette": round(float(cum[keep[-1]]), 5),
        "masks_written": written, "classes": pal_meta,
    }
    (out_dir / f"{stem}_palette.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in
          ["n_classes", "n_unique_colors_raw", "coverage_of_palette", "masks_written"]}, indent=2))
    for e in pal_meta:
        line = f"  [{e['index']:2d}] #{e['hex']}  {e['coverage_frac']*100:5.2f}%"
        if args.koppen:
            line += f"  ~{e['koppen_guess']} (d={e['koppen_dist']})"
        if e["is_white_bg"]:
            line += "  <- white/background"
        print(line)


if __name__ == "__main__":
    main()
