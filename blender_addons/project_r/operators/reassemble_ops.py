from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple, Set

import bpy

from .. import manifest as manifest_lib
from .. import imaging
from ..projection_backend import ProjectionParams, project_hammer_array_to_equirect

import numpy as np


class PP_OT_validate_processed(bpy.types.Operator):
    bl_idname = "pp.validate_processed"
    bl_label = "Validate Processed Files"
    bl_description = "Check that expected processed section files exist"

    def execute(self, context: bpy.types.Context):
        s = context.scene.projection_pasta
        root = s.project_root_path()
        mp = s.manifest_path()
        if root is None or mp is None or not mp.exists():
            self.report({"ERROR"}, "manifest.json not found (set Project Root and Init Project)")
            return {"CANCELLED"}

        manifest = manifest_lib.read_manifest(mp)
        missing: List[str] = []
        present: int = 0
        for sec in manifest.get("sections", []):
            sec_id = sec.get("id", "")
            if not sec_id:
                continue
            proc_dir = root / "processed" / sec_id
            if not proc_dir.exists():
                missing.append(str(Path("processed") / sec_id))
                continue
            files = [p for p in proc_dir.iterdir() if p.is_file()]
            present += len(files)
        if missing:
            self.report({"WARNING"}, f"Missing processed section folders: {len(missing)} (see console)")
            for p in missing:
                print("[Project-R] Missing:", p)
        else:
            self.report({"INFO"}, f"Processed folders present; found {present} file(s) total")
        return {"FINISHED"}


def _is_mask_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("mask", "land", "plates", "labels"))


def _is_height_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("height", "elev", "dem"))


def _interp_for_name(name: str) -> str:
    if _is_mask_name(name):
        return "nearest"
    if _is_height_name(name):
        return "linear"
    return "linear"


def _treat_as_color_name(name: str) -> bool:
    if _is_mask_name(name) or _is_height_name(name):
        return False
    ext = Path(name).suffix.lower()
    return ext in (".png", ".jpg", ".jpeg")


def _resample_image(
    pixels: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """
    Resample an image by a scale factor using bilinear interpolation.
    scale_factor > 1 = upscale (finer resolution), < 1 = downscale.
    """
    if abs(scale_factor - 1.0) < 0.001:
        return pixels
    
    h, w = pixels.shape[:2]
    new_h = max(1, int(round(h * scale_factor)))
    new_w = max(1, int(round(w * scale_factor)))
    
    # Ensure we have 3D array
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    channels = pixels.shape[2]
    
    # Create coordinate grids for the new size
    y_new = np.linspace(0, h - 1, new_h)
    x_new = np.linspace(0, w - 1, new_w)
    xx, yy = np.meshgrid(x_new, y_new)
    
    # Bilinear interpolation
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
            v00 * (1 - dx) * (1 - dy) +
            v01 * dx * (1 - dy) +
            v10 * (1 - dx) * dy +
            v11 * dx * dy
        )
    
    return output


# ---------------------------------------------------------------------------
# Geographic-pole repair (Bug: streaky/transparent band at the top of polar sections)
# ---------------------------------------------------------------------------
# A geographic pole is a SINGLE point in Hammer space, so during Hammer->equirect
# reassembly the equirect rows nearest a pole all map to a tiny disk of the Hammer crop
# -> that disk is magnified to fill the whole top/bottom band. Two artifacts result:
#   (a) the effective mask's UV-singularity "starburst" of uncovered spokes at the pole
#       becomes a transparent/checkerboard gap band, and
#   (b) the extreme magnification combs the RGB into comma/dash streaks.
# We repair (a) by filling the pole-neighborhood disk in the (Hammer crop-space) mask
# before reprojection, and (b) by collapsing the magnified equirect rows toward their
# mask-weighted row average after reprojection. Both are no-ops for non-polar sections.
POLE_CAP_DEG = 6.0  # treat the spherical cap within this colatitude of a pole


def _get_pp():
    from ..vendor.projectionpasta import projectionpasta as pp  # type: ignore

    return pp


def _hammer_forward_px(pp, lon_deg: float, lat_deg: float, center, full_w: int, full_h: int):
    """Forward-project a lon/lat (deg) to pixel coords on the section's Hammer full canvas."""
    aspect = np.array(
        [math.radians(center[0]), math.radians(center[1]), math.radians(center[2])],
        dtype=np.float64,
    )
    lon_r, lat_r = pp.Rotate_to(
        np.array([math.radians(lon_deg)]), np.array([math.radians(lat_deg)]), aspect
    )
    opts = dict(pp.def_opts)
    opts["in"] = False
    hx, hy = pp.posl["Hammer"](lon_r, lat_r, opts)  # [-1, 1]
    fx = (float(hx[0]) + 1.0) / 2.0 * full_w - 0.5
    fy = (1.0 - float(hy[0])) / 2.0 * full_h - 0.5
    return fx, fy


def _colat_to_crop_radius(pp, center, full_w, full_h, rect_x, rect_y, cx, cy, is_north, colat_deg):
    """Max crop-space distance from the pole point to the colatitude circle (handles
    Hammer distortion: the cap is an ellipse, so we take the outer extent)."""
    pole_lat = 90.0 if is_north else -90.0
    edge_lat = pole_lat - colat_deg if is_north else pole_lat + colat_deg
    rmax = 0.0
    for lon in range(0, 360, 30):
        try:
            ex, ey = _hammer_forward_px(pp, float(lon), edge_lat, center, full_w, full_h)
        except Exception:
            continue
        d = math.hypot((ex - rect_x) - cx, (ey - rect_y) - cy)
        if math.isfinite(d):
            rmax = max(rmax, d)
    return rmax


def _detect_pole_fill_radius(mask: np.ndarray, cx: float, cy: float, max_r: float) -> float:
    """Radius (crop px) out to which the pole neighbourhood should be solidified.

    Around a pole-covering cap the effective mask is feathered LOW (the UV-singularity
    'starburst' spokes act as nearby boundaries, so generate_effective_mask fades the
    whole pole region down even where it is 'covered'). The artifact band is exactly this
    feathered/spoked region; it ends at the INNER EDGE of the truly-solid cap interior
    (mask value >= 0.9). We return that radius (everything inside is genuine cap interior
    and should be solid). For a small cap with no solid interior, we stop at the cap
    boundary (coverage falls away) instead. Returns 0 if there's no covered core."""
    h, w = mask.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_r = float(max(8.0, min(max_r, math.hypot(w, h))))
    nb = 64
    ring = np.clip((dist / max_r * nb).astype(np.int64), 0, nb - 1)
    cnt = np.bincount(ring.ravel(), minlength=nb).astype(np.float64)
    solid = np.bincount(ring.ravel(), weights=(mask >= 0.9).ravel().astype(np.float64), minlength=nb)
    covd = np.bincount(ring.ravel(), weights=(mask > 0.01).ravel().astype(np.float64), minlength=nb)
    f_solid = np.where(cnt > 0, solid / np.maximum(cnt, 1.0), 0.0)
    f_cov = np.where(cnt > 0, covd / np.maximum(cnt, 1.0), 0.0)
    ring_w = max_r / nb
    if cnt[0] == 0 or f_cov[0] < 0.1:
        return 0.0
    r_fill = ring_w
    for i in range(nb):
        if cnt[i] == 0 or f_cov[i] < 0.05:
            break  # reached the cap boundary (small cap) -> stop here
        # Inner edge of the solid cap interior: ring is solid and stays solid.
        if f_solid[i] >= 0.9 and np.all(f_solid[i : min(i + 3, nb)] >= 0.85):
            break
        r_fill = (i + 1) * ring_w
    return r_fill


def _seal_pole_starburst(
    effective_mask_crop: np.ndarray,
    coverage_for_extend: np.ndarray,
    *,
    params: ProjectionParams,
    full_w: int,
    full_h: int,
    rect_x: int,
    rect_y: int,
    w: int,
    h: int,
    gh: int,
    cap_deg: float = POLE_CAP_DEG,
) -> List[Tuple[bool, int]]:
    """Fill the pole-neighborhood disk in the crop-space mask when the section covers a
    geographic pole. The disk radius adapts to the actual UV-singularity 'starburst'
    extent (resolution-independent). Returns a list of (is_north, band_rows) for the
    equirect collapse step; empty (and no mutation) when the section touches no pole."""
    try:
        pp = _get_pp()
    except Exception:
        return []
    center = (params.center_lon_deg, params.center_lat_deg, params.rot_deg)
    sealed: List[Tuple[bool, int]] = []
    yy = xx = None  # allocated lazily, only once a pole is found inside the crop
    for is_north in (True, False):
        pole_lat = 90.0 if is_north else -90.0
        try:
            fx, fy = _hammer_forward_px(pp, 0.0, pole_lat, center, full_w, full_h)
        except Exception:
            continue
        cx, cy = fx - rect_x, fy - rect_y
        if not (0.0 <= cx < w and 0.0 <= cy < h):
            continue  # this pole is not inside the crop
        if yy is None:
            yy, xx = np.mgrid[0:h, 0:w]

        # Search radius cap: a generous colatitude so detection has room to find the
        # solid interior even for coarse sections, but never the whole crop.
        search_r = _colat_to_crop_radius(
            pp, center, full_w, full_h, rect_x, rect_y, cx, cy, is_north, 6.0 * cap_deg
        )
        fill_r = _detect_pole_fill_radius(effective_mask_crop, cx, cy, search_r or min(w, h) / 2.0)
        # Floor at the cap_deg radius so we always seal at least the immediate pole, and
        # pad a touch to swallow the ragged inner edge of the solid cap.
        floor_r = _colat_to_crop_radius(
            pp, center, full_w, full_h, rect_x, rect_y, cx, cy, is_north, cap_deg
        )
        disk_r = max(floor_r, fill_r * 1.1)
        if disk_r < 1.0:
            continue

        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= disk_r * disk_r
        if disk.sum() == 0:
            continue
        # Guard: only seal a genuine pole-COVERING cap, never a high-latitude section
        # whose rectangular crop merely contains the pole pixel, or a wedge that reaches
        # the pole along some longitudes only. A real cap wraps the pole in every
        # direction, so require coverage present in most angular sectors around it (the
        # spokes still leave each sector partly covered) and a reasonable overall mean.
        covered = disk & (effective_mask_crop > 0.01)
        if float(covered[disk].mean()) < 0.2:  # coverage FRACTION, not feathered value
            continue
        ang = np.arctan2(yy[covered] - cy, xx[covered] - cx)
        sectors = np.unique(((ang + math.pi) / (2.0 * math.pi) * 8).astype(np.int64) % 8)
        if sectors.size < 6:  # coverage only on one side -> wedge/edge, not a cap
            continue
        effective_mask_crop[disk] = 1.0
        coverage_for_extend[disk] = 1.0

        # Collapse band = equirect rows the disk projects to. The disk reaches out to a
        # colatitude whose row index is colat/180*gh; find that colatitude by inverting
        # the colat->radius mapping, then pad slightly so the collapse covers the band.
        colat = cap_deg
        th = 0.5
        while th <= 4.0 * cap_deg:
            r_th = _colat_to_crop_radius(
                pp, center, full_w, full_h, rect_x, rect_y, cx, cy, is_north, th
            )
            if r_th >= disk_r:
                colat = th
                break
            colat = th
            th += 0.5
        # Pad the band so the faded RGB collapse comfortably covers the sealed footprint
        # (the alpha is already solid out to the disk; the extra rows just fade to 0).
        band_rows = max(1, int(round(colat / 180.0 * gh * 1.3)))
        sealed.append((is_north, band_rows))
    return sealed


def _collapse_pole_rows(
    img_eq: np.ndarray,
    mask_eq: np.ndarray,
    sealed: List[Tuple[bool, int]],
    *,
    gh: int,
    gamma: float = 2.0,
) -> None:
    """Blend each magnified near-pole equirect row toward its mask-weighted row average.
    Geographically a near-pole row is ~a single point, so this removes the longitudinal
    comb/streak aliasing while preserving the cap colour. Strength fades to 0 at the band
    edge. Operates in place on img_eq."""
    for is_north, band in sealed:
        band = min(band, gh)
        for i in range(band):
            r = i if is_north else (gh - 1 - i)
            w_blend = (1.0 - i / float(band)) ** gamma
            if w_blend <= 0.0:
                continue
            m = mask_eq[r, :, 0]
            cov = m > 0.01
            if int(cov.sum()) < 5:
                continue
            avg = img_eq[r][cov].mean(axis=0)
            img_eq[r] = (1.0 - w_blend) * img_eq[r] + w_blend * avg[None, :]


class PP_OT_reassemble(bpy.types.Operator):
    bl_idname = "pp.reassemble"
    bl_label = "Reassemble"
    bl_description = "Reassemble processed section crops back into a global equirectangular output"

    def execute(self, context: bpy.types.Context):
        s = context.scene.projection_pasta
        root = s.project_root_path()
        mp = s.manifest_path()
        if root is None or mp is None or not mp.exists():
            self.report({"ERROR"}, "manifest.json not found (set Project Root and Init Project)")
            return {"CANCELLED"}

        manifest = manifest_lib.read_manifest(mp)
        gsize = manifest.get("global", {}).get("size", [0, 0])
        gw, gh = int(gsize[0]), int(gsize[1])
        if gw <= 0 or gh <= 0:
            self.report({"ERROR"}, "Global size not set. Use Load World Map first.")
            return {"CANCELLED"}

        sections = manifest.get("sections", []) or []
        if not sections:
            self.report({"ERROR"}, "No sections in manifest.json")
            return {"CANCELLED"}

        # Group processed files by exact filename (including extension).
        groups: Dict[str, List[Tuple[dict, Path]]] = {}
        for sec in sections:
            sec_id = sec.get("id", "")
            if not sec_id:
                continue
            proc_dir = root / "processed" / sec_id
            if not proc_dir.exists():
                continue
            for p in proc_dir.iterdir():
                if not p.is_file():
                    continue
                fname = p.name
                groups.setdefault(fname, []).append((sec, p))

        # Find the target km/pixel ratio for resolution normalization
        # Use the finest (smallest km/pixel) across all sections that have processed files
        all_km_per_pixel: List[Tuple[str, float]] = []
        for sec in sections:
            sec_id = sec.get("id", "")
            size_info = sec.get("size_info", {})
            km_per_pixel = float(size_info.get("km_per_pixel", 0.0))
            if km_per_pixel > 0:
                proc_dir = root / "processed" / sec_id
                if proc_dir.exists() and any(proc_dir.iterdir()):
                    all_km_per_pixel.append((sec_id, km_per_pixel))
        
        target_km_per_pixel: float = 0.0
        if all_km_per_pixel:
            target_km_per_pixel = min(kpp for _, kpp in all_km_per_pixel)
            # Report if sections have different resolutions
            unique_kpp = set(round(kpp, 4) for _, kpp in all_km_per_pixel)
            if len(unique_kpp) > 1:
                print(f"[Project-R] Resolution normalization enabled. Target: {target_km_per_pixel:.4f} km/pixel")
                for sec_id, kpp in all_km_per_pixel:
                    if abs(kpp - target_km_per_pixel) > 0.0001:
                        scale_factor = kpp / target_km_per_pixel
                        print(f"[Project-R]   - {sec_id}: {kpp:.4f} km/pixel -> will scale by {scale_factor:.2f}x")

        if not groups:
            self.report({"ERROR"}, "No processed files found under processed/<section_id>/")
            return {"CANCELLED"}

        out_dir = root / "reassembled"
        out_dir.mkdir(parents=True, exist_ok=True)
        layers_dir = out_dir / "layers"
        layers_dir.mkdir(parents=True, exist_ok=True)

        # Get heightmap settings for elevation normalization
        heightmap_filename = manifest.get("global", {}).get("heightmap_filename", "")
        global_max_elevation_m = float(manifest.get("global", {}).get("max_elevation_m", 0.0))
        
        # Build a map of section_id -> section_max_elevation for heightmap normalization
        section_elevations: Dict[str, float] = {}
        if heightmap_filename and s.normalize_heightmaps:
            for sec in sections:
                sec_id = sec.get("id", "")
                elev_info = sec.get("elevation_info", {})
                if elev_info:
                    section_elevations[sec_id] = float(elev_info.get("section_max_elevation_m", 0.0))
            
            if section_elevations:
                # Find the global max elevation across all sections
                actual_global_max = max(section_elevations.values())
                print(f"[Project-R] Heightmap normalization enabled for '{heightmap_filename}'")
                print(f"[Project-R]   Global max elevation: {actual_global_max:.0f} m")
                for sec_id, elev in section_elevations.items():
                    scale = elev / actual_global_max if actual_global_max > 0 else 1.0
                    print(f"[Project-R]   - {sec_id}: {elev:.0f} m (scale: {scale:.3f})")

        warnings: List[str] = []
        layers_saved = 0

        for fname, entries in groups.items():
            is_mask = _is_mask_name(fname)
            interp = _interp_for_name(fname)
            treat_as_color = _treat_as_color_name(fname)
            
            # Create layers subdirectory for this filename
            fname_stem = Path(fname).stem
            fname_layers_dir = layers_dir / fname_stem
            fname_layers_dir.mkdir(parents=True, exist_ok=True)

            # Initialize accumulators
            base_layer: np.ndarray | None = None
            accumulated_mask: np.ndarray | None = None
            channels = 0

            for sec, crop_path in entries:
                sec_id = sec.get("id", "unknown")
                proj = sec.get("projection", {}) or {}
                params = ProjectionParams(
                    center_lon_deg=float(proj.get("center_lon_deg", 0.0)),
                    center_lat_deg=float(proj.get("center_lat_deg", 0.0)),
                    rot_deg=float(proj.get("rot_deg", 0.0)),
                )

                rect = sec.get("crop", {}).get("rect_xywh", [0, 0, 0, 0])
                x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
                full_size = sec.get("full_canvas", {}).get("size", [0, 0])
                full_w, full_h = int(full_size[0]), int(full_size[1])
                if full_w <= 0 or full_h <= 0 or w <= 0 or h <= 0:
                    warnings.append(f"{sec_id}/{fname}: invalid dimensions, skipping")
                    continue
                
                # Calculate scale factor for resolution normalization
                size_info = sec.get("size_info", {})
                sec_km_per_pixel = float(size_info.get("km_per_pixel", 0.0))
                scale_factor = 1.0
                if target_km_per_pixel > 0 and sec_km_per_pixel > 0:
                    # scale_factor > 1 means section is coarser and needs upscaling
                    scale_factor = sec_km_per_pixel / target_km_per_pixel

                # Load effective mask (single combined mask, half-res)
                masks_info = sec.get("masks", {}) or {}
                effective_rel = masks_info.get("effective_mask_path", "")
                res_scale = float(masks_info.get("resolution_scale", 0.5))

                # Load processed crop
                try:
                    crop_img = imaging.load_image(crop_path)
                except Exception as e:
                    warnings.append(f"{sec_id}/{fname}: failed to load - {e}")
                    continue

                # Users routinely re-export the crop at a different resolution in their
                # erosion/painting tool (Gaea/Wilbur even per section_info.txt's "Upscaled
                # Suggestions"), so the processed file rarely matches the manifest crop rect.
                # Everything downstream (mask, paste, extend_nearest_valid) is indexed at the
                # rect size, so resample the processed crop back to (w, h) here. Without this,
                # extend_nearest_valid silently samples only the rect-sized top-left corner of
                # an oversized image (pure Hammer background) -> a completely empty output.
                if crop_img.width != w or crop_img.height != h:
                    resized = imaging.resize_to(
                        crop_img.pixels, w, h,
                        interp="nearest" if is_mask else "linear",
                    )
                    crop_img = imaging.ImageBuffer(
                        width=w, height=h, channels=resized.shape[2], pixels=resized,
                    )

                if channels == 0:
                    channels = crop_img.channels

                # Load effective mask (half-res, already includes coverage + feathering)
                effective_mask_crop: np.ndarray
                coverage_for_extend: np.ndarray
                if effective_rel:
                    effective_path = (root / effective_rel).resolve()
                    if effective_path.exists():
                        eff_buf = imaging.load_image(effective_path)
                        eff_half = eff_buf.pixels[:, :, 0] if eff_buf.channels >= 1 else eff_buf.pixels.squeeze()
                        # Upscale to crop size
                        eff_full = imaging.resize_double_bilinear(eff_half[..., None])[:, :, 0]
                        # Clamp to crop size (may be off by 1 pixel)
                        eff_full = eff_full[:h, :w]
                        if eff_full.shape[0] < h or eff_full.shape[1] < w:
                            padded = np.zeros((h, w), dtype=np.float32)
                            padded[: eff_full.shape[0], : eff_full.shape[1]] = eff_full
                            eff_full = padded
                        effective_mask_crop = eff_full
                        # For extend_nearest_valid, use thresholded mask as coverage
                        coverage_for_extend = (eff_full > 0.01).astype(np.float32)
                    else:
                        # Mask file missing, use all-ones
                        effective_mask_crop = np.ones((h, w), dtype=np.float32)
                        coverage_for_extend = np.ones((h, w), dtype=np.float32)
                else:
                    # No mask info, use all-ones (backward compat)
                    effective_mask_crop = np.ones((h, w), dtype=np.float32)
                    coverage_for_extend = np.ones((h, w), dtype=np.float32)

                # Repair the geographic-pole singularity in the crop-space mask (no-op
                # unless this section actually covers a pole). See _seal_pole_starburst.
                sealed_poles = _seal_pole_starburst(
                    effective_mask_crop, coverage_for_extend,
                    params=params, full_w=full_w, full_h=full_h,
                    rect_x=x, rect_y=y, w=w, h=h, gh=gh,
                )

                # Apply nearest-valid extension to processed image
                crop_pixels = crop_img.pixels
                if crop_pixels.ndim == 2:
                    crop_pixels = crop_pixels[..., None]
                crop_filled = imaging.extend_nearest_valid(crop_pixels, coverage_for_extend)

                # Apply heightmap elevation normalization if this is the heightmap
                is_heightmap = heightmap_filename and fname == heightmap_filename
                if is_heightmap and s.normalize_heightmaps and sec_id in section_elevations:
                    section_elev = section_elevations[sec_id]
                    actual_global_max = max(section_elevations.values()) if section_elevations else 1.0
                    if actual_global_max > 0:
                        elev_scale = section_elev / actual_global_max
                        crop_filled = crop_filled.astype(np.float32) * elev_scale

                # Apply resolution normalization if needed
                if abs(scale_factor - 1.0) > 0.001:
                    # Resample the processed image to match target resolution
                    crop_filled = _resample_image(crop_filled.astype(np.float32), scale_factor)
                    effective_mask_crop = _resample_image(effective_mask_crop[..., None].astype(np.float32), scale_factor)[:, :, 0]
                    
                    # Adjust crop dimensions and position for the scaled image
                    new_w = crop_filled.shape[1]
                    new_h = crop_filled.shape[0]
                    
                    # Scale the full canvas proportionally
                    scaled_full_w = int(round(full_w * scale_factor))
                    scaled_full_h = int(round(full_h * scale_factor))
                    scaled_x = int(round(x * scale_factor))
                    scaled_y = int(round(y * scale_factor))
                else:
                    new_w, new_h = w, h
                    scaled_full_w, scaled_full_h = full_w, full_h
                    scaled_x, scaled_y = x, y

                # Uncrop to full Hammer canvas (use scaled dimensions if resampled)
                full_img = imaging.paste_into(
                    dst_size=(scaled_full_w, scaled_full_h),
                    dst_channels=crop_filled.shape[2],
                    src=imaging.ImageBuffer(
                        width=new_w,
                        height=new_h,
                        channels=crop_filled.shape[2],
                        pixels=crop_filled[:new_h, :new_w, :].astype(np.float32),
                    ),
                    rect_xywh=(scaled_x, scaled_y, new_w, new_h),
                )
                full_mask = imaging.paste_into(
                    dst_size=(scaled_full_w, scaled_full_h),
                    dst_channels=1,
                    src=imaging.ImageBuffer(
                        width=new_w,
                        height=new_h,
                        channels=1,
                        pixels=effective_mask_crop[:new_h, :new_w, None].astype(np.float32),
                    ),
                    rect_xywh=(scaled_x, scaled_y, new_w, new_h),
                )

                # Reproject to global equirect
                img_eq = project_hammer_array_to_equirect(
                    data_in=full_img.pixels,
                    dst_size=(gw, gh),
                    params=params,
                    interp=interp,  # type: ignore[arg-type]
                    treat_as_color=treat_as_color,
                )
                mask_eq = project_hammer_array_to_equirect(
                    data_in=full_mask.pixels,
                    dst_size=(gw, gh),
                    params=params,
                    interp="linear",
                )
                mask_eq = mask_eq[:, :, :1].astype("float32")
                if mask_eq.ndim == 2:
                    mask_eq = mask_eq[..., None]

                # Collapse the magnified pole rows to remove comb/streak aliasing
                # (no-op unless this section covers a pole). See _collapse_pole_rows.
                if sealed_poles:
                    _collapse_pole_rows(img_eq, mask_eq, sealed_poles, gh=gh)

                # Save individual section layer with transparency
                # Combine RGB(A) with mask as alpha for Photoshop layering
                layer_eq = img_eq.astype(np.float32)
                if layer_eq.shape[2] == 3:
                    # Add alpha channel from mask
                    layer_with_alpha = np.concatenate([layer_eq, mask_eq], axis=2)
                elif layer_eq.shape[2] == 4:
                    # Multiply existing alpha with mask
                    layer_with_alpha = layer_eq.copy()
                    layer_with_alpha[:, :, 3:4] *= mask_eq
                elif layer_eq.shape[2] == 1:
                    # Grayscale: convert to RGBA
                    layer_with_alpha = np.concatenate([
                        layer_eq, layer_eq, layer_eq, mask_eq
                    ], axis=2)
                else:
                    # Fallback: just append mask
                    layer_with_alpha = np.concatenate([layer_eq, mask_eq], axis=2)
                
                layer_path = fname_layers_dir / f"{sec_id}.png"
                layer_buf = imaging.ImageBuffer(
                    width=gw,
                    height=gh,
                    channels=layer_with_alpha.shape[2],
                    pixels=layer_with_alpha,
                )
                # Save as PNG with alpha (16-bit for non-color, 8-bit for color)
                layer_depth = "8" if treat_as_color else "16"
                imaging.save_image(layer_buf, layer_path, "PNG", color_depth=layer_depth)
                layers_saved += 1

                # Initialize base layer if needed
                if base_layer is None:
                    base_layer = np.zeros((gh, gw, img_eq.shape[2]), dtype=np.float32)
                    accumulated_mask = np.zeros((gh, gw, 1), dtype=np.float32)

                # Ensure channel count matches base layer
                img_channels = img_eq.shape[2]
                base_channels = base_layer.shape[2]
                if img_channels < base_channels:
                    # Pad with alpha=1 or zeros
                    padding = np.ones((gh, gw, base_channels - img_channels), dtype=np.float32)
                    img_eq = np.concatenate([img_eq, padding], axis=2)
                elif img_channels > base_channels:
                    # Expand base layer to match
                    padding = np.ones((gh, gw, img_channels - base_channels), dtype=np.float32)
                    base_layer = np.concatenate([base_layer, padding], axis=2)

                # Max-mask overlap: update where new mask > accumulated
                update = mask_eq > accumulated_mask
                update_bc = np.broadcast_to(update, base_layer.shape)
                base_layer = np.where(update_bc, img_eq.astype(np.float32), base_layer)
                accumulated_mask = np.where(update, mask_eq, accumulated_mask)

            # Save final output
            if base_layer is None:
                continue

            # Optionally extend edge colors to fill empty areas
            if s.extend_edge_colors and accumulated_mask is not None:
                # Coverage mask: 1 where we have data, 0 where empty
                coverage_2d = (accumulated_mask[:, :, 0] > 0.01).astype(np.float32)
                base_layer = imaging.extend_nearest_valid(base_layer, coverage_2d)

            out_path = out_dir / fname
            out_buf = imaging.ImageBuffer(
                width=gw,
                height=gh,
                channels=base_layer.shape[2],
                pixels=base_layer.astype(np.float32),
            )

            ext = out_path.suffix.lower()
            if ext == ".exr":
                fmt, depth = "OPEN_EXR", "32"
            elif ext == ".png":
                fmt, depth = "PNG", "8" if treat_as_color else "16"
            elif ext in (".jpg", ".jpeg"):
                fmt, depth = "JPEG", None
            else:
                fmt, depth = "PNG", "8" if treat_as_color else "16"

            imaging.save_image(out_buf, out_path, fmt, color_depth=depth)

        if warnings:
            for w in warnings:
                print(f"[Project-R] Warning: {w}")
            self.report({"WARNING"}, f"Reassembled with {len(warnings)} warning(s) (see console)")
        else:
            self.report({"INFO"}, f"Reassembled {len(groups)} file(s) + {layers_saved} layer(s) to {out_dir}")

        return {"FINISHED"}


_CLASSES = (
    PP_OT_validate_processed,
    PP_OT_reassemble,
)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
