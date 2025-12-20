from __future__ import annotations

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

        if not groups:
            self.report({"ERROR"}, "No processed files found under processed/<section_id>/")
            return {"CANCELLED"}

        out_dir = root / "reassembled"
        out_dir.mkdir(parents=True, exist_ok=True)

        warnings: List[str] = []

        for fname, entries in groups.items():
            is_mask = _is_mask_name(fname)
            interp = _interp_for_name(fname)
            treat_as_color = _treat_as_color_name(fname)

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

                # Apply nearest-valid extension to processed image
                crop_pixels = crop_img.pixels
                if crop_pixels.ndim == 2:
                    crop_pixels = crop_pixels[..., None]
                crop_filled = imaging.extend_nearest_valid(crop_pixels, coverage_for_extend)

                # Uncrop to full Hammer canvas
                full_img = imaging.paste_into(
                    dst_size=(full_w, full_h),
                    dst_channels=crop_filled.shape[2],
                    src=imaging.ImageBuffer(
                        width=w,
                        height=h,
                        channels=crop_filled.shape[2],
                        pixels=crop_filled[:h, :w, :],
                    ),
                    rect_xywh=(x, y, w, h),
                )
                full_mask = imaging.paste_into(
                    dst_size=(full_w, full_h),
                    dst_channels=1,
                    src=imaging.ImageBuffer(
                        width=w,
                        height=h,
                        channels=1,
                        pixels=effective_mask_crop[:h, :w, None],
                    ),
                    rect_xywh=(x, y, w, h),
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

                # Initialize base layer if needed
                if base_layer is None:
                    base_layer = np.zeros((gh, gw, img_eq.shape[2]), dtype=np.float32)
                    accumulated_mask = np.zeros((gh, gw, 1), dtype=np.float32)

                # Max-mask overlap: update where new mask > accumulated
                update = mask_eq > accumulated_mask
                update_bc = np.broadcast_to(update, base_layer.shape)
                base_layer = np.where(update_bc, img_eq.astype(np.float32), base_layer)
                accumulated_mask = np.where(update, mask_eq, accumulated_mask)

            # Save final output
            if base_layer is None:
                continue

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
            self.report({"INFO"}, f"Reassembled {len(groups)} file(s) to {out_dir}")

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
