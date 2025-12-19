from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

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
                print("[ProjectionPasta] Missing:", p)
        else:
            self.report({"INFO"}, f"Processed folders present; found {present} file(s) total")
        return {"FINISHED"}


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

        def _is_mask_name(name: str) -> bool:
            n = name.lower()
            return any(k in n for k in ("mask", "land", "plates", "labels"))

        def _is_height_name(name: str) -> bool:
            n = name.lower()
            return any(k in n for k in ("height", "elev", "dem"))

        def _interp_for_name(name: str) -> str:
            n = name.lower()
            if _is_mask_name(name):
                return "nearest"
            if _is_height_name(name):
                return "linear"
            return "linear"

        def _treat_as_color_name(name: str) -> bool:
            # Only apply sRGB<->linear conversions for typical color image formats.
            if _is_mask_name(name) or _is_height_name(name):
                return False
            ext = Path(name).suffix.lower()
            return ext in (".png", ".jpg", ".jpeg")

        out_dir = root / "reassembled"
        out_dir.mkdir(parents=True, exist_ok=True)

        for fname, entries in groups.items():
            is_mask = _is_mask_name(fname)
            interp = _interp_for_name(fname)
            treat_as_color = _treat_as_color_name(fname)

            # Initialize accumulators lazily based on first image's channels.
            sum_img = None
            sum_w = None
            best_w = None
            out_mask_value = None

            for sec, crop_path in entries:
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
                    continue

                weight_rel = sec.get("reassembly", {}).get("weight_mask_path", "")
                if not weight_rel:
                    continue
                weight_path = (root / weight_rel).resolve()
                if not weight_path.exists():
                    continue

                crop_img = imaging.load_image(crop_path)
                crop_w = imaging.load_image(weight_path)

                # Uncrop both into full Hammer canvas.
                full_img = imaging.paste_into(
                    dst_size=(full_w, full_h),
                    dst_channels=crop_img.channels,
                    src=crop_img,
                    rect_xywh=(x, y, w, h),
                )
                full_weight = imaging.paste_into(
                    dst_size=(full_w, full_h),
                    dst_channels=1,
                    src=imaging.ImageBuffer(
                        width=crop_w.width,
                        height=crop_w.height,
                        channels=1,
                        pixels=crop_w.pixels[:, :, :1],
                    ),
                    rect_xywh=(x, y, w, h),
                )

                # Reproject to global equirect.
                img_eq = project_hammer_array_to_equirect(
                    data_in=full_img.pixels,
                    dst_size=(gw, gh),
                    params=params,
                    interp=interp,  # type: ignore[arg-type]
                    treat_as_color=treat_as_color,
                )
                w_eq = project_hammer_array_to_equirect(
                    data_in=full_weight.pixels[:, :, :1],
                    dst_size=(gw, gh),
                    params=params,
                    interp="nearest",
                )
                w_eq = w_eq[:, :, :1].astype("float32")

                if is_mask:
                    if best_w is None:
                        best_w = np.full((gh, gw, 1), -1.0, dtype="float32")
                        out_mask_value = np.zeros_like(img_eq, dtype="float32")
                    take = w_eq > best_w
                    out_mask_value = np.where(take, img_eq.astype("float32"), out_mask_value)
                    best_w = np.where(take, w_eq, best_w)
                else:
                    if sum_img is None:
                        sum_img = np.zeros_like(img_eq, dtype="float32")
                        sum_w = np.zeros((gh, gw, 1), dtype="float32")
                    sum_img += img_eq.astype("float32") * w_eq
                    sum_w += w_eq

            out_path = out_dir / fname
            if is_mask:
                if out_mask_value is None:
                    continue
                out_buf = imaging.ImageBuffer(width=gw, height=gh, channels=out_mask_value.shape[2] if out_mask_value.ndim == 3 else 1, pixels=out_mask_value.astype("float32"))
                fmt, depth = ("PNG", "16") if out_path.suffix.lower() == ".png" else ("OPEN_EXR", "32") if out_path.suffix.lower() == ".exr" else ("PNG", None)
                imaging.save_image(out_buf, out_path, fmt, color_depth=depth)
            else:
                if sum_img is None or sum_w is None:
                    continue
                denom = np.maximum(sum_w, 1e-8)
                out = sum_img / denom
                out_buf = imaging.ImageBuffer(width=gw, height=gh, channels=out.shape[2] if out.ndim == 3 else 1, pixels=out.astype("float32"))
                fmt, depth = (
                    ("OPEN_EXR", "32")
                    if out_path.suffix.lower() == ".exr"
                    else ("PNG", "8" if treat_as_color else "16")
                    if out_path.suffix.lower() == ".png"
                    else ("JPEG", None)
                    if out_path.suffix.lower() in (".jpg", ".jpeg")
                    else ("PNG", "8" if treat_as_color else "16")
                )
                imaging.save_image(out_buf, out_path, fmt, color_depth=depth)

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


