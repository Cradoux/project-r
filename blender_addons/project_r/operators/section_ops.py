from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import List, Tuple

import bpy
import bmesh
import numpy as np

from .. import geo
from .. import imaging
from .. import manifest as manifest_lib
from ..projection_backend import ProjectionParams, project_equirect_to_hammer
from ..vendor.projectionpasta import projectionpasta as pp


def _selected_uv_lonlats(context: bpy.types.Context) -> List[geo.LonLat]:
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        return []
    if context.mode != "EDIT_MESH":
        return []
    me = obj.data
    uv_layer = me.uv_layers.active
    if uv_layer is None:
        return []

    bm = bmesh.from_edit_mesh(me)
    uv = bm.loops.layers.uv.active
    if uv is None:
        return []

    pts: List[geo.LonLat] = []
    for f in bm.faces:
        if not f.select:
            continue
        for loop in f.loops:
            u, v = loop[uv].uv
            pts.append(geo.uv_to_lonlat(float(u), float(v)))
    return pts


def _selected_face_indices(context: bpy.types.Context) -> List[int]:
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        return []
    if context.mode != "EDIT_MESH":
        return []
    bm = bmesh.from_edit_mesh(obj.data)
    return [int(f.index) for f in bm.faces if f.select]


def _compute_crop_rect(
    *,
    lonlats: List[geo.LonLat],
    center: geo.LonLat,
    rot_rad: float,
    full_size: Tuple[int, int],
    margin_px: int,
    square: bool,
) -> geo.RectI:
    w, h = full_size
    # IMPORTANT: Use projectionpasta's own forward projection for Hammer so
    # crop bounds match exactly what we render in hammer_full.
    lons = np.array([p.lon for p in lonlats], dtype=np.float64)
    lats = np.array([p.lat for p in lonlats], dtype=np.float64)

    aspect = np.array([center.lon, center.lat, rot_rad], dtype=np.float64)
    lon_r, lat_r = pp.Rotate_to(lons, lats, aspect)

    opts = dict(pp.def_opts)
    opts["in"] = False
    x, y = pp.posl["Hammer"](lon_r, lat_r, opts)  # x,y are in [-1,1]

    xs = ((x + 1.0) / 2.0 * w - 0.5).tolist()
    ys = ((1.0 - y) / 2.0 * h - 0.5).tolist()

    if not xs:
        return geo.RectI(0, 0, w, h)

    x0 = int(math.floor(min(xs))) - margin_px
    y0 = int(math.floor(min(ys))) - margin_px
    x1 = int(math.ceil(max(xs))) + 1 + margin_px
    y1 = int(math.ceil(max(ys))) + 1 + margin_px

    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(1, min(x1, w))
    y1 = max(1, min(y1, h))

    rect = geo.RectI(x=x0, y=y0, w=max(1, x1 - x0), h=max(1, y1 - y0))
    if square:
        rect = geo.expand_rect_square(rect, w, h)
    return rect


class PP_OT_create_section(bpy.types.Operator):
    bl_idname = "pp.create_section"
    bl_label = "Create Section"
    bl_description = "Create a Hammer (oblique) section crop from the selected faces and record it in manifest.json"

    def execute(self, context: bpy.types.Context):
        s = context.scene.projection_pasta
        root = s.project_root_path()
        mp = s.manifest_path()
        if root is None or mp is None:
            self.report({"ERROR"}, "Project Root is not set")
            return {"CANCELLED"}
        if not mp.exists():
            self.report({"ERROR"}, "manifest.json does not exist (run Init Project)")
            return {"CANCELLED"}

        lonlats = _selected_uv_lonlats(context)
        if len(lonlats) < 3:
            self.report({"ERROR"}, "Select some faces on the sphere in Edit Mode")
            return {"CANCELLED"}

        face_indices = _selected_face_indices(context)
        center = geo.mean_center_lonlat(lonlats)
        params = ProjectionParams(
            center_lon_deg=math.degrees(center.lon),
            center_lat_deg=math.degrees(center.lat),
            rot_deg=0.0,
        )

        manifest = manifest_lib.read_manifest(mp)
        global_size = manifest.get("global", {}).get("size", [0, 0])
        if not global_size or global_size[0] <= 0 or global_size[1] <= 0:
            self.report({"ERROR"}, "Global size not set. Use Load World Map first.")
            return {"CANCELLED"}

        sec_id = f"sec_{len(manifest.get('sections', [])) + 1:03d}_{s.new_section_name.lower()}"
        (root / "sections" / sec_id).mkdir(parents=True, exist_ok=True)
        (root / "processed" / sec_id).mkdir(parents=True, exist_ok=True)

        full_w = int(s.hammer_full_width)
        full_h = int(s.hammer_full_height)
        rect = _compute_crop_rect(
            lonlats=lonlats,
            center=center,
            rot_rad=0.0,
            full_size=(full_w, full_h),
            margin_px=int(s.crop_margin_px),
            square=bool(s.square_crop),
        )

        # Export all configured global layers to Hammer full + crop.
        layers = manifest.get("global", {}).get("layers", []) or []
        full_paths: dict[str, str] = {}
        crop_paths: dict[str, str] = {}

        def _interp_for_layer(layer_id: str, filename: str) -> str:
            name = (layer_id + " " + filename).lower()
            if any(k in name for k in ("mask", "land", "plates", "labels")):
                return "nearest"
            if any(k in name for k in ("height", "elev", "dem")):
                return "linear"
            return "linear"

        def _treat_as_color(layer_id: str, filename: str) -> bool:
            name = (layer_id + " " + filename).lower()
            if any(k in name for k in ("mask", "land", "plates", "labels")):
                return False
            if any(k in name for k in ("height", "elev", "dem")):
                return False
            # Treat typical PNG/JPG albedo-like maps as sRGB.
            return True

        for layer in layers:
            layer_id = str(layer.get("id", "layer")).strip() or "layer"
            src = str(layer.get("path", "")).strip()
            if not src:
                continue

            src_path = Path(src)
            if not src_path.is_absolute():
                src_path = (root / src_path).resolve()

            ext = src_path.suffix.lower() or ".png"
            full_rel = Path("sections") / sec_id / f"{layer_id}__hammer_full{ext}"
            crop_rel = Path("sections") / sec_id / f"{layer_id}__crop{ext}"
            full_path = (root / full_rel).resolve()
            crop_path = (root / crop_rel).resolve()

            interp = _interp_for_layer(layer_id, src_path.name)
            treat_as_color = _treat_as_color(layer_id, src_path.name) and ext in (".png", ".jpg", ".jpeg")
            project_equirect_to_hammer(
                src_path=src_path,
                dst_path=full_path,
                dst_size=(full_w, full_h),
                params=params,
                interp=interp,  # type: ignore[arg-type]
                treat_as_color=treat_as_color,
            )

            full_img = imaging.load_image(full_path)
            crop_img = imaging.crop(full_img, rect.x, rect.y, rect.w, rect.h)
            # Save crop with same format as full_path
            fmt, depth = (
                ("OPEN_EXR", "32")
                if ext == ".exr"
                else ("PNG", "8" if treat_as_color else "16")
                if ext == ".png"
                else ("JPEG", None)
            )
            imaging.save_image(crop_img, crop_path, fmt, color_depth=depth)

            full_paths[layer_id] = str(full_rel).replace("\\", "/")
            crop_paths[layer_id] = str(crop_rel).replace("\\", "/")

        # Weight mask (always PNG)
        weight = imaging.make_feather_weight_mask(rect.w, rect.h, int(s.feather_px))
        weight_rel = Path("sections") / sec_id / "weightmask__crop.png"
        imaging.save_image(weight, root / weight_rel, "PNG", color_depth="16")

        manifest.setdefault("sections", []).append(
            {
                "id": sec_id,
                "name": s.new_section_name,
                "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "projection": {
                    "type": "Hammer",
                    "center_lon_deg": params.center_lon_deg,
                    "center_lat_deg": params.center_lat_deg,
                    "rot_deg": params.rot_deg,
                },
                "full_canvas": {
                    "size": [int(s.hammer_full_width), int(s.hammer_full_height)],
                    "path_by_layer": full_paths,
                },
                "crop": {
                    "used_square": bool(s.square_crop),
                    "rect_xywh": [int(rect.x), int(rect.y), int(rect.w), int(rect.h)],
                    "margin_px": int(s.crop_margin_px),
                    "paths_by_layer": crop_paths,
                },
                "processed": {"expected_paths_by_layer": {}},
                "reassembly": {
                    "feather_px": int(s.feather_px),
                    "weight_mask_path": str(weight_rel).replace("\\", "/"),
                },
                "created_from_selection": {
                    "mesh_object": context.active_object.name if context.active_object else "",
                    "selected_face_indices": face_indices,
                    "selection_uv_sample_count": len(lonlats),
                },
            }
        )

        manifest_lib.write_manifest(mp, manifest)
        self.report(
            {"INFO"},
            f"Created section {sec_id} (center: {params.center_lon_deg:.2f}, {params.center_lat_deg:.2f})",
        )
        return {"FINISHED"}


_CLASSES = (PP_OT_create_section,)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)


