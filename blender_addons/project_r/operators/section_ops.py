from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import List, Tuple

import bpy
import bmesh
import numpy as np
import random

from .. import geo
from .. import imaging
from .. import manifest as manifest_lib
from ..projection_backend import ProjectionParams, project_equirect_to_hammer
from ..vendor.projectionpasta import projectionpasta as pp
from . import sphere_ops


def _point_in_triangle_2d(
    px: np.ndarray,
    py: np.ndarray,
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float,
) -> np.ndarray:
    """
    Vectorized check if points (px, py) are inside triangle ABC.
    Returns boolean mask.
    """
    def sign(p1x, p1y, p2x, p2y, p3x, p3y):
        return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)

    d1 = sign(px, py, ax, ay, bx, by)
    d2 = sign(px, py, bx, by, cx, cy)
    d3 = sign(px, py, cx, cy, ax, ay)

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    return ~(has_neg & has_pos)


def _rasterize_coverage_mask(
    *,
    triangles_uv: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
    center: geo.LonLat,
    rot_rad: float,
    full_size: Tuple[int, int],
    crop_rect: geo.RectI,
    half_res: bool = True,
) -> np.ndarray:
    """
    Rasterize selected face triangles into a coverage mask in Hammer crop space.
    
    For each pixel in the crop:
    1. Compute its Hammer x,y coordinates
    2. Inverse-project to lon/lat
    3. Convert to equirect UV
    4. Check if inside any selected triangle
    
    Returns: (H, W) float32 array with 1.0 where selected, 0.0 elsewhere.
    """
    full_w, full_h = full_size
    cx, cy, cw, ch = crop_rect.x, crop_rect.y, crop_rect.w, crop_rect.h

    # Output size (half res if requested)
    if half_res:
        out_w, out_h = cw // 2, ch // 2
    else:
        out_w, out_h = cw, ch

    if out_w <= 0 or out_h <= 0:
        return np.zeros((max(1, out_h), max(1, out_w)), dtype=np.float32)

    # Create pixel grid for the crop
    # Map crop pixel coords to full canvas coords, then to Hammer projection coords
    crop_x = np.linspace(0.5, out_w - 0.5, out_w, dtype=np.float64)
    crop_y = np.linspace(0.5, out_h - 0.5, out_h, dtype=np.float64)
    crop_xx, crop_yy = np.meshgrid(crop_x, crop_y)

    # Scale crop coords to full canvas coords
    if half_res:
        full_xx = cx + crop_xx * 2.0
        full_yy = cy + crop_yy * 2.0
    else:
        full_xx = cx + crop_xx
        full_yy = cy + crop_yy

    # Convert full canvas pixel coords to Hammer projection space [-1, 1]
    hammer_x = (full_xx + 0.5) / full_w * 2.0 - 1.0
    hammer_y = 1.0 - (full_yy + 0.5) / full_h * 2.0

    # Inverse Hammer projection: get lon/lat from hammer_x, hammer_y
    # Using projectionpasta's inverse projection
    opts = dict(pp.def_opts)
    opts["in"] = True  # inverse direction

    aspect = np.array([center.lon, center.lat, rot_rad], dtype=np.float64)

    # projectionpasta Hammer inverse: x,y -> lon,lat
    lon_r, lat_r = pp.posl["Hammer"](hammer_x.ravel(), hammer_y.ravel(), opts)

    # Rotate back from centered coords
    lon, lat = pp.Rotate_from(lon_r, lat_r, aspect)

    lon = lon.reshape(out_h, out_w)
    lat = lat.reshape(out_h, out_w)

    # Check which pixels are outside valid Hammer projection (NaN or outside range)
    valid_proj = np.isfinite(lon) & np.isfinite(lat)
    
    print(f"[Project-R] Coverage mask: valid_proj has {np.sum(valid_proj)} / {valid_proj.size} valid pixels")
    if np.any(valid_proj):
        print(f"[Project-R] lon range: {np.nanmin(lon):.4f} to {np.nanmax(lon):.4f}")
        print(f"[Project-R] lat range: {np.nanmin(lat):.4f} to {np.nanmax(lat):.4f}")

    # Convert lon/lat to UV
    # lon in radians: -pi..pi -> U 0..1
    # lat in radians: -pi/2..pi/2 -> V 0..1
    u = (lon / (2 * np.pi) + 0.5) % 1.0
    v = (lat / np.pi + 0.5)

    if np.any(valid_proj):
        valid_u = u[valid_proj]
        valid_v = v[valid_proj]
        print(f"[Project-R] UV range: u=[{np.min(valid_u):.4f}, {np.max(valid_u):.4f}], v=[{np.min(valid_v):.4f}, {np.max(valid_v):.4f}]")
    
    # Show first triangle UV for comparison
    if triangles_uv:
        tri0 = triangles_uv[0]
        print(f"[Project-R] First triangle UVs: {tri0}")

    # Now check if each UV point falls inside any selected triangle
    mask = np.zeros((out_h, out_w), dtype=np.float32)

    for tri in triangles_uv:
        (u0, v0), (u1, v1), (u2, v2) = tri

        # Handle seam-crossing triangles
        us = [u0, u1, u2]
        span = max(us) - min(us)

        if span > 0.5:
            # Triangle crosses U seam; test both shifted versions
            # Variant A: shift low U values up
            tri_a = [(uu + 1.0 if uu < 0.5 else uu, vv) for (uu, vv) in tri]
            # Variant B: shift high U values down
            tri_b = [(uu - 1.0 if uu > 0.5 else uu, vv) for (uu, vv) in tri]

            # Test shifted UVs
            u_shifted_a = np.where(u < 0.5, u + 1.0, u)
            u_shifted_b = np.where(u > 0.5, u - 1.0, u)

            inside_a = _point_in_triangle_2d(
                u_shifted_a, v,
                tri_a[0][0], tri_a[0][1],
                tri_a[1][0], tri_a[1][1],
                tri_a[2][0], tri_a[2][1],
            )
            inside_b = _point_in_triangle_2d(
                u_shifted_b, v,
                tri_b[0][0], tri_b[0][1],
                tri_b[1][0], tri_b[1][1],
                tri_b[2][0], tri_b[2][1],
            )
            inside = inside_a | inside_b
        else:
            inside = _point_in_triangle_2d(
                u, v,
                u0, v0,
                u1, v1,
                u2, v2,
            )

        mask = np.where(inside & valid_proj, 1.0, mask)

    coverage_count = int(np.sum(mask > 0.5))
    print(f"[Project-R] Coverage mask: {coverage_count} pixels covered out of {mask.size}")
    
    return mask.astype(np.float32)


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


def _grow_faces(seed: set[bmesh.types.BMFace], rings: int) -> set[bmesh.types.BMFace]:
    rings = max(0, int(rings))
    grown: set[bmesh.types.BMFace] = set(seed)
    frontier: set[bmesh.types.BMFace] = set(seed)
    for _ in range(rings):
        new_frontier: set[bmesh.types.BMFace] = set()
        for f in frontier:
            for e in f.edges:
                for nb in e.link_faces:
                    if nb not in grown:
                        grown.add(nb)
                        new_frontier.add(nb)
        frontier = new_frontier
        if not frontier:
            break
    return grown


def _gather_uv_triangles_for_faces(
    bm: bmesh.types.BMesh,
    faces: set[bmesh.types.BMFace],
    uv_layer,
) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    tris: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    for f in faces:
        loops = list(f.loops)
        if len(loops) < 3:
            continue
        u0, v0 = loops[0][uv_layer].uv
        p0 = (float(u0), float(v0))
        for i in range(1, len(loops) - 1):
            u1, v1 = loops[i][uv_layer].uv
            u2, v2 = loops[i + 1][uv_layer].uv
            tris.append((p0, (float(u1), float(v1)), (float(u2), float(v2))))
    return tris


def _face_uv_center(loop_uvs: list[tuple[float, float]]) -> tuple[float, float]:
    us = [u for (u, _) in loop_uvs]
    vs = [v for (_, v) in loop_uvs]
    if not us:
        return (0.0, 0.0)
    if max(us) - min(us) > 0.5:
        # Seam-crossing face: shift low-u up for averaging.
        us2 = [u + (1.0 if u < 0.5 else 0.0) for u in us]
        u = (sum(us2) / len(us2)) % 1.0
    else:
        u = (sum(us) / len(us)) % 1.0
    v = sum(vs) / len(vs)
    return (u, v)


def _gather_uv_centers_for_faces(
    bm: bmesh.types.BMesh,
    faces: set[bmesh.types.BMFace],
    uv_layer,
) -> list[tuple[float, float]]:
    centers: list[tuple[float, float]] = []
    for f in faces:
        luv = []
        for loop in f.loops:
            u, v = loop[uv_layer].uv
            luv.append((float(u), float(v)))
        centers.append(_face_uv_center(luv))
    return centers


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

        # Gather UV triangles NOW while bmesh is definitely valid (for coverage mask later)
        selected_triangles_uv: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = []
        try:
            obj = context.active_object
            if obj is not None and obj.type == "MESH" and context.mode == "EDIT_MESH":
                bm = bmesh.from_edit_mesh(obj.data)
                uv_layer = bm.loops.layers.uv.active
                if uv_layer is not None:
                    seed_faces = {f for f in bm.faces if f.select}
                    selected_triangles_uv = _gather_uv_triangles_for_faces(bm, seed_faces, uv_layer)
        except Exception as e:
            print(f"[Project-R] Warning: could not gather UV triangles: {e}")
            selected_triangles_uv = []
        
        print(f"[Project-R] Gathered {len(selected_triangles_uv)} UV triangles from selection")
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

        # ---- Update extracted overlay (per-section color) ----
        try:
            obj = context.active_object
            if obj is not None and obj.type == "MESH" and context.mode == "EDIT_MESH":
                bm = bmesh.from_edit_mesh(obj.data)
                uv_layer = bm.loops.layers.uv.active
                if uv_layer is not None:
                    seed_faces = {f for f in bm.faces if f.select}
                    grown_faces = _grow_faces(seed_faces, int(s.expand_selection_rings))

                    centers_uv = _gather_uv_centers_for_faces(bm, grown_faces, uv_layer)

                    # Deterministic per-section color
                    rr = random.Random(sec_id)
                    col = (
                        int(80 + rr.random() * 175),
                        int(80 + rr.random() * 175),
                        int(80 + rr.random() * 175),
                        160,  # alpha
                    )

                    overlay_rel = "project_r_overlay.png"
                    overlay_path = (root / overlay_rel).resolve()
                    ow, oh = int(global_size[0]), int(global_size[1])
                    overlay = imaging.load_or_create_overlay_rgba_u8(overlay_path, (ow, oh))
                    # Draw markers at face centers instead of filling faces.
                    radius_px = max(2, int(min(ow, oh) / 512))
                    overlay = imaging.paint_uv_circles_on_overlay(
                        overlay,
                        centers_uv=centers_uv,
                        radius_px=radius_px,
                        color_rgba_u8=col,
                    )
                    imaging.save_overlay_rgba_u8(overlay_path, overlay)

                    manifest.setdefault("global", {})["overlay"] = {
                        "path": overlay_rel,
                        "size": [ow, oh],
                    }

                    # If the overlay image is loaded in Blender, reload it so it updates on the sphere.
                    for img in bpy.data.images:
                        try:
                            fp = bpy.path.abspath(img.filepath_raw)
                        except Exception:
                            fp = img.filepath_raw
                        if fp and Path(fp).resolve() == overlay_path:
                            img.reload()
                            break

                    overlay_color_rgba = [c / 255.0 for c in col]
                else:
                    overlay_color_rgba = None
            else:
                overlay_color_rgba = None
        except Exception:
            overlay_color_rgba = None
        else:
            # If we created/updated overlay info, ensure the sphere material is wired to show it.
            try:
                if "overlay_path" in locals():
                    sphere_ops.ensure_overlay_connected_with_paths(context, overlay_path=overlay_path)
                else:
                    sphere_ops.ensure_overlay_connected(context)
            except Exception:
                pass

        # Use the world map size for Hammer full dimensions
        full_w = int(global_size[0])
        full_h = int(global_size[1])
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

        # ---- Generate Coverage Mask (half-res) ----
        # Use triangles gathered at the start of execute (when bmesh was valid)
        print(f"[Project-R] Rasterizing coverage mask with {len(selected_triangles_uv)} triangles...")

        coverage_mask = _rasterize_coverage_mask(
            triangles_uv=selected_triangles_uv,
            center=center,
            rot_rad=0.0,
            full_size=(full_w, full_h),
            crop_rect=rect,
            half_res=True,
        )
        coverage_rel = Path("sections") / sec_id / "coverage_mask__crop.png"
        coverage_path = root / coverage_rel
        coverage_buf = imaging.ImageBuffer(
            width=coverage_mask.shape[1],
            height=coverage_mask.shape[0],
            channels=1,
            pixels=coverage_mask[..., None],
        )
        imaging.save_image(coverage_buf, coverage_path, "PNG", color_depth="8")

        # ---- Generate Feather Mask (half-res, from coverage) ----
        feather_mask = imaging.generate_feather_mask_from_coverage(
            coverage_mask,
            feather_px=int(s.feather_px) // 2,  # Scale feather for half-res
        )
        feather_rel = Path("sections") / sec_id / "feather_mask__crop.png"
        feather_path = root / feather_rel
        feather_buf = imaging.ImageBuffer(
            width=feather_mask.shape[1],
            height=feather_mask.shape[0],
            channels=1,
            pixels=feather_mask[..., None],
        )
        imaging.save_image(feather_buf, feather_path, "PNG", color_depth="16")

        # Legacy weight mask (edge-based, for backward compat)
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
                    "size": [full_w, full_h],
                    "path_by_layer": full_paths,
                },
                "crop": {
                    "used_square": bool(s.square_crop),
                    "rect_xywh": [int(rect.x), int(rect.y), int(rect.w), int(rect.h)],
                    "margin_px": int(s.crop_margin_px),
                    "paths_by_layer": crop_paths,
                },
                "processed": {"expected_paths_by_layer": {}},
                "masks": {
                    "coverage_path": str(coverage_rel).replace("\\", "/"),
                    "feather_path": str(feather_rel).replace("\\", "/"),
                    "resolution_scale": 0.5,
                },
                "reassembly": {
                    "feather_px": int(s.feather_px),
                    "weight_mask_path": str(weight_rel).replace("\\", "/"),
                },
                "created_from_selection": {
                    "mesh_object": context.active_object.name if context.active_object else "",
                    "selected_face_indices": face_indices,
                    "selection_uv_sample_count": len(lonlats),
                },
                "overlay_color_rgba": overlay_color_rgba,
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


