"""Map Inputs: an optional, consolidated place to load the source maps Project-R
consumes, plus folder auto-detection for a consistent export set (e.g. Gleba's), and
a categorical -> per-class B&W mask exporter (Gaea downstream masks).

Design notes
------------
Section creation reprojects+crops EVERY image in ``source/`` into each section, then
the erosion stage finds the heightmap/rainfall crop by filename. So "loading a map"
just means getting it into ``source/`` and recording which filename plays which role.

A colormap-encoded map (viridis rainfall) must be DECODED to a single-channel field
before it enters ``source/`` -- otherwise the crop stays RGB and the erosion stage
reads its red channel as "rainfall", which is wrong. ``_ingest_map`` handles that:
genuine colormap RGB is inverted to a 16-bit grayscale ``<stem>__decoded.png``; an
already-grayscale (or RGB-but-gray) map is copied as-is.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import bpy
import numpy as np
from bpy.types import Operator

from .. import decode
from .. import imaging
from .. import layers
from .. import manifest as manifest_lib
from ..projection_backend import ProjectionParams, project_equirect_array_to_hammer
from . import erode_ops


def _source_dir(context) -> Optional[Path]:
    s = context.scene.projection_pasta
    root = s.project_root_path()
    if root is None:
        return None
    return root / "source"


def _is_effectively_gray(px: np.ndarray) -> bool:
    """True if an (H,W,C>=3) array is colour-flat (R==G==B almost everywhere) and so
    needs no colormap decode."""
    if px.ndim < 3 or px.shape[2] < 3:
        return True
    rgb = px[:, :, :3]
    spread = rgb.max(axis=-1) - rgb.min(axis=-1)
    return float((spread < (2.0 / 255.0)).mean()) > 0.99


def _ingest_map(root: Path, src_path: Path, *, decode_colormap: bool) -> str:
    """Bring a map into ``source/`` and return the filename to record.

    ``decode_colormap`` (for the rainfall/uplift roles): a true colormap RGB image is
    inverted via the baked viridis LUT to a 16-bit single-channel ``<stem>__decoded.png``;
    everything else is copied unchanged. Idempotent: re-ingesting overwrites the cache.
    """
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    src_path = Path(src_path)

    if decode_colormap:
        buf = imaging.load_image(src_path)
        if buf.channels >= 3 and not _is_effectively_gray(buf.pixels):
            scalar = decode.decode_colormap_to_scalar(buf.pixels)
            out = source / f"{src_path.stem}__decoded.png"
            out_buf = imaging.ImageBuffer(width=scalar.shape[1], height=scalar.shape[0],
                                          channels=1, pixels=scalar[..., None].astype(np.float32))
            imaging.save_image(out_buf, out, "PNG", color_depth="16")
            return out.name

    dst = source / src_path.name
    try:
        same = dst.exists() and src_path.resolve() == dst.resolve()
    except OSError:
        same = False
    if not same:
        shutil.copy2(src_path, dst)
    return dst.name


# Slot -> (where the filename lives). Only the roles Project-R consumes today are
# wired; the rest are surfaced by auto-detect as suggestions for a later pass.
def _set_slot_filename(context, slot: str, filename: str) -> None:
    s = context.scene.projection_pasta
    es = context.scene.projection_pasta_erosion
    if slot == "heightmap":
        s.heightmap_filename = filename
    elif slot == "rainfall":
        es.rainfall_filename = filename


class PP_OT_set_input_map(Operator):
    bl_idname = "pp.set_input_map"
    bl_label = "Set Input Map"
    bl_description = ("Pick a map file (anywhere on disk) for this input slot. It is copied into "
                     "the project's source/ folder so sections crop it; colormap maps are decoded "
                     "to a single channel first")

    slot: bpy.props.EnumProperty(  # type: ignore[valid-type]
        items=[
            ("heightmap", "Heightmap", "Single-channel elevation (brighter = higher)"),
            ("rainfall", "Rainfall", "Runoff weight (colormap maps are decoded to a scalar)"),
        ],
        default="heightmap",
        options={"SKIP_SAVE"},
    )
    filepath: bpy.props.StringProperty(subtype="FILE_PATH", default="", options={"SKIP_SAVE"})  # type: ignore[valid-type]
    clear: bpy.props.BoolProperty(default=False, options={"SKIP_SAVE"})  # type: ignore[valid-type]

    @classmethod
    def poll(cls, context) -> bool:
        s = getattr(context.scene, "projection_pasta", None)
        mp = s.manifest_path() if s is not None else None
        if mp is None or not mp.exists():
            cls.poll_message_set("Open or create a project first")
            return False
        return True

    def invoke(self, context, event):
        if self.clear:
            return self.execute(context)
        src = _source_dir(context)
        if src is not None and src.exists():
            self.filepath = str(src) + "\\"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if self.clear:
            _set_slot_filename(context, self.slot, "")
            self.report({"INFO"}, f"{self.slot.title()} cleared")
            return {"FINISHED"}
        if not self.filepath:
            self.report({"ERROR"}, "No file selected")
            return {"CANCELLED"}
        root = context.scene.projection_pasta.project_root_path()
        if root is None:
            self.report({"ERROR"}, "Project Root is not set")
            return {"CANCELLED"}
        try:
            name = _ingest_map(root, Path(self.filepath), decode_colormap=(self.slot == "rainfall"))
        except Exception as e:
            self.report({"ERROR"}, f"Failed to load map: {e}")
            return {"CANCELLED"}
        _set_slot_filename(context, self.slot, name)
        self.report({"INFO"}, f"{self.slot.title()} set to: {name}")
        return {"FINISHED"}


class PP_OT_detect_source_maps(Operator):
    bl_idname = "pp.detect_source_maps"
    bl_label = "Detect Maps in source/"
    bl_description = ("Scan the project's source/ folder and auto-fill the input slots from a "
                     "consistent export set (e.g. Gleba). Only fills EMPTY slots; never overwrites "
                     "a map you set yourself")

    @classmethod
    def poll(cls, context) -> bool:
        src = _source_dir(context)
        if src is None or not src.exists():
            cls.poll_message_set("Set Project Root (with a source/ folder) first")
            return False
        return True

    def execute(self, context):
        s = context.scene.projection_pasta
        es = context.scene.projection_pasta_erosion
        root = s.project_root_path()
        src = root / "source"
        exts = {".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff"}
        # Ignore our own decoded caches so re-detect is stable.
        names = [f.name for f in src.iterdir()
                 if f.is_file() and f.suffix.lower() in exts and "__decoded" not in f.name]
        picks = layers.classify_source_folder(names)

        filled, notes = [], []
        # Heightmap (consumed): fill if empty.
        if not (s.heightmap_filename or "").strip() and picks["heightmap"]:
            s.heightmap_filename = picks["heightmap"]
            filled.append(f"heightmap={picks['heightmap']}")

        # Rainfall (consumed): decode-on-ingest if it's a colormap map.
        if not (es.rainfall_filename or "").strip() and picks["rainfall"]:
            try:
                name = _ingest_map(root, src / picks["rainfall"], decode_colormap=True)
                es.rainfall_filename = name
                filled.append(f"rainfall={name}" + (" (decoded)" if name.endswith("__decoded.png") else ""))
            except Exception as e:
                notes.append(f"rainfall decode failed: {e}")

        # The rest are not consumed yet -- surface them as suggestions.
        for slot in ("world_map", "bathymetry", "landsea_mask", "uplift", "erodibility"):
            if picks.get(slot):
                notes.append(f"{slot}: {picks[slot]}")

        msg = "Auto-filled " + (", ".join(filled) if filled else "nothing new")
        if notes:
            msg += "  |  available: " + ", ".join(notes)
        print(f"[Project-R] Detect maps: {msg}")
        self.report({"INFO"}, msg)
        return {"FINISHED"}


def _reproject_categorical_to_section(rgb: np.ndarray, sec: dict, root: Path) -> np.ndarray:
    """Reproject an equirect categorical map (H,W,3 uint8) into a section's exact
    oblique-Hammer crop using NEAREST sampling (so no in-between class colours are
    invented), sized to the section's exported heightmap crop. Returns uint8 RGB.
    """
    proj = sec.get("projection", {}) or {}
    params = ProjectionParams(
        center_lon_deg=float(proj.get("center_lon_deg", 0.0)),
        center_lat_deg=float(proj.get("center_lat_deg", 0.0)),
        rot_deg=float(proj.get("rot_deg", 0.0)),
    )
    full = sec.get("full_canvas", {}).get("size", None)
    rect = (sec.get("crop", {}) or {}).get("rect_xywh", None)
    if not full or not rect:
        raise ValueError("section is missing full_canvas size / crop rect (recreate the section)")
    full_w, full_h = int(full[0]), int(full[1])
    x, y, w, h = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))

    data = rgb.astype(np.float32) / 255.0
    hammer = project_equirect_array_to_hammer(
        data_in=data, dst_size=(full_w, full_h), params=params,
        interp="nearest", treat_as_color=False,
    )
    crop = hammer[y:y + h, x:x + w, :3]

    # Match the section's exported heightmap-crop size so the masks register 1:1 with
    # the terrain taken into Gaea. Nearest resize keeps exact palette colours.
    target = _section_crop_size(sec, root)
    if target is not None and target != (crop.shape[1], crop.shape[0]):
        crop = imaging.resize_to(crop, target[0], target[1], interp="nearest")
    return (np.clip(crop, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _section_crop_size(sec: dict, root: Path):
    """(W,H) of the section's heightmap crop, so masks can match it. None if unknown.
    Crop paths are stored relative to the project root, so resolve against it."""
    crop_paths = (sec.get("crop", {}) or {}).get("paths_by_layer", {}) or {}
    hm = erode_ops._find_heightmap_filename(sec, "")
    target_rel = crop_paths.get(hm) if hm else None
    if not target_rel:
        target_rel = next(iter(crop_paths.values()), None)  # any crop
    if target_rel:
        try:
            from PIL import Image as PILImage
            with PILImage.open(root / target_rel) as im:
                return (int(im.size[0]), int(im.size[1]))
        except Exception:
            pass
    rect = (sec.get("crop", {}) or {}).get("rect_xywh", None)
    if rect:
        return (int(rect[2]), int(rect[3]))
    return None


def _write_class_masks(out_dir: Path, stem: str, rgb: np.ndarray, *, koppen: bool, skip_white: bool):
    """Split a categorical RGB raster and write one hard 8-bit mask per class plus a
    palette.json. Returns (n_written, split_result)."""
    res = decode.split_categorical(rgb, koppen=koppen)
    out_dir.mkdir(parents=True, exist_ok=True)
    assign = res["assignment"]
    written = 0
    from PIL import Image as PILImage
    for c in res["classes"]:
        if c["is_white_bg"] and skip_white:
            continue
        mask = np.where(assign == c["index"], np.uint8(255), np.uint8(0))
        label = f"_{c['koppen_guess']}" if koppen else ""
        fname = f"{stem}_mask_{c['index']:02d}_{c['hex']}{label}.png"
        PILImage.fromarray(mask).save(out_dir / fname)  # uint8 2D -> mode 'L'
        written += 1
    (out_dir / f"{stem}_palette.json").write_text(
        json.dumps({"n_classes": res["n_classes"], "n_unique_colors_raw": res["n_unique_colors_raw"],
                    "masks_written": written, "classes": res["classes"]}, indent=2),
        encoding="utf-8",
    )
    return written, res


class PP_OT_export_class_masks(Operator):
    bl_idname = "pp.export_class_masks"
    bl_label = "Export Class Masks"
    bl_description = ("Split a categorical map (Biome / Koppen / ...) into one black-and-white mask "
                     "per class for use as Gaea masks. GLOBAL writes the full equirect map; SECTION "
                     "reprojects it into the erosion-target section's exact crop so the masks align "
                     "with the terrain you take into Gaea")

    scope: bpy.props.EnumProperty(  # type: ignore[valid-type]
        items=[
            ("GLOBAL", "Global", "Split the whole equirectangular map"),
            ("SECTION", "Section", "Reproject into the erosion-target section's Hammer crop"),
        ],
        default="GLOBAL",
        options={"SKIP_SAVE"},
    )
    filepath: bpy.props.StringProperty(subtype="FILE_PATH", default="", options={"SKIP_SAVE"})  # type: ignore[valid-type]
    koppen: bpy.props.BoolProperty(  # type: ignore[valid-type]
        name="Auto-name Köppen classes",
        description="Label each class with the nearest standard Köppen-Geiger code (for the Koppen map)",
        default=False,
    )
    skip_white: bpy.props.BoolProperty(  # type: ignore[valid-type]
        name="Skip ocean/background",
        description="Don't write a mask for the pure-white background class",
        default=False,
    )

    @classmethod
    def poll(cls, context) -> bool:
        s = getattr(context.scene, "projection_pasta", None)
        if s is None or s.project_root_path() is None:
            cls.poll_message_set("Set Project Root first")
            return False
        return True

    def invoke(self, context, event):
        src = _source_dir(context)
        if src is not None and src.exists():
            self.filepath = str(src) + "\\"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if not self.filepath:
            self.report({"ERROR"}, "No categorical map selected")
            return {"CANCELLED"}
        s = context.scene.projection_pasta
        root = s.project_root_path()
        if root is None:
            self.report({"ERROR"}, "Project Root is not set")
            return {"CANCELLED"}
        src_path = Path(self.filepath)
        koppen = bool(self.koppen) or ("koppen" in src_path.stem.lower())

        try:
            from PIL import Image as PILImage  # required dep
            rgb = np.asarray(PILImage.open(src_path).convert("RGB"), dtype=np.uint8)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to load map: {e}")
            return {"CANCELLED"}

        if rgb.shape[0] * rgb.shape[1] and decode.detect_palette(rgb)[1] > 1500:
            self.report({"WARNING"}, f"{src_path.name} has many colours -- this may not be a "
                                     f"categorical map; exporting the dominant classes anyway.")

        # SECTION scope: reproject the categorical map into the target section's crop first.
        sec_id = None
        if self.scope == "SECTION":
            mp = s.manifest_path()
            if mp is None or not mp.exists():
                self.report({"ERROR"}, "No project manifest; create a project and a section first")
                return {"CANCELLED"}
            es = context.scene.projection_pasta_erosion
            manifest = manifest_lib.read_manifest(mp)
            sec = erode_ops._resolve_section(manifest, (es.section or "").strip())
            if sec is None:
                self.report({"ERROR"}, "No matching section (pick one in the Erosion panel, or create a section)")
                return {"CANCELLED"}
            sec_id = str(sec.get("id", ""))
            win = context.window
            win.cursor_set("WAIT")
            try:
                rgb = _reproject_categorical_to_section(rgb, sec, root)
            except Exception as e:
                self.report({"ERROR"}, f"Section reprojection failed: {e}")
                return {"CANCELLED"}
            finally:
                win.cursor_set("DEFAULT")
            out_dir = root / "sections" / sec_id / "masks" / src_path.stem
            where = f"sections/{sec_id}/masks/{src_path.stem}/"
        else:
            out_dir = root / "masks" / src_path.stem
            where = f"masks/{src_path.stem}/"

        try:
            written, res = _write_class_masks(out_dir, src_path.stem, rgb,
                                              koppen=koppen, skip_white=self.skip_white)
        except Exception as e:
            self.report({"ERROR"}, f"Failed writing masks: {e}")
            return {"CANCELLED"}

        print(f"[Project-R] Exported {written} class masks -> {out_dir}")
        self.report({"INFO"}, f"Exported {written} class masks to {where}"
                              + (" (Köppen-named)" if koppen else ""))
        return {"FINISHED"}


_CLASSES = (PP_OT_set_input_map, PP_OT_detect_source_maps, PP_OT_export_class_masks)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
