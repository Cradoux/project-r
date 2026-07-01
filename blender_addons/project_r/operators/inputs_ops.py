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


def _ingest_map(root: Path, src_path: Path, *, transform: str = "none") -> str:
    """Bring a map into ``source/`` and return the filename to record. Idempotent.

    ``transform``:
      "none"        copy unchanged.
      "colormap"    (rainfall): a true colormap RGB image is inverted via the baked
                    viridis LUT to a 16-bit single channel ``<stem>__decoded.png``.
      "invert_depth" (bathymetry): a Gleba ocean-depth map (1=shore/land, 0=deepest) is
                    inverted to the seafloor convention (1=deepest) and written 16-bit
                    single channel ``<stem>__bathy.png``. Land (1->0) is harmless: the
                    sea-floor pass only rewrites sub-sea cells.
    """
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    src_path = Path(src_path)

    if transform == "colormap":
        buf = imaging.load_image(src_path)
        if buf.channels >= 3 and not _is_effectively_gray(buf.pixels):
            scalar = decode.decode_colormap_to_scalar(buf.pixels)
            return _save_single_channel(source, f"{src_path.stem}__decoded.png", scalar)
    elif transform == "luminance":
        buf = imaging.load_image(src_path)
        if buf.channels >= 3:
            lum = decode.decode_luminance(buf.pixels)
            return _save_single_channel(source, f"{src_path.stem}__decoded.png", lum)
        # already single-channel: fall through to a plain copy (usable field as-is)
    elif transform == "invert_depth":
        buf = imaging.load_image(src_path)
        px = buf.pixels[:, :, 0] if buf.pixels.ndim == 3 else buf.pixels
        depth = (1.0 - np.clip(px.astype(np.float32), 0.0, 1.0))
        return _save_single_channel(source, f"{src_path.stem}__bathy.png", depth)

    dst = source / src_path.name
    try:
        same = dst.exists() and src_path.resolve() == dst.resolve()
    except OSError:
        same = False
    if not same:
        shutil.copy2(src_path, dst)
    return dst.name


def _save_single_channel(source: Path, name: str, arr: np.ndarray) -> str:
    out_buf = imaging.ImageBuffer(width=arr.shape[1], height=arr.shape[0], channels=1,
                                  pixels=arr[..., None].astype(np.float32))
    imaging.save_image(out_buf, source / name, "PNG", color_depth="16")
    return name


# Per-slot ingest transform.
_SLOT_TRANSFORM = {"heightmap": "none", "rainfall": "colormap", "bathymetry": "invert_depth",
                   "uplift": "luminance", "erodibility": "luminance", "landsea": "luminance"}


# Slot -> (where the filename lives). Only the roles Project-R consumes today are
# wired; the rest are surfaced by auto-detect as suggestions for a later pass.
def _set_slot_filename(context, slot: str, filename: str) -> None:
    s = context.scene.projection_pasta
    es = context.scene.projection_pasta_erosion
    if slot == "heightmap":
        s.heightmap_filename = filename
    elif slot == "rainfall":
        es.rainfall_filename = filename
    elif slot == "bathymetry":
        es.seafloor_bathy_filename = filename
        # The bathymetry input only matters with the Sea Floor pass on; enabling it on
        # load is the obvious intent (clearing the slot leaves the pass as the user set it).
        if filename:
            es.enable_seafloor = True
    elif slot == "uplift":
        es.uplift_filename = filename
        # Loading a map with influence still at 0 would be a no-op; nudge it on so it acts.
        if filename and es.uplift_influence <= 1e-6:
            es.uplift_influence = 0.5
    elif slot == "erodibility":
        es.erodibility_filename = filename
        if filename and es.erodibility_contrast <= 1.0001:
            es.erodibility_contrast = 2.0
    elif slot == "landsea":
        es.landsea_filename = filename
        # A coastline mask is only consumed by Lock Coastline; turning it on when one is
        # loaded matches the obvious intent (clearing leaves the toggle as the user set it).
        if filename:
            es.lock_coastline = True


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
            ("bathymetry", "Bathymetry", "Ocean depth for the Sea Floor pass (Gleba depth maps are inverted)"),
            ("uplift", "Uplift", "Orogeny/uplift intensity (luminance-decoded); concentrates relief in belts"),
            ("erodibility", "Erodibility", "Continuous rock-softness (luminance-decoded); softer erodes faster"),
            ("landsea", "Land/Sea", "Coastline mask for Lock Coastline (auto-oriented vs the heightmap)"),
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
            name = _ingest_map(root, Path(self.filepath),
                               transform=_SLOT_TRANSFORM.get(self.slot, "none"))
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
                name = _ingest_map(root, src / picks["rainfall"], transform="colormap")
                es.rainfall_filename = name
                filled.append(f"rainfall={name}" + (" (decoded)" if name.endswith("__decoded.png") else ""))
            except Exception as e:
                notes.append(f"rainfall decode failed: {e}")

        # Bathymetry (consumed by the Sea Floor pass): invert the Gleba depth map on ingest.
        if not (es.seafloor_bathy_filename or "").strip() and picks["bathymetry"]:
            try:
                name = _ingest_map(root, src / picks["bathymetry"], transform="invert_depth")
                es.seafloor_bathy_filename = name
                es.enable_seafloor = True
                filled.append(f"bathymetry={name} (Sea Floor on)")
            except Exception as e:
                notes.append(f"bathymetry decode failed: {e}")

        # Uplift (consumed): luminance-decode the orogeny map on ingest, nudge influence on.
        if not (es.uplift_filename or "").strip() and picks["uplift"]:
            try:
                name = _ingest_map(root, src / picks["uplift"], transform="luminance")
                _set_slot_filename(context, "uplift", name)
                filled.append(f"uplift={name} (influence {es.uplift_influence:.1f})")
            except Exception as e:
                notes.append(f"uplift decode failed: {e}")

        # Erodibility: a continuous soil-softness map (SoilDepth), luminance-decoded.
        if not (es.erodibility_filename or "").strip() and picks["erodibility"]:
            try:
                name = _ingest_map(root, src / picks["erodibility"], transform="luminance")
                _set_slot_filename(context, "erodibility", name)
                filled.append(f"erodibility={name} (contrast {es.erodibility_contrast:.1f})")
            except Exception as e:
                notes.append(f"erodibility decode failed: {e}")

        # Land/Sea mask (consumed by Lock Coastline): luminance-decode, turn the lock on.
        if not (es.landsea_filename or "").strip() and picks["landsea_mask"]:
            try:
                name = _ingest_map(root, src / picks["landsea_mask"], transform="luminance")
                _set_slot_filename(context, "landsea", name)
                filled.append(f"landsea={name} (Lock Coastline on)")
            except Exception as e:
                notes.append(f"landsea decode failed: {e}")

        for slot in ("world_map", "landsea_mask"):
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


# Above this many distinct colours a map isn't a class raster (it's a photo/continuous
# ramp that happens to match a categorical keyword) -- don't shatter it into masks.
_CATEGORICAL_MAX_COLORS = 1500


def export_section_class_masks(root: Path, sec: dict, source_paths) -> list:
    """Silently emit per-class Gaea masks for every categorical classification map
    (Köppen / biome / rock type / geologic province / ...) among a section's source
    maps. Each is reprojected into the section's exact Hammer crop with NEAREST sampling
    (palette-preserving), split into one hard 8-bit B&W mask per class, and written to
    ``sections/<id>/masks/<stem>/`` -- additional outputs that register 1:1 with the
    terrain taken into Gaea.

    Recognised by filename (``layers.is_categorical_name``); a keyword match that turns
    out to be continuous (too many colours) is skipped. Never raises: a per-map failure
    is logged and skipped so it can't fail section creation. Returns a summary list of
    ``{"map", "classes"}`` for the maps that produced masks.
    """
    sec_id = str(sec.get("id", ""))
    results: list = []
    for src_path in source_paths:
        src_path = Path(src_path)
        if not layers.is_categorical_name(src_path.name):
            continue
        try:
            from PIL import Image as PILImage  # required dep
            rgb = np.asarray(PILImage.open(src_path).convert("RGB"), dtype=np.uint8)
        except Exception as e:
            print(f"[Project-R] Class masks: skipped {src_path.name} (load failed: {e})")
            continue
        if not rgb.size or decode.detect_palette(rgb)[1] > _CATEGORICAL_MAX_COLORS:
            print(f"[Project-R] Class masks: skipped {src_path.name} (not categorical -- too many colours)")
            continue
        koppen = "koppen" in src_path.stem.lower()
        try:
            sec_rgb = _reproject_categorical_to_section(rgb, sec, root)
            out_dir = root / "sections" / sec_id / "masks" / src_path.stem
            # Skip the pure-white background (ocean / no-data) so the extra outputs stay
            # meaningful terrain-class masks.
            written, _ = _write_class_masks(out_dir, src_path.stem, sec_rgb,
                                            koppen=koppen, skip_white=True)
        except Exception as e:
            print(f"[Project-R] Class masks: failed for {src_path.name} ({e})")
            continue
        print(f"[Project-R] Class masks: {written} written for {src_path.name} "
              f"-> sections/{sec_id}/masks/{src_path.stem}/" + (" (Köppen-named)" if koppen else ""))
        results.append({"map": src_path.name, "classes": written})
    return results


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
