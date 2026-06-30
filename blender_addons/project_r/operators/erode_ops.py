from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List, Optional, Tuple

import bpy
import numpy as np

from .. import deps
from .. import erosion
from .. import imaging
from .. import layers
from .. import manifest as manifest_lib


# Sentinel for "erode whatever section was created last" (props.MOST_RECENT_ID).
_MOST_RECENT = "__MOST_RECENT__"


def _resolve_section(manifest: dict, section_value: str) -> Optional[dict]:
    sections = manifest.get("sections", []) or []
    if not sections:
        return None
    key = (section_value or "").strip()
    if not key or key == _MOST_RECENT:
        return sections[-1]  # most recently created
    low = key.lower()
    for sec in sections:
        if str(sec.get("id", "")).lower() == low or str(sec.get("name", "")).lower() == low:
            return sec
    return None


def _find_heightmap_filename(sec: dict, explicit: str) -> Optional[str]:
    """Return the crop filename to erode: the explicit heightmap if present in this section's crops,
    else the first height-like crop, else None."""
    crop_paths = (sec.get("crop", {}) or {}).get("paths_by_layer", {}) or {}
    names = list(crop_paths.keys())
    explicit = (explicit or "").strip()
    if explicit and explicit in names:
        return explicit
    if explicit:
        # Allow matching by stem (explicit may omit extension differences).
        stem = Path(explicit).stem.lower()
        for n in names:
            if Path(n).stem.lower() == stem:
                return n
    for n in names:
        if layers.is_height_name(n):
            return n
    return None


class PP_OT_erode_section(bpy.types.Operator):
    bl_idname = "pp.erode_section"
    bl_label = "Erode Section"
    bl_description = ("Carve dendritic river drainage into a section's heightmap crop "
                      "(stream-power erosion in the equal-area oblique projection), and write the "
                      "result to processed/ so Reassemble blends it back into the global map")

    # When set (e.g. by Create Section's "erode after creating"), erode this exact
    # section id instead of the dropdown selection -- no reliance on a fallback.
    section_override: bpy.props.StringProperty(default="", options={"SKIP_SAVE", "HIDDEN"})  # type: ignore[valid-type]

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        s = getattr(context.scene, "projection_pasta", None)
        mp = s.manifest_path() if s is not None else None
        if mp is None or not mp.exists():
            cls.poll_message_set("Open or create a project first")
            return False
        return True

    def execute(self, context: bpy.types.Context):
        s = context.scene.projection_pasta
        es = context.scene.projection_pasta_erosion

        deps.ensure_on_path()  # make a just-installed landlab importable without restart
        if not deps.landlab_available():
            self.report({"ERROR"}, "landlab is not installed. Click 'Install Dependencies' (no restart needed).")
            return {"CANCELLED"}

        root = s.project_root_path()
        mp = s.manifest_path()
        if root is None or mp is None or not mp.exists():
            self.report({"ERROR"}, "manifest.json not found (Open or Create a project first)")
            return {"CANCELLED"}

        manifest = manifest_lib.read_manifest(mp)
        target_value = (self.section_override or es.section or "").strip()
        sec = _resolve_section(manifest, target_value)
        if sec is None:
            self.report({"ERROR"}, "No matching section found (pick one from the Section dropdown, or create a section first)")
            return {"CANCELLED"}
        sec_id = str(sec.get("id", ""))

        hm_name = _find_heightmap_filename(sec, s.heightmap_filename)
        if not hm_name:
            self.report({"ERROR"}, "No heightmap crop found for this section. Set 'Heightmap File' or "
                                   "name a layer with 'height'/'elev'/'dem'.")
            return {"CANCELLED"}

        crop_path = root / "sections" / sec_id / "crops" / hm_name
        if not crop_path.exists():
            self.report({"ERROR"}, f"Heightmap crop not found: {crop_path}")
            return {"CANCELLED"}

        # --- Load crop heightmap -> metres (brightness * global max elevation) ---
        try:
            crop_img = imaging.load_image(crop_path)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to load heightmap crop: {e}")
            return {"CANCELLED"}

        bright = crop_img.pixels
        if bright.ndim == 3:
            bright = bright[:, :, 0]
        bright = bright.astype(np.float32)
        H0, W0 = bright.shape

        max_elev_m = float(s.max_elevation_m)
        if max_elev_m <= 0.0:
            max_elev_m = 8849.0
        height_m = bright * max_elev_m

        # --- Ground scale (single representative cell size in metres) ---
        size_info = sec.get("size_info", {}) or {}
        km_per_pixel = float(size_info.get("km_per_pixel", 0.0))
        if km_per_pixel <= 0.0:
            extent_km = size_info.get("extent_km", [0.0, 0.0])
            if extent_km and extent_km[0] > 0 and W0 > 0:
                km_per_pixel = float(extent_km[0]) / float(W0)
        if km_per_pixel <= 0.0:
            self.report({"ERROR"}, "Section is missing physical scale (size_info.km_per_pixel). Recreate the section.")
            return {"CANCELLED"}

        cell_native_m = km_per_pixel * 1000.0
        ground_w_m = cell_native_m * W0
        tile_km = ground_w_m / 1000.0

        # --- Optional downsample to keep Blender responsive on large crops ---
        max_work = int(es.max_work_px)
        if max_work > 0 and max(H0, W0) > max_work:
            scale = max_work / float(max(H0, W0))
            work_w = max(8, int(round(W0 * scale)))
            work_h = max(8, int(round(H0 * scale)))
            work_seed = layers.resample_2d(height_m, work_w, work_h)
        else:
            work_w, work_h = W0, H0
            work_seed = height_m
        cell_work_m = ground_w_m / float(work_w)

        # Target peak (metres) = the section's restored absolute elevation. Explicit override, else
        # the section's tracked elevation, else the crop's own pre-erosion peak.
        elev_info = sec.get("elevation_info", {}) or {}
        tracked_max_m = float(elev_info.get("section_max_elevation_m", 0.0))
        if float(es.target_peak_m) > 0.0:
            target_peak_m = float(es.target_peak_m)
        elif tracked_max_m > 0.0:
            target_peak_m = tracked_max_m
        else:
            target_peak_m = float(height_m.max())
        if target_peak_m <= 0.0:
            target_peak_m = max_elev_m  # all-water/flat crop: avoid divide-by-zero on encode

        print(f"[Project-R] Eroding '{sec_id}/{hm_name}': native {W0}x{H0}, work {work_w}x{work_h}, "
              f"cell {cell_work_m:.0f} m/px, tile {tile_km:.0f} km, peak target {target_peak_m:.0f} m")

        # Erosion is a blocking compute (seconds to many minutes). Until it becomes a
        # background job, at least show an honest busy state: a WAIT cursor and a
        # status-bar progress range, so the user can tell a long run from a hang.
        win = context.window
        wm = context.window_manager
        win.cursor_set("WAIT")
        wm.progress_begin(0, int(es.steps))
        try:
            # --- Run erosion (metrics computed pre-rescale, inside run_erosion) ---
            try:
                z_work, metrics = erosion.run_erosion(
                    work_seed,
                    cell_work_m,
                    tile_km,
                    work_w,
                    noise_kind=str(es.noise_kind),
                    noise_amp=float(es.noise_amp),
                    noise_seed=int(es.noise_seed),
                    climate_kind=str(es.climate_kind),
                    climate_strength=float(es.climate_strength),
                    k_sp=float(es.k_sp),
                    m_sp=float(es.m_sp),
                    n_sp=float(es.n_sp),
                    diffusivity=float(es.diffusivity),
                    uplift=float(es.uplift),
                    dt=float(es.dt),
                    steps=int(es.steps),
                    enable_overlay=bool(es.enable_overlay),
                    overlay_depth_m=float(es.overlay_depth_m),
                    overlay_w_macro_km=float(es.overlay_w_macro_km),
                    overlay_r=float(es.overlay_r),
                    target_peak_m=None,  # rescale AFTER upsample so the peak lands exactly
                    base=0.0,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({"ERROR"}, f"Erosion failed: {e}")
                return {"CANCELLED"}

            # --- Upsample back to native crop size ---
            if (work_w, work_h) != (W0, H0):
                z_native = layers.resample_2d(z_work, W0, H0)
            else:
                z_native = z_work

            # Fail loudly on numerical instability instead of silently shipping a corrupt (all-black)
            # section: an unstable diffusivity/dt/steps combo can blow the LEM up to NaN/Inf, which
            # would propagate through rescale_peak (z.max() -> nan) into the saved heightmap.
            if not np.isfinite(z_native).all():
                self.report({"ERROR"}, "Erosion became numerically unstable (NaN/Inf). Reduce "
                                       "Diffusivity, Timestep (dt), or Steps and retry.")
                return {"CANCELLED"}

            z_native = erosion.rescale_peak(z_native, target_peak_m, base=0.0)
        finally:
            wm.progress_end()
            win.cursor_set("DEFAULT")

        # --- Encode SECTION-normalized (peak -> 1.0), matching the Gaea/Wilbur export convention the
        # reassembly pipeline expects. With Normalize Heights ON (default), Reassemble multiplies each
        # processed heightmap by section_elev/global_max, restoring the correct absolute elevation and
        # keeping multi-section relative heights right. We also write the section's elevation back into
        # the manifest below so that scaling has the right value. (A global-absolute encoding would be
        # double-scaled here and silently flatten every non-tallest section.)
        denom = max(target_peak_m, 1e-6)
        out_bright = np.clip(z_native / denom, 0.0, 1.0).astype(np.float32)
        out_buf = imaging.ImageBuffer(width=W0, height=H0, channels=1, pixels=out_bright[..., None])

        # Write to processed/ (drop-in for Reassemble) + an inspection copy in the section folder.
        processed_dir = root / "processed" / sec_id
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / hm_name
        imaging.save_image(out_buf, processed_path, "PNG", color_depth="16")

        inspect_path = root / "sections" / sec_id / f"{Path(hm_name).stem}__eroded.png"
        imaging.save_image(out_buf, inspect_path, "PNG", color_depth="16")

        # --- Quality readout ---
        theta = metrics.get("theta")
        r2 = metrics.get("r2")
        bsl = metrics.get("band_slope")
        router = metrics.get("router", "")
        secs = (metrics.get("lem_secs") or 0.0) + (metrics.get("overlay_secs") or 0.0)

        es.last_theta = float(theta) if theta is not None and np.isfinite(theta) else 0.0
        es.last_r2 = float(r2) if r2 is not None and np.isfinite(r2) else 0.0
        es.last_band_slope = float(bsl) if bsl is not None and np.isfinite(bsl) else 0.0
        es.last_router = str(router)
        es.last_secs = float(secs)

        theta_ok = (theta is not None and np.isfinite(theta) and 0.4 <= theta <= 0.55
                    and r2 is not None and r2 > 0.9)
        verdict = "drainage OK" if theta_ok else "check hillshade"
        report = (f"theta={es.last_theta:.3f} R2={es.last_r2:.3f} band_slope={es.last_band_slope:.3f} "
                  f"({verdict}) | {router} {secs:.1f}s")
        es.last_report = report

        # --- Update manifest so Reassemble restores the section's elevation correctly ---
        # Because the eroded heightmap is now section-normalized (peak 1.0), Reassemble's Normalize
        # Heights step needs this section's elevation_info to scale it back to absolute metres. Write
        # it (and the global heightmap pointer) so eroded sections compose with each other and with
        # Gaea-processed sections.
        sec_min_m = max(float(z_native.min()), 0.0)
        try:
            sec["elevation_info"] = {
                "heightmap_file": hm_name,
                "global_max_elevation_m": max_elev_m,
                "section_max_brightness": round(target_peak_m / max_elev_m, 4),
                "section_max_elevation_m": round(target_peak_m, 2),
                "section_min_brightness": round(sec_min_m / max_elev_m, 4),
                "section_min_elevation_m": round(sec_min_m, 2),
            }
            manifest.setdefault("global", {})["heightmap_filename"] = hm_name
            manifest["global"]["max_elevation_m"] = max_elev_m

            def _json_safe(v):
                if isinstance(v, (int, float)):
                    return None if not np.isfinite(v) else float(v)
                return v

            sec["erosion"] = {
                "heightmap": hm_name,
                "processed_path": str((Path("processed") / sec_id / hm_name)).replace("\\", "/"),
                "ran_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "work_px": [work_w, work_h],
                "cell_m": round(cell_work_m, 2),
                "target_peak_m": round(target_peak_m, 1),
                "params": {
                    "noise_kind": str(es.noise_kind), "noise_amp": float(es.noise_amp),
                    "climate_kind": str(es.climate_kind), "climate_strength": float(es.climate_strength),
                    "k_sp": float(es.k_sp), "m_sp": float(es.m_sp), "n_sp": float(es.n_sp),
                    "diffusivity": float(es.diffusivity), "uplift": float(es.uplift),
                    "dt": float(es.dt), "steps": int(es.steps),
                    "overlay": bool(es.enable_overlay), "overlay_depth_m": float(es.overlay_depth_m),
                },
                "metrics": {k: _json_safe(v) for k, v in metrics.items()},
            }
            manifest_lib.write_manifest(mp, manifest)
        except Exception as e:
            print(f"[Project-R] Warning: could not record erosion provenance: {e}")

        # --- Load the result for preview (non-color) ---
        try:
            img = bpy.data.images.load(str(processed_path), check_existing=True)
            try:
                img.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
            img.reload()
        except Exception:
            pass

        sec_name = str(sec.get("name", sec_id))
        self.report({"INFO"}, f"Eroded '{sec_name}' ({sec_id}/{hm_name}): {report}")
        return {"FINISHED"}


_CLASSES = (PP_OT_erode_section,)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
