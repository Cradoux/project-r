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


def _find_rainfall_filename(sec: dict, explicit: str, heightmap_name: Optional[str] = None) -> Optional[str]:
    """Crop filename to use as the rainfall map: the explicit one if present in this
    section's crops, else the first rain/precip-like crop (never the heightmap), else
    None. An explicit name that isn't present returns None (the caller warns and falls
    back to uniform) rather than silently substituting a different layer."""
    crop_paths = (sec.get("crop", {}) or {}).get("paths_by_layer", {}) or {}
    names = list(crop_paths.keys())
    explicit = (explicit or "").strip()
    if explicit:
        if explicit in names:
            return explicit
        stem = Path(explicit).stem.lower()
        for n in names:
            if Path(n).stem.lower() == stem:
                return n
        return None
    for n in names:
        if heightmap_name and n == heightmap_name:
            continue  # never auto-reuse the heightmap as the rainfall field
        if layers.is_rainfall_name(n):
            return n
    return None


def _find_layer_crop(sec: dict, explicit: str) -> Optional[str]:
    """The crop filename matching an explicit layer name (exact, then by stem), or None.
    Used for the optional spatial-driver crops (uplift / erodibility)."""
    explicit = (explicit or "").strip()
    if not explicit:
        return None
    crop_paths = (sec.get("crop", {}) or {}).get("paths_by_layer", {}) or {}
    names = list(crop_paths.keys())
    if explicit in names:
        return explicit
    stem = Path(explicit).stem.lower()
    for n in names:
        if Path(n).stem.lower() == stem:
            return n
    return None


def _load_crop_field(crop_path: Path, work_w: int, work_h: int) -> Optional[np.ndarray]:
    """Load a single-channel crop and resample to (work_h, work_w) as float32. None on failure."""
    try:
        img = imaging.load_image(crop_path)
        px = img.pixels
        if px.ndim == 3:
            px = px[:, :, 0]
        return layers.resample_2d(px.astype(np.float32), work_w, work_h)
    except Exception as ex:
        print(f"[Project-R] Warning: failed to load spatial-driver crop '{crop_path.name}': {ex}")
        return None


def _find_landsea_filename(sec: dict, explicit: str) -> Optional[str]:
    """Crop filename to use as the land/sea (coastline) mask: the explicit one if present
    in this section's crops, else the first land/sea-like crop, else None. An explicit name
    that isn't present returns None (the caller falls back to the heightmap's sea level)."""
    explicit = (explicit or "").strip()
    if explicit:
        return _find_layer_crop(sec, explicit)
    crop_paths = (sec.get("crop", {}) or {}).get("paths_by_layer", {}) or {}
    for n in crop_paths:
        if layers.is_landsea_name(n):
            return n
    return None


def _full_canvas_window(root: Path, rel_path: str, win, out_w: int, out_h: int) -> Optional[np.ndarray]:
    """Read the pixel window ``[y0:y1, x0:x1]`` from a section's retained full Hammer canvas
    (``rel_path`` relative to ``root``) and resample to (out_h, out_w) as float32. This is how
    the seam halo pulls REAL neighbour terrain -- already in the section's projection -- from
    the canvas the crop was cut from. Returns None on any failure (caller falls back)."""
    try:
        x0, y0, x1, y1 = (int(v) for v in win)
        img = imaging.load_image(root / rel_path)
        px = img.pixels
        if px.ndim == 3:
            px = px[:, :, 0]
        sub = px[y0:y1, x0:x1].astype(np.float32)
        if sub.size == 0 or sub.shape[0] < 1 or sub.shape[1] < 1:
            return None
        return layers.resample_2d(sub, out_w, out_h)
    except Exception as ex:
        print(f"[Project-R] Warning: failed to read full-canvas window '{rel_path}': {ex}")
        return None


def _match_halo(core_field: Optional[np.ndarray], root: Path, sec: dict, stem: str,
                halo: Optional[dict]) -> Optional[np.ndarray]:
    """Resize a core (work-res) per-node field to the enlarged halo shape: prefer the layer's
    own full-canvas window (real neighbour data), else edge-pad the core into the halo ring.
    Pass-through when there is no halo or no field."""
    if core_field is None or halo is None:
        return core_field
    fc = (sec.get("full_canvas", {}) or {}).get("path_by_layer", {}) or {}
    rel = fc.get(stem)
    if rel and (root / rel).exists():
        enl = _full_canvas_window(root, rel, halo["win"], halo["Wk"], halo["Hk"])
        if enl is not None:
            return enl
    ht_w, hb_w, hl_w, hr_w = halo["pad"]
    return np.pad(core_field, ((ht_w, hb_w), (hl_w, hr_w)), mode="edge")


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
        ocean_floor_depth_m = float(s.ocean_floor_depth_m)  # world floor for the Gaea sea datum

        # --- Ground scale from the section's PHYSICAL extent, not pixel count, so the
        # cell size stays correct at any output/crop resolution (the crop may have been
        # re-exported or resampled to a different size). ---
        size_info = sec.get("size_info", {}) or {}
        extent_km = size_info.get("extent_km", [0.0, 0.0]) or [0.0, 0.0]
        ground_w_km = float(extent_km[0]) if (extent_km and extent_km[0] > 0) else 0.0
        if ground_w_km <= 0.0:
            kpp = float(size_info.get("km_per_pixel", 0.0))  # legacy fallback
            ground_w_km = kpp * W0 if kpp > 0 else 0.0
        if ground_w_km <= 0.0:
            self.report({"ERROR"}, "Section is missing physical scale (size_info.extent_km). Recreate the section.")
            return {"CANCELLED"}

        ground_w_m = ground_w_km * 1000.0
        tile_km = ground_w_km

        # --- Output/detail resolution: erode AT the chosen longest-edge size and
        # write the processed map at it (Reassemble resamples to the section rect as
        # needed). 'Auto' picks a balanced size from the crop's native resolution. ---
        out_res = erosion.resolve_resolution(str(s.output_resolution), max(W0, H0))
        if max(W0, H0) != out_res:
            scale = out_res / float(max(W0, H0))
            work_w = max(8, int(round(W0 * scale)))
            work_h = max(8, int(round(H0 * scale)))
            work_seed = layers.resample_2d(height_m, work_w, work_h)
        else:
            work_w, work_h = W0, H0
            work_seed = height_m
        cell_work_m = ground_w_m / float(work_w)

        # --- Seam Halo: replace the work seed with an ENLARGED window read from the section's
        # retained full Hammer canvas (same projection, real neighbour terrain), so the LEM's
        # no-flow boundary sits OUT in a ring we discard -- rivers/relief then stay continuous
        # across section seams. Everything downstream runs on the enlarged array; the core is
        # sliced back out just before encoding. Falls back to no halo for sections created
        # before full-canvas retention, or if the canvas is missing. ---
        halo = None
        halo_px = int(es.seam_halo_px)
        if halo_px > 0:
            rect_xywh = (sec.get("crop", {}) or {}).get("rect_xywh")
            fc = sec.get("full_canvas", {}) or {}
            fc_size = fc.get("size") or [0, 0]
            hm_full_rel = (fc.get("path_by_layer", {}) or {}).get(Path(hm_name).stem)
            fcw, fch = (int(fc_size[0]), int(fc_size[1])) if len(fc_size) >= 2 else (0, 0)
            if rect_xywh and hm_full_rel and fcw > 0 and fch > 0 and (root / hm_full_rel).exists():
                rx, ry, rw, rh = (int(v) for v in rect_xywh)
                sx, sy = work_w / float(rw), work_h / float(rh)  # work px per canvas px, per axis
                hcx, hcy = int(round(halo_px / max(sx, 1e-9))), int(round(halo_px / max(sy, 1e-9)))
                x0, y0 = max(0, rx - hcx), max(0, ry - hcy)
                x1, y1 = min(fcw, rx + rw + hcx), min(fch, ry + rh + hcy)
                hl_w, ht_w = int(round((rx - x0) * sx)), int(round((ry - y0) * sy))
                hr_w, hb_w = int(round((x1 - (rx + rw)) * sx)), int(round((y1 - (ry + rh)) * sy))
                Wk, Hk = hl_w + work_w + hr_w, ht_w + work_h + hb_w
                win = (x0, y0, x1, y1)
                enl = _full_canvas_window(root, hm_full_rel, win, Wk, Hk)
                if enl is not None and (hl_w + ht_w + hr_w + hb_w) > 0:
                    work_seed = (enl * max_elev_m).astype(np.float32)
                    halo = dict(cx0=hl_w, cy0=ht_w, Wk=Wk, Hk=Hk, win=win,
                                pad=(ht_w, hb_w, hl_w, hr_w))
                    print(f"[Project-R] Seam halo +{halo_px}px: core {work_w}x{work_h} inside "
                          f"{Wk}x{Hk} (canvas window {x1 - x0}x{y1 - y0})")
                elif enl is None:
                    self.report({"WARNING"}, "Seam Halo: could not read the full Hammer canvas; "
                                             "eroding without a halo.")
                # else: canvas present but the crop already spans it (no room for a ring) -> no-op.
            else:
                self.report({"WARNING"}, "Seam Halo needs a section created with full-canvas "
                                         "retention (re-create the section); eroding without a halo.")

        # --- Lock Coastline: snapshot the authoritative shore from the PRISTINE input,
        # before any pre-pass reshapes work_seed. Source priority: a dedicated Land/Sea mask
        # (auto-oriented vs the elevation so either tone-convention works), else the input
        # heightmap's sea level. Because every section derives this from the SAME global
        # source, adjacent sections agree along their shared edge -> coastlines tile. The
        # final compose re-pins to this mask so erosion never moves the shore. ---
        locked_sea = None
        orig_seed = None
        if bool(es.lock_coastline):
            orig_seed = np.array(work_seed, copy=True)
            elev_sea = orig_seed <= float(es.sea_level_m)
            ls_name = _find_landsea_filename(sec, es.landsea_filename)
            ls_field = None
            if ls_name:
                ls_field = _load_crop_field(root / "sections" / sec_id / "crops" / ls_name, work_w, work_h)
                ls_field = _match_halo(ls_field, root, sec, Path(ls_name).stem, halo)
            if ls_field is not None:
                sea_if_low = ls_field < 0.5
                # Auto-orient: pick the polarity that agrees more with the elevation's sea.
                locked_sea = sea_if_low if float((sea_if_low == elev_sea).mean()) >= 0.5 else ~sea_if_low
            else:
                if (es.landsea_filename or "").strip() and ls_name is None:
                    self.report({"WARNING"}, f"Land/Sea mask '{es.landsea_filename}' not found in this "
                                             f"section's crops; locking to the heightmap's sea level.")
                locked_sea = elev_sea

        # --- Optional glacial (fjord) pre-pass: carve U-troughs / over-deepened basins
        # FIRST, before coastal and the LEM. This is the EARLIEST structural pre-pass. Like
        # coastal, it reshapes work_seed ITSELF: the deepest troughs drop BELOW sea level, so
        # the sea_mask re-derived from work_seed below treats them as ocean and the ocean
        # restore KEEPS the flooded fjord instead of undoing it. Applied once here (not via
        # run_erosion) so the coastal pass and the strength-blend baseline both see the fjords.
        glacial_info = None
        glacial_ice = None
        if bool(es.enable_glacial):
            g_sea = work_seed <= float(es.sea_level_m)
            g_has_sea = bool(g_sea.any()) and not bool(g_sea.all())
            g_land = ~g_sea
            relief_hi = (float(np.percentile(work_seed[g_land], 98)) if g_land.any()
                         else float(work_seed.max()))
            g_ela = float(es.sea_level_m) + float(es.glacial_ela_frac) * (relief_hi - float(es.sea_level_m))
            work_seed, glacial_ice, glacial_info = erosion.glacial_erode(
                work_seed, cell_work_m, ela_m=g_ela,
                k_g=float(es.glacial_k_g), quarry_mult=float(es.glacial_quarry_mult),
                diffuse=float(es.glacial_diffuse), steps=int(es.glacial_steps),
                sea_mask=g_sea if g_has_sea else None, sea_level=float(es.sea_level_m),
            )
            print(f"[Project-R] Glacial pre-pass (ELA {g_ela:.0f} m, K_g {es.glacial_k_g:.2e}, "
                  f"steps {es.glacial_steps}): {glacial_info}")

        # --- Optional coastal (wave) pre-pass: rework the shoreline BEFORE the LEM ---
        # It reshapes work_seed ITSELF so the downstream sea_mask, the strength blend, and
        # the ocean restore below all operate on the reworked coast -- otherwise the blend
        # ("restore the original ocean") would silently undo every coastline change. We
        # apply it once here (not inside run_erosion) so the blend baseline stays consistent.
        coastal_info = None
        if bool(es.enable_coastal) and locked_sea is not None:
            self.report({"INFO"}, "Lock Coastline is on: skipping the coastal wave pass (it would "
                                  "move the shore).")
            print("[Project-R] Coastal pre-pass skipped (Lock Coastline on).")
        elif bool(es.enable_coastal):
            # Rate/steps/reach/fetch follow the Scale x Intensity preset (auto-sized to the
            # section) unless Scale is Custom; swell/talus/beach style stay user-controlled.
            if str(es.lem_scale) == "CUSTOM":
                c_rate, c_steps = float(es.coastal_rate_m), int(es.coastal_steps)
                c_notch, c_fetch = float(es.coastal_notch_m), float(es.coastal_max_fetch_km)
            else:
                cp = erosion.coastal_preset(tile_km, str(es.lem_scale), str(es.lem_intensity))
                c_rate, c_steps = cp["rate_m"], cp["steps"]
                c_notch, c_fetch = cp["notch_m"], cp["max_fetch_km"]
            work_seed, coastal_info = erosion.coastal_erode(
                work_seed, cell_work_m, sea_level=float(es.sea_level_m),
                steps=c_steps, rate_m=c_rate, notch_m=c_notch, max_fetch_km=c_fetch,
                swell_deg=float(es.coastal_swell_deg), swell_focus=float(es.coastal_swell_focus),
                talus_deg=float(es.coastal_talus_deg), deposition=bool(es.coastal_deposition),
            )
            mode = "custom" if str(es.lem_scale) == "CUSTOM" else "preset"
            print(f"[Project-R] Coastal pre-pass ({mode}, rate {c_rate:.1f} m, steps {c_steps}, "
                  f"fetch {c_fetch:.0f} km): {coastal_info}")

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

        # --- Optional rainfall map -> per-node runoff field at work resolution ---
        # An explicit name (or an auto-detected rain/precip crop) drives where incision
        # concentrates; missing/empty falls back to uniform rainfall.
        rainfall_work = None
        rain_name = _find_rainfall_filename(sec, es.rainfall_filename, heightmap_name=hm_name)
        if (es.rainfall_filename or "").strip() and rain_name is None:
            self.report({"WARNING"}, f"Rainfall map '{es.rainfall_filename}' not found in this "
                                     f"section's crops; using uniform rainfall.")
        if rain_name:
            rain_path = root / "sections" / sec_id / "crops" / rain_name
            try:
                rain_img = imaging.load_image(rain_path)
                rain_px = rain_img.pixels
                if rain_px.ndim == 3:
                    rain_px = rain_px[:, :, 0]
                rainfall_work = layers.resample_2d(rain_px.astype(np.float32), work_w, work_h)
            except Exception as ex:
                print(f"[Project-R] Warning: failed to load rainfall map '{rain_name}': {ex}")
                rainfall_work = None

        # --- Optional spatial uplift / erodibility crops -> normalized fields at work res ---
        # Like rainfall: a layer dropped into source/ is cropped per section, and we look it up
        # by the filename stored on the slot. Missing/empty falls back to uniform.
        uplift_work = None
        if (es.uplift_filename or "").strip():
            up_name = _find_layer_crop(sec, es.uplift_filename)
            if up_name:
                uplift_work = _load_crop_field(root / "sections" / sec_id / "crops" / up_name, work_w, work_h)
            else:
                self.report({"WARNING"}, f"Uplift map '{es.uplift_filename}' not found in this "
                                         f"section's crops; using uniform uplift.")
        erod_work = None
        if (es.erodibility_filename or "").strip():
            er_name = _find_layer_crop(sec, es.erodibility_filename)
            if er_name:
                erod_work = _load_crop_field(root / "sections" / sec_id / "crops" / er_name, work_w, work_h)
            else:
                self.report({"WARNING"}, f"Erodibility map '{es.erodibility_filename}' not found in this "
                                         f"section's crops; using uniform erodibility.")

        # Seam halo: enlarge the per-node fields the LEM consumes to the halo shape (real
        # neighbour data from each layer's full canvas, else edge-padded), so run_erosion sees a
        # grid matching the enlarged seed. Bathymetry stays core -- its pass runs after slicing.
        if halo is not None:
            if rainfall_work is not None and rain_name:
                rainfall_work = _match_halo(rainfall_work, root, sec, Path(rain_name).stem, halo)
            if uplift_work is not None:
                uplift_work = _match_halo(uplift_work, root, sec, Path(es.uplift_filename).stem, halo)
            if erod_work is not None:
                erod_work = _match_halo(erod_work, root, sec, Path(es.erodibility_filename).stem, halo)

        # --- Optional direct bathymetry map (for the sea-floor pass) -> [0..1] depth at work res ---
        bathy_work = None
        bathy_name = (es.seafloor_bathy_filename or "").strip()
        if bool(es.enable_seafloor) and bathy_name:
            bathy_path = root / "sections" / sec_id / "crops" / bathy_name
            if bathy_path.exists():
                try:
                    bathy_img = imaging.load_image(bathy_path)
                    bathy_px = bathy_img.pixels
                    if bathy_px.ndim == 3:
                        bathy_px = bathy_px[:, :, 0]
                    bathy_work = layers.resample_2d(bathy_px.astype(np.float32), work_w, work_h)
                except Exception as ex:
                    print(f"[Project-R] Warning: failed to load bathymetry map '{bathy_name}': {ex}")
                    bathy_work = None
            else:
                self.report({"WARNING"}, f"Bathymetry map '{bathy_name}' not found in this section's "
                                         f"crops; using the procedural sea floor.")

        # Resolve LEM physics: a (scale x intensity) preset sized to the section, or the
        # manual sliders when Scale is Custom. Seed-noise stays user-controlled either
        # way; overlay params remain the Channel Overlay ones.
        if str(es.lem_scale) == "CUSTOM":
            lem_kw = dict(
                k_sp=float(es.k_sp), m_sp=float(es.m_sp), n_sp=float(es.n_sp),
                diffusivity=float(es.diffusivity), uplift=float(es.uplift),
                dt=float(es.dt), steps=int(es.steps),
            )
            scale_label = "custom"
        else:
            p = erosion.lem_preset(tile_km, str(es.lem_scale), str(es.lem_intensity))
            lem_kw = dict(
                k_sp=p["k_sp"], m_sp=p["m_sp"], n_sp=p["n_sp"],
                diffusivity=p["diffusivity"], uplift=p["uplift"],
                dt=p["dt"], steps=p["steps"],
            )
            scale_label = f"{p['scale_band'].lower()}/{str(es.lem_intensity).lower()}"

        sea_level_m = float(es.sea_level_m)
        strength = float(es.erosion_strength)

        print(f"[Project-R] Eroding '{sec_id}/{hm_name}': native {W0}x{H0}, work {work_w}x{work_h}, "
              f"cell {cell_work_m:.0f} m/px, tile {tile_km:.0f} km, peak {target_peak_m:.0f} m, "
              f"preset {scale_label}, steps {lem_kw['steps']}, strength {strength:.2f}, sea {sea_level_m:.0f} m, "
              f"rain {rain_name or 'uniform'}, "
              f"uplift-map {(es.uplift_filename if uplift_work is not None else 'uniform')}, "
              f"erod-map {(es.erodibility_filename if erod_work is not None else 'uniform')}, "
              f"coastline {'locked' if locked_sea is not None else 'emergent'}, "
              f"shore-taper {float(es.shore_taper_m):.0f} m, "
              f"seam-halo {('+' + str(halo_px) + 'px' if halo is not None else 'off')}, "
              f"deposition {('SPACE v_s=' + format(float(es.depo_v_s), '.2g') if es.enable_deposition else 'off')}")

        # Erosion is a blocking compute (seconds to many minutes). Until it becomes a
        # background job, at least show an honest busy state: a WAIT cursor and a
        # status-bar progress range, so the user can tell a long run from a hang.
        win = context.window
        wm = context.window_manager
        win.cursor_set("WAIT")
        wm.progress_begin(0, int(lem_kw["steps"]))
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
                    rainfall=rainfall_work,
                    erodibility_norm=erod_work,
                    erodibility_contrast=float(es.erodibility_contrast),
                    uplift_norm=uplift_work,
                    uplift_influence=float(es.uplift_influence),
                    enable_overlay=bool(es.enable_overlay),
                    overlay_depth_m=float(es.overlay_depth_m),
                    overlay_w_macro_km=float(es.overlay_w_macro_km),
                    overlay_r=float(es.overlay_r),
                    target_peak_m=None,  # rescale AFTER the blend so the peak lands exactly
                    base=0.0,
                    sea_level_m=sea_level_m,
                    enable_deposition=bool(es.enable_deposition),
                    depo_v_s=float(es.depo_v_s),
                    **lem_kw,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({"ERROR"}, f"Erosion failed: {e}")
                return {"CANCELLED"}

            # Fail loudly on numerical instability instead of silently shipping a corrupt (all-black)
            # section: an unstable diffusivity/dt/steps combo can blow the LEM up to NaN/Inf, which
            # would propagate through rescale_peak (z.max() -> nan) into the saved heightmap.
            if not np.isfinite(z_work).all():
                self.report({"ERROR"}, "Erosion became numerically unstable (NaN/Inf). Reduce "
                                       "Intensity (or Diffusivity/Timestep/Steps in Custom) and retry.")
                return {"CANCELLED"}

            # Blend the eroded LAND back toward the original by Erosion Strength (the
            # over-erosion dial) and restore the OCEAN exactly, at the output resolution,
            # so the coastline/landmass shape is kept and the sea stays flat/black.
            # work_seed is the original surface resampled to the output grid.
            #
            # Sea mask + baseline: Lock Coastline forces the PRISTINE input shore (so erosion
            # never moves it and neighbouring sections tile) and restores the original flat
            # ocean; otherwise the post-pre-pass seed defines the shore (preserves fjords etc).
            if locked_sea is not None:
                sea_mask_w = locked_sea
                sea_baseline = orig_seed
            else:
                sea_mask_w = work_seed <= sea_level_m
                sea_baseline = work_seed
            # Shore Taper: fade Erosion Strength to zero within `shore_taper_m` metres above
            # sea level, so the coast is carved progressively less approaching the water --
            # softens the serrated 'teeth' a hard land/sea cutoff leaves. 0 = hard cutoff.
            taper_m = float(es.shore_taper_m)
            if taper_m > 0.0:
                w = np.clip((work_seed - sea_level_m) / taper_m, 0.0, 1.0) * strength
            else:
                w = strength
            blended = work_seed + w * (z_work - work_seed)
            z_out = np.where(sea_mask_w, sea_baseline, blended).astype(np.float32)
            land = ~sea_mask_w
            z_out[land] = np.maximum(z_out[land], sea_level_m)

            # Seam halo: discard the neighbour ring now -- slice the core (exactly work_w x
            # work_h, the size Reassemble expects) out of every enlarged array so all downstream
            # encoding, ocean fill and ice export see the section alone, not its context.
            if halo is not None:
                core = (slice(halo["cy0"], halo["cy0"] + work_h),
                        slice(halo["cx0"], halo["cx0"] + work_w))
                z_out = z_out[core]
                sea_mask_w = sea_mask_w[core]
                if glacial_ice is not None:
                    glacial_ice = {k: (v[core] if isinstance(v, np.ndarray)
                                       and v.shape == (halo["Hk"], halo["Wk"]) else v)
                                   for k, v in glacial_ice.items()}

            z_ocean_real = np.where(sea_mask_w, z_out, 0.0)  # real-metre ocean (incl. fjords) for the Gaea floor
            pre_rescale_max = float(z_out.max())  # land peak BEFORE rescale -> sets brightness-per-metre
            z_out = erosion.rescale_peak(z_out, target_peak_m, base=0.0)
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
        out_bright = np.clip(z_out / denom, 0.0, 1.0).astype(np.float32)
        out_buf = imaging.ImageBuffer(width=work_w, height=work_h, channels=1, pixels=out_bright[..., None])

        # Write to processed/ (drop-in for Reassemble) + an inspection copy in the section folder.
        processed_dir = root / "processed" / sec_id
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / hm_name
        imaging.save_image(out_buf, processed_path, "PNG", color_depth="16")

        inspect_path = root / "sections" / sec_id / f"{Path(hm_name).stem}__eroded.png"
        imaging.save_image(out_buf, inspect_path, "PNG", color_depth="16")

        # --- Glacial ice as a SEPARATE overlay layer (metres of ice), encoded at the SAME
        # brightness-per-metre as the heightmap above (thickness / pre-rescale land peak), so it
        # composites 1:1 on top of the bedrock in Gaea2 -- add the two maps to recover the ice
        # surface. Only written when a glacier actually formed. ---
        ice_path = None
        if glacial_ice is not None:
            ice_thick = np.asarray(glacial_ice.get("thickness"))
            if ice_thick.size and float(ice_thick.max()) > 0.0:
                ice_bright = np.clip(ice_thick / max(pre_rescale_max, 1e-6), 0.0, 1.0).astype(np.float32)
                ice_buf = imaging.ImageBuffer(width=work_w, height=work_h, channels=1,
                                              pixels=ice_bright[..., None])
                ice_path = root / "sections" / sec_id / f"{Path(hm_name).stem}__ice.png"
                imaging.save_image(ice_buf, ice_path, "PNG", color_depth="16")
                print(f"[Project-R] Saved glacial ice layer (max {float(ice_thick.max()):.0f} m): {ice_path}")

        # --- Sea floor + Gaea export: fill the ocean with a realistic shelf/slope/abyssal floor
        # (keeping the glacial fjords) and encode the WHOLE surface against the WORLD elevation
        # range [sea - ocean_floor, max_elev]. That single range is shared by every section, so sea
        # level lands at the SAME brightness everywhere (no colour seams) and Gaea's vertical scale is
        # one constant. The processed/ heightmap above is untouched (land-normalized, sea=0), so
        # in-Blender Reassembly is unaffected. ---
        gaea_path = None
        sf_info = None
        if bool(es.enable_seafloor):
            z_gaea_base = np.where(sea_mask_w, z_ocean_real, z_out)  # land at target metres, ocean real (fjords)
            z_gaea, sf_info = erosion.seafloor_bathymetry(
                z_gaea_base, cell_work_m, sea_level=sea_level_m,
                shelf_depth_m=float(es.seafloor_shelf_depth_m),
                shelf_width_km=float(es.seafloor_shelf_width_km),
                shelf_relief_mod=float(es.seafloor_shelf_relief_mod),
                slope_width_km=float(es.seafloor_slope_width_km),
                floor_depth_m=ocean_floor_depth_m,
                input_depth=bathy_work, input_weight=float(es.seafloor_input_weight),
            )
            g_min = sea_level_m - ocean_floor_depth_m
            g_max = max_elev_m
            span = max(g_max - g_min, 1e-6)
            gaea_bright = np.clip((z_gaea - g_min) / span, 0.0, 1.0).astype(np.float32)
            gaea_buf = imaging.ImageBuffer(width=work_w, height=work_h, channels=1,
                                           pixels=gaea_bright[..., None])
            gaea_path = root / "sections" / sec_id / f"{Path(hm_name).stem}__gaea.png"
            imaging.save_image(gaea_buf, gaea_path, "PNG", color_depth="16")

            es.last_gaea_sea = float((sea_level_m - g_min) / span)
            es.last_gaea_height_m = float(min(span, 10000.0))
            es.last_gaea_width_scale = float(min(span, 10000.0) / span)
            print(f"[Project-R] Gaea export: datum [{g_min:.0f},{g_max:.0f}] m (span {span:.0f} m) | "
                  f"set Gaea sea level = {es.last_gaea_sea:.4f}, Height = {es.last_gaea_height_m:.0f} m, "
                  f"terrain width x{es.last_gaea_width_scale:.3f} | floor {sf_info}")

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
        sec_min_m = max(float(z_out.min()), 0.0)
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
                    "scale": str(es.lem_scale), "intensity": str(es.lem_intensity),
                    "strength": strength, "sea_level_m": sea_level_m,
                    "noise_kind": str(es.noise_kind), "noise_amp": float(es.noise_amp),
                    "rainfall_map": rain_name or "",
                    "uplift_map": (es.uplift_filename if uplift_work is not None else ""),
                    "uplift_influence": float(es.uplift_influence),
                    "erodibility_map": (es.erodibility_filename if erod_work is not None else ""),
                    "erodibility_contrast": float(es.erodibility_contrast),
                    "k_sp": float(lem_kw["k_sp"]), "m_sp": float(lem_kw["m_sp"]), "n_sp": float(lem_kw["n_sp"]),
                    "diffusivity": float(lem_kw["diffusivity"]), "uplift": float(lem_kw["uplift"]),
                    "dt": float(lem_kw["dt"]), "steps": int(lem_kw["steps"]),
                    "overlay": bool(es.enable_overlay), "overlay_depth_m": float(es.overlay_depth_m),
                    "coastal": bool(es.enable_coastal),
                    "coastal_rate_m": float(es.coastal_rate_m), "coastal_steps": int(es.coastal_steps),
                    "coastal_swell_focus": float(es.coastal_swell_focus),
                    "coastal_talus_deg": float(es.coastal_talus_deg),
                    "glacial": bool(es.enable_glacial),
                    "glacial_ela_frac": float(es.glacial_ela_frac),
                    "glacial_k_g": float(es.glacial_k_g),
                    "glacial_quarry_mult": float(es.glacial_quarry_mult),
                    "glacial_steps": int(es.glacial_steps),
                    "seafloor": bool(es.enable_seafloor),
                    "seafloor_shelf_depth_m": float(es.seafloor_shelf_depth_m),
                    "seafloor_shelf_width_km": float(es.seafloor_shelf_width_km),
                    "seafloor_shelf_relief_mod": float(es.seafloor_shelf_relief_mod),
                    "seafloor_slope_width_km": float(es.seafloor_slope_width_km),
                    "seafloor_bathy_map": bathy_name if bool(es.enable_seafloor) else "",
                    "ocean_floor_depth_m": ocean_floor_depth_m,
                },
                "metrics": {k: _json_safe(v) for k, v in metrics.items()},
                "coastal_metrics": ({k: _json_safe(v) for k, v in coastal_info.items()}
                                    if coastal_info else None),
                "glacial_metrics": ({k: _json_safe(v) for k, v in glacial_info.items()}
                                    if glacial_info else None),
                "glacial_ice_path": (str((Path("sections") / sec_id /
                                          f"{Path(hm_name).stem}__ice.png")).replace("\\", "/")
                                     if ice_path is not None else None),
                "gaea_export": ({
                    "path": str((Path("sections") / sec_id /
                                 f"{Path(hm_name).stem}__gaea.png")).replace("\\", "/"),
                    "sea_brightness": round(float(es.last_gaea_sea), 4),
                    "gaea_height_m": round(float(es.last_gaea_height_m), 1),
                    "width_scale": round(float(es.last_gaea_width_scale), 4),
                    "datum_min_m": round(float(sea_level_m - ocean_floor_depth_m), 1),
                    "datum_max_m": round(float(max_elev_m), 1),
                    "floor_metrics": ({k: _json_safe(v) for k, v in sf_info.items()}
                                      if sf_info else None),
                } if gaea_path is not None else None),
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
