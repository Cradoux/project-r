from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import AddonPreferences, PropertyGroup

from . import manifest as manifest_lib
from .operators import sphere_ops


def _addon_id() -> str:
    # Package name: blender_addons.project_r
    return __package__


# ---------------------------------------------------------------------------
# Dynamic erosion-target dropdown
# ---------------------------------------------------------------------------
# Blender keeps only raw C pointers to the identifier/label STRINGS an
# EnumProperty items() callback returns; it does NOT hold the Python objects
# alive. Returning a freshly built local list lets the GC free those strings and
# corrupts/crashes the dropdown. So we keep the list AND its strings referenced at
# module level and return that same object, rebuilding it in place only when the
# manifest changes (keyed on path+mtime, because items() fires on every redraw).
MOST_RECENT_ID = "__MOST_RECENT__"
_SECTION_ENUM_ITEMS: list = []
# Holds the prior generation's tuples (and thus their strings) across one rebuild,
# so a C-side pointer Blender cached from the previous items() call can't dangle
# when we clear() and repopulate the live list.
_SECTION_ENUM_PREV: list = []
_section_cache = {"key": None}


def _manifest_path_for(root: str):
    if not root:
        return None
    from pathlib import Path as _P

    return _P(bpy.path.abspath(root)) / "manifest.json"


def _section_items(self, context):
    items = _SECTION_ENUM_ITEMS
    try:
        root = context.scene.projection_pasta.project_root
    except Exception:
        root = ""

    mp = _manifest_path_for(root)
    manifest_path = str(mp) if mp else ""
    mtime = None
    if mp is not None:
        try:
            mtime = mp.stat().st_mtime if mp.exists() else None
        except OSError:
            mtime = None

    key = (manifest_path, mtime)
    if key == _section_cache["key"] and items:
        return items
    _section_cache["key"] = key

    # Retain the outgoing generation for one cycle before clearing (see above).
    global _SECTION_ENUM_PREV
    _SECTION_ENUM_PREV = list(items)
    items.clear()
    # Synthetic default first: preserves the old "most recently created" fallback
    # and guarantees the list is never empty (an empty items list renders broken).
    items.append((MOST_RECENT_ID, "Most recent", "Erode the most recently created section", "TIME", 0))
    choices = []
    if mp is not None:
        try:
            choices = manifest_lib.section_choices(mp.parent)
        except Exception:
            choices = []
    for n, (sid, name) in enumerate(choices, start=1):
        items.append((str(sid), str(name or sid), f"Erode section '{sid}'", "MESH_GRID", n))
    return items


# ---------------------------------------------------------------------------
# Auto-load a project when the Project Root path changes
# ---------------------------------------------------------------------------
# Calling bpy.ops / writing data from a property update= callback is an unsafe
# restricted context (re-entrancy, broken undo, crashes). The blessed pattern is
# to stash the new path and do the real work from a one-shot bpy.app.timers
# callback, which runs on the main thread in a normal context.
_pending_root = {"path": None}
_loading = False


def _deferred_load_project():
    global _loading
    path = _pending_root["path"]
    mp = _manifest_path_for(path) if path else None
    if mp is None or not mp.exists():
        return None  # not a project folder (yet) -- nothing to auto-load
    _loading = True
    try:
        bpy.ops.pp.open_project()
    except Exception as e:  # pragma: no cover - defensive
        print(f"[Project-R] Auto-load of project failed: {e}")
    finally:
        _loading = False
    return None  # one-shot


def _project_root_update(self, context):
    if _loading:
        return  # ignore writes made by the load itself -> no feedback loop
    _pending_root["path"] = self.project_root
    try:
        if not bpy.app.timers.is_registered(_deferred_load_project):
            bpy.app.timers.register(_deferred_load_project, first_interval=0.0)
    except Exception:
        pass


class ProjectionPastaAddonPreferences(AddonPreferences):
    bl_idname = _addon_id()

    default_project_root: StringProperty(  # type: ignore[valid-type]
        name="Default Project Root",
        description="Default folder to store manifest.json and section exports",
        subtype="DIR_PATH",
        default="",
    )

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        layout.prop(self, "default_project_root")


class ProjectionPastaProjectSettings(PropertyGroup):
    # Project paths
    project_root: StringProperty(  # type: ignore[valid-type]
        name="Project Root",
        description="Folder holding manifest.json plus source/, sections/, processed/ and "
                    "reassembled/. Pointing this at an existing project loads it automatically",
        subtype="DIR_PATH",
        default="",
        update=_project_root_update,
    )

    # Global output config
    global_width: IntProperty(  # type: ignore[valid-type]
        name="Global Width",
        description="Width in pixels of the global equirectangular output (set from the loaded world map)",
        default=3600,
        min=16,
    )
    global_height: IntProperty(  # type: ignore[valid-type]
        name="Global Height",
        description="Height in pixels of the global equirectangular output (half the width for a 2:1 map)",
        default=1800,
        min=16,
    )

    # Hammer full canvas defaults
    hammer_full_width: IntProperty(  # type: ignore[valid-type]
        name="Hammer Full Width",
        description="Default width of the full Hammer projection canvas a section is cropped from",
        default=8192,
        min=64,
    )
    hammer_full_height: IntProperty(  # type: ignore[valid-type]
        name="Hammer Full Height",
        description="Default height of the full Hammer projection canvas a section is cropped from",
        default=4096,
        min=64,
    )

    crop_margin_px: IntProperty(  # type: ignore[valid-type]
        name="Crop Margin (px)",
        description="Extra padding in pixels added around the section's bounding box when cropping",
        default=64,
        min=0,
    )
    square_crop: BoolProperty(  # type: ignore[valid-type]
        name="Square Crop",
        default=True,
        description="Force square crop (useful for Gaea)",
    )
    feather_px: IntProperty(  # type: ignore[valid-type]
        name="Feather (px)",
        default=64,
        min=0,
        description="Edge feather size for blending during reassembly",
    )

    def _overlay_opacity_update(self, context: bpy.types.Context) -> None:
        try:
            sphere_ops.update_overlay_opacity(self.overlay_opacity)
        except Exception:
            pass

    overlay_opacity: bpy.props.FloatProperty(  # type: ignore[valid-type]
        name="Overlay Opacity",
        default=0.6,
        min=0.0,
        max=1.0,
        description="Opacity of extracted-region overlay on the sphere",
        update=_overlay_opacity_update,
    )

    # Sphere tools
    sphere_object_name: StringProperty(  # type: ignore[valid-type]
        name="Sphere Object",
        description="Name of the preview sphere object the world map is projected onto",
        default="ProjectionSphere",
    )

    # UI-only: new section info
    new_section_name: StringProperty(  # type: ignore[valid-type]
        name="Section Name",
        description="Name for the next section created from the selected faces",
        default="NewSection",
    )

    # Advanced: Projection center override (its disclosure is the Section > Advanced
    # sub-panel, whose header checkbox drives this flag).
    override_projection_center: BoolProperty(  # type: ignore[valid-type]
        name="Override Center",
        description="Manually specify the Hammer projection center instead of auto-computing from selection",
        default=False,
    )

    override_center_lon: FloatProperty(  # type: ignore[valid-type]
        name="Longitude",
        description="Override projection center longitude in degrees (-180 to 180)",
        default=0.0,
        min=-180.0,
        max=180.0,
    )

    override_center_lat: FloatProperty(  # type: ignore[valid-type]
        name="Latitude",
        description="Override projection center latitude in degrees (-90 to 90)",
        default=0.0,
        min=-90.0,
        max=90.0,
    )

    # Selection expansion rings
    expand_selection_rings: IntProperty(  # type: ignore[valid-type]
        name="Expand Rings",
        description="Number of face rings to grow/shrink the selection by",
        default=1,
        min=1,
        max=50,
    )

    # Reassembly options
    extend_edge_colors: BoolProperty(  # type: ignore[valid-type]
        name="Extend Edge Colors",
        description="Fill empty areas by extending colors from nearest section edges (useful for ocean)",
        default=False,
    )

    # Planet/world settings
    planet_radius_km: FloatProperty(  # type: ignore[valid-type]
        name="Planet Radius (km)",
        description="Radius of the planet in kilometers (Earth = 6371)",
        default=6371.0,
        min=100.0,
        soft_max=100000.0,
    )

    # Heightmap elevation tracking
    heightmap_filename: StringProperty(  # type: ignore[valid-type]
        name="Heightmap File",
        description="Filename in source/ to track as heightmap (e.g., heightmap.png). Leave empty to disable elevation tracking.",
        default="",
    )

    max_elevation_m: FloatProperty(  # type: ignore[valid-type]
        name="Max Elevation (m)",
        description="Maximum elevation in meters (pure white in heightmap). Default is Mount Everest.",
        default=8849.0,
        min=1.0,
        soft_max=20000.0,
    )

    normalize_heightmaps: BoolProperty(  # type: ignore[valid-type]
        name="Normalize Heights",
        description="Scale heightmaps during reassembly so each section's max matches its calculated elevation",
        default=True,
    )

    # Paths derived
    def project_root_path(self) -> Optional[Path]:
        # Guard on the RAW field: bpy.path.abspath("") resolves to a non-empty path
        # (the blend dir / cwd), so checking the resolved path would treat an unset
        # root as valid and let Create Project write a manifest into the wrong place.
        raw = (self.project_root or "").strip()
        if not raw:
            return None
        return Path(bpy.path.abspath(self.project_root)).resolve()

    def manifest_path(self) -> Optional[Path]:
        root = self.project_root_path()
        if root is None:
            return None
        return root / "manifest.json"


class ProjectionPastaErosionSettings(PropertyGroup):
    """Settings for the erosion stage (carve dendritic drainage into a section heightmap).

    Erosion runs on the section crop, which already sits in the equal-area oblique Hammer
    projection (the geomorphically correct space to erode in), then the existing reassembly
    feather-blends the result back into the global equirect map.
    """

    # --- Target ---
    section: EnumProperty(  # type: ignore[valid-type]
        name="Section",
        description="Which section to erode. 'Most recent' targets the last section created",
        items=_section_items,
        # Dynamic-enum values are stored positionally; SKIP_SAVE avoids persisting an
        # index that could point at a different section after the manifest changes and
        # the file is reloaded -- it just resets to 'Most recent' each session.
        options={"SKIP_SAVE"},
    )
    erode_on_create: BoolProperty(  # type: ignore[valid-type]
        name="Erode after creating",
        description="Immediately run erosion on a section right after Create Section, using the "
                    "current erosion settings below",
        default=False,
    )
    max_work_px: IntProperty(  # type: ignore[valid-type]
        name="Max Work Resolution (px)",
        description="Downsample the crop to at most this size before eroding, then upscale the result "
                    "(keeps Blender responsive on large crops). Erosion is blocking and scales ~linearly "
                    "with pixel count: ~2 min at 256 px, ~7 min at 512 px, ~30 min at 1024 px for 200 "
                    "steps. 256-512 already carves crisp drainage. 0 = erode at native resolution",
        default=512,
        min=0,
        soft_max=4096,
    )

    # --- Seed conditioning (breaks D8 grid-bias -> meandering channels) ---
    noise_kind: EnumProperty(  # type: ignore[valid-type]
        name="Seed Noise",
        description="Fine noise added to the seed BEFORE eroding so channels meander off-grid. "
                    "Smoothed-gauss / fractal break grid-bias; white noise does NOT",
        items=[
            ("none", "None", "No conditioning (channels may snap to grid directions)"),
            ("gauss", "Smoothed Gauss", "Smoothed-gauss texture (the proven grid-bias fix)"),
            ("fractal", "Fractal (fBm)", "Multi-scale 1/f noise (crisper dendritic trunks)"),
            ("white", "White", "Per-pixel white noise (does NOT break grid-bias; for parity only)"),
        ],
        default="gauss",
    )
    noise_amp: FloatProperty(  # type: ignore[valid-type]
        name="Noise Amount",
        description="Conditioning-noise amplitude in 30 m units. ~0.5-0.6 breaks the grid while "
                    "keeping the macro form; 1.0 swamps it",
        default=0.55,
        min=0.0,
        soft_max=2.0,
    )
    noise_seed: IntProperty(  # type: ignore[valid-type]
        name="Noise Seed",
        description="Random seed for the seed-conditioning noise (change for a different channel pattern)",
        default=7,
        min=0,
    )

    # --- Climate forcing (right erosion in the right place) ---
    climate_kind: EnumProperty(  # type: ignore[valid-type]
        name="Climate",
        description="Per-node rainfall field that drives discharge so incision concentrates where it rains",
        items=[
            ("uniform", "Uniform", "Even rainfall everywhere"),
            ("tropical", "Tropical Bands", "Wet equatorial + mid-latitude bands (Hadley-ish)"),
            ("gradient", "Gradient", "Wet on one side fading to dry on the other"),
            ("orographic", "Orographic", "Wet windward + sharp rain shadow (strongest wet/dry contrast)"),
        ],
        default="uniform",
    )
    climate_strength: FloatProperty(  # type: ignore[valid-type]
        name="Climate Strength",
        description="Scales the rainfall contrast away from uniform (0 = flat, 1 = full pattern)",
        default=1.0,
        min=0.0,
        soft_max=3.0,
    )

    # --- Stream-power LEM parameters ---
    k_sp: FloatProperty(  # type: ignore[valid-type]
        name="Erodibility (K)",
        description="Stream-power erodibility K_sp. Smaller crops want a larger K than the global default",
        default=3e-5,
        min=0.0,
        soft_max=1e-3,
        precision=6,
        step=0.01,
    )
    m_sp: FloatProperty(  # type: ignore[valid-type]
        name="Area Exponent (m)",
        description="Stream-power drainage-area exponent. ~0.5 with n=1 gives the mature theta~0.5 concavity",
        default=0.5,
        min=0.0,
        soft_max=2.0,
    )
    n_sp: FloatProperty(  # type: ignore[valid-type]
        name="Slope Exponent (n)",
        description="Stream-power slope exponent. 1.0 is the standard linear case",
        default=1.0,
        min=0.1,
        soft_max=4.0,
    )
    diffusivity: FloatProperty(  # type: ignore[valid-type]
        name="Hillslope Diffusivity",
        description="Linear hillslope diffusion (sets drainage density / valley spacing)",
        default=0.5,
        min=0.0,
        soft_max=10.0,
    )
    uplift: FloatProperty(  # type: ignore[valid-type]
        name="Uplift (U)",
        description="Tectonic uplift rate added each step. Balances incision to keep relief from decaying away",
        default=1e-3,
        min=0.0,
        soft_max=1.0,
        precision=5,
        step=0.01,
    )
    dt: FloatProperty(  # type: ignore[valid-type]
        name="Timestep (dt)",
        description="Years per erosion step. Larger dt erodes faster but risks numerical instability",
        default=1000.0,
        min=1.0,
        soft_max=100000.0,
    )
    steps: IntProperty(  # type: ignore[valid-type]
        name="Steps",
        description="Erosion iterations. ~200 reaches the mature theta~0.5 equilibrium concavity",
        default=200,
        min=1,
        soft_max=2000,
    )

    # --- Multi-scale blur overlay (flat-bottomed channels) ---
    enable_overlay: BoolProperty(  # type: ignore[valid-type]
        name="Channel Overlay",
        description="Engrave flat-bottomed channels onto the eroded surface with a LIGHT, fine-pass "
                    "multi-scale blur overlay (preserves drainage concavity; a deep overlay would destroy it)",
        default=False,
    )
    overlay_depth_m: FloatProperty(  # type: ignore[valid-type]
        name="Overlay Depth (m)",
        description="Carve depth of the overlay. Keep shallow (~150-250 m) so it sharpens channels "
                    "without gouging out the LEM's concavity",
        default=200.0,
        min=0.0,
        soft_max=1000.0,
    )
    overlay_w_macro_km: FloatProperty(  # type: ignore[valid-type]
        name="Widest Valley (km)",
        description="Full width of the widest river valley; sets the coarsest blur in the schedule",
        default=8.0,
        min=0.5,
        soft_max=100.0,
    )
    overlay_r: FloatProperty(  # type: ignore[valid-type]
        name="Blur Ratio (r)",
        description="Geometric blur-shrink factor between passes (coarse -> fine)",
        default=0.4,
        min=0.1,
        max=0.9,
    )

    # --- Output ---
    target_peak_m: FloatProperty(  # type: ignore[valid-type]
        name="Target Peak (m)",
        description="Linearly rescale the eroded land so its max equals this height. "
                    "0 = preserve the section's pre-erosion peak",
        default=0.0,
        min=0.0,
        soft_max=20000.0,
    )

    # --- Last-run quality readout (filled by the operator; transient internal
    # state, not user tunables -- SKIP_SAVE so it doesn't persist in the .blend) ---
    _RO = {"SKIP_SAVE"}
    last_theta: FloatProperty(name="theta", description="Last run slope-area concavity", default=0.0, options=_RO)  # type: ignore[valid-type]
    last_r2: FloatProperty(name="R2", description="Last run slope-area fit quality", default=0.0, options=_RO)  # type: ignore[valid-type]
    last_band_slope: FloatProperty(name="band_slope", description="Last run spectral band slope", default=0.0, options=_RO)  # type: ignore[valid-type]
    last_router: StringProperty(name="router", description="Flow router used in the last run", default="", options=_RO)  # type: ignore[valid-type]
    last_secs: FloatProperty(name="secs", description="Last run wall-clock seconds", default=0.0, options=_RO)  # type: ignore[valid-type]
    last_report: StringProperty(name="report", description="Last run summary line", default="", options=_RO)  # type: ignore[valid-type]


def register() -> None:
    bpy.utils.register_class(ProjectionPastaAddonPreferences)
    bpy.utils.register_class(ProjectionPastaProjectSettings)
    bpy.utils.register_class(ProjectionPastaErosionSettings)


def unregister() -> None:
    bpy.utils.unregister_class(ProjectionPastaErosionSettings)
    bpy.utils.unregister_class(ProjectionPastaProjectSettings)
    bpy.utils.unregister_class(ProjectionPastaAddonPreferences)


