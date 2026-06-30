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
    ocean_floor_depth_m: FloatProperty(  # type: ignore[valid-type]
        name="Ocean Floor Depth (m)",
        description="Depth of the deepest ocean floor below sea level, WORLD-WIDE (pure black in the "
                    "Gaea sea-floor export). With Max Elevation it sets the single elevation range every "
                    "section is encoded against, so sea level lands at the SAME brightness in every "
                    "section (otherwise sections show colour seams in Gaea). ~6000 m is a deep abyssal plain",
        default=6000.0,
        min=0.0,
        soft_max=11000.0,
    )

    normalize_heightmaps: BoolProperty(  # type: ignore[valid-type]
        name="Normalize Heights",
        description="Scale heightmaps during reassembly so each section's max matches its calculated elevation",
        default=True,
    )

    # Target output resolution (longest edge) for processed section maps + the
    # in-Blender erosion detail, and the reassembled global map. One quality knob.
    output_resolution: EnumProperty(  # type: ignore[valid-type]
        name="Output Resolution",
        description="Per-SECTION longest-edge pixel size: the exported section crops and the "
                    "in-Blender erosion detail. 'Auto' picks a balanced size for the section; "
                    "higher = finer detail, but erosion time scales ~linearly with pixel count. "
                    "(The reassembled global map size is set separately under Reassembly.)",
        items=[
            ("AUTO", "Auto (optimal)", "A balanced size derived from the section's native resolution"),
            ("512", "512 px", "512 px longest edge"),
            ("1024", "1024 px", "1024 px longest edge"),
            ("2048", "2048 px", "2048 px longest edge (slow erosion)"),
            ("4096", "4096 px", "4096 px longest edge (very slow erosion)"),
            ("8192", "8192 px", "8192 px longest edge (extremely slow erosion)"),
        ],
        default="AUTO",
    )

    # Final reassembled global map size -- a GLOBAL (world-scale) resolution, kept
    # separate from the per-section output_resolution so a per-section detail choice
    # can never silently shrink the world deliverable.
    reassembly_resolution: EnumProperty(  # type: ignore[valid-type]
        name="Reassembly Resolution",
        description="Longest-edge pixel size of the final reassembled global equirectangular map. "
                    "'Auto' keeps the loaded world map's size",
        items=[
            ("AUTO", "Auto (world size)", "Use the loaded world map's resolution"),
            ("4096", "4096 px", "4096 px longest edge"),
            ("8192", "8192 px", "8192 px longest edge"),
            ("16384", "16384 px", "16384 px longest edge"),
        ],
        default="AUTO",
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

    # --- Presets (scale x intensity) -------------------------------------
    lem_scale: EnumProperty(  # type: ignore[valid-type]
        name="Scale",
        description="Physical scale preset. 'Auto' tunes the physics to the section's real-world "
                    "size; 'Custom' exposes every parameter for manual control",
        items=[
            ("AUTO", "Auto (by size)", "Pick the scale band from the section's km extent"),
            ("LOCAL", "Local (<500 km)", "Local-scale physics"),
            ("REGIONAL", "Regional (<1500 km)", "Regional-scale physics"),
            ("CONTINENTAL", "Continental (<4000 km)", "Continental-scale physics"),
            ("SUPER", "Supercontinental", "Supercontinental-scale physics"),
            ("CUSTOM", "Custom", "Use the manual parameters below instead of a preset"),
        ],
        default="AUTO",
    )
    lem_intensity: EnumProperty(  # type: ignore[valid-type]
        name="Intensity",
        description="How developed the river network gets. Lower = subtler drainage that keeps more "
                    "of the original terrain",
        items=[
            ("GENTLE", "Gentle", "Fewer steps, softer incision"),
            ("MODERATE", "Moderate", "Balanced"),
            ("STRONG", "Strong", "More steps, deeper incision"),
        ],
        default="MODERATE",
    )
    erosion_strength: FloatProperty(  # type: ignore[valid-type]
        name="Erosion Strength",
        description="Blend the eroded result back toward your original heightmap. 1.0 = full "
                    "erosion; lower keeps more of the input shape (the direct cure for an "
                    "over-eroded look). Ocean is always preserved regardless",
        default=0.7,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    sea_level_m: FloatProperty(  # type: ignore[valid-type]
        name="Sea Level (m)",
        description="Cells at or below this elevation are treated as ocean: pinned as a fixed "
                    "base level the rivers drain into, never uplifted or eroded, and restored "
                    "unchanged afterwards. 0 matches a pure-black ocean",
        default=0.0,
        soft_min=-1000.0,
        soft_max=2000.0,
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
        description="Seed-conditioning noise as a fraction of the terrain's relief (scales with "
                    "terrain height, so it works at any elevation). ~0.5 perturbs the drainage while "
                    "keeping the macro form; higher reshapes it more. Change the Noise Seed for a "
                    "different river pattern",
        default=0.55,
        min=0.0,
        soft_max=3.0,
    )
    noise_seed: IntProperty(  # type: ignore[valid-type]
        name="Noise Seed",
        description="Random seed for the seed-conditioning noise (change for a different channel pattern)",
        default=7,
        min=0,
    )

    # --- Rainfall map (optional per-node runoff that drives where incision concentrates) ---
    rainfall_filename: StringProperty(  # type: ignore[valid-type]
        name="Rainfall Map",
        description="Filename of a rainfall/precipitation image in source/ (brighter = wetter) used "
                    "to drive erosion. The section's crop of it is used; leave empty for uniform "
                    "rainfall. Auto-detected if a source map is named with 'rain'/'precip'",
        default="",
    )

    # --- Optional spatial drivers (per-cell fields cropped per section) ---
    uplift_filename: StringProperty(  # type: ignore[valid-type]
        name="Uplift Map",
        description="Filename in source/ of an orogeny/uplift-intensity map (brighter = more uplift). "
                    "Decoded by luminance; concentrates relief in active belts. Empty = uniform uplift",
        default="",
    )
    uplift_influence: FloatProperty(  # type: ignore[valid-type]
        name="Uplift Influence",
        description="How strongly the Uplift Map modulates uplift. 0 = uniform (ignore the map); "
                    "1 = uplift scales fully with map brightness (no uplift where the map is dark)",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    erodibility_filename: StringProperty(  # type: ignore[valid-type]
        name="Erodibility Map",
        description="Filename in source/ of a CONTINUOUS rock-softness map (brighter = softer = erodes "
                    "faster). Decoded by luminance. Categorical rock-type maps need a class legend and "
                    "are not supported here yet. Empty = uniform erodibility",
        default="",
    )
    erodibility_contrast: FloatProperty(  # type: ignore[valid-type]
        name="Erodibility Contrast",
        description="Spread of erodibility (K) around the base value. 1 = uniform (ignore the map); "
                    "higher = softer rock erodes much faster than hard rock (K x contrast at brightest, "
                    "K / contrast at darkest)",
        default=1.0,
        min=1.0,
        soft_max=8.0,
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
        description="Engrave flat-bottomed, MEANDERING channels onto the eroded surface using a "
                    "multi-flow router (off-grid, unlike the D8 stream-power pass) plus a light "
                    "multi-scale blur. This is what makes rivers curve instead of snapping to the "
                    "grid; keep it light so it preserves the drainage concavity",
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

    # --- Glacial (fjord) erosion (runs BEFORE coastal and the LEM; carves U-troughs) ---
    enable_glacial: BoolProperty(  # type: ignore[valid-type]
        name="Glacial Erosion",
        description="Carve glacial U-troughs and over-deepened basins BEFORE the coast and rivers: "
                    "ice gathers above the snowline, flows downhill, and grinds valleys that can drop "
                    "BELOW sea level -- so once flooded they become fjords. Runs first, as the earliest "
                    "structural pre-pass; the coast and rivers then work the glaciated terrain",
        default=False,
    )
    glacial_ela_frac: FloatProperty(  # type: ignore[valid-type]
        name="Snowline Height",
        description="Height of the permanent snowline (equilibrium line altitude) as a fraction of the "
                    "section's relief above sea level. LOWER = more of the map sits under ice = deeper, "
                    "more widespread fjords; higher = only the tallest peaks glaciate",
        default=0.30,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    glacial_k_g: FloatProperty(  # type: ignore[valid-type]
        name="Carving Strength",
        description="Glacial erosion coefficient (K_g). Linearly scales trough depth. The default is the "
                    "published calibration; raise for deeper troughs, but prefer LOWERING the Snowline "
                    "for deep SUSTAINED fjords (too-strong carving can eat the icefield's own snowfield)",
        default=1.9e-5,
        min=0.0,
        soft_max=1.0e-4,
        precision=6,
    )
    glacial_quarry_mult: FloatProperty(  # type: ignore[valid-type]
        name="Quarrying",
        description="Extra plucking of bedrock steps (risers) on top of abrasion. Higher = blockier, more "
                    "deeply over-deepened trough floors (real fjords need this; abrasion alone under-cuts)",
        default=1.0,
        min=0.0,
        soft_max=4.0,
    )
    glacial_diffuse: FloatProperty(  # type: ignore[valid-type]
        name="U-Trough Smoothing",
        description="Lateral smoothing under thick ice that rounds valley cross-sections from V-shaped "
                    "(river) toward U-shaped (glacial). 0 = off",
        default=0.3,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    glacial_steps: IntProperty(  # type: ignore[valid-type]
        name="Glacial Steps",
        description="Ice-erosion iterations. More steps = deeper troughs and longer fjords (and more "
                    "compute)",
        default=120,
        min=1,
        soft_max=400,
    )

    # --- Coastal (wave) erosion (runs BEFORE the LEM; reworks the coastline) ---
    enable_coastal: BoolProperty(  # type: ignore[valid-type]
        name="Coastal Erosion",
        description="Rework the shoreline with wave energy BEFORE carving rivers: exposed headlands "
                    "(long fetch over open water) erode into cliffs and retreat, while the removed "
                    "sediment is redeposited as beaches in sheltered bays. Unlike the river stage, "
                    "this actively MOVES the coastline (the river stage keeps it fixed)",
        default=False,
    )
    coastal_rate_m: FloatProperty(  # type: ignore[valid-type]
        name="Wave Erosion Rate (m)",
        description="Max vertical lowering per step at the most wave-exposed waterline cell. Higher "
                    "= faster cliff retreat",
        default=3.0,
        min=0.0,
        soft_max=30.0,
    )
    coastal_steps: IntProperty(  # type: ignore[valid-type]
        name="Coastal Steps",
        description="Wave-erosion iterations. More steps = more coastline retreat and smoother bays",
        default=25,
        min=1,
        soft_max=200,
    )
    coastal_notch_m: FloatProperty(  # type: ignore[valid-type]
        name="Wave Reach (m)",
        description="How far up the cliff face waves bite, above sea level. Larger = taller cliffs "
                    "attacked per step",
        default=20.0,
        min=0.5,
        soft_max=200.0,
    )
    coastal_max_fetch_km: FloatProperty(  # type: ignore[valid-type]
        name="Max Fetch (km)",
        description="Open-water distance past which extra fetch no longer adds wave energy. Also "
                    "caps cost: compute scales with fetch / cell size",
        default=25.0,
        min=1.0,
        soft_max=200.0,
    )
    coastal_swell_focus: FloatProperty(  # type: ignore[valid-type]
        name="Swell Focus",
        description="How directional the wave climate is. 0 = waves from all directions (coasts "
                    "erode evenly); 1 = a dominant swell from 'Swell Direction' dominates",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    coastal_swell_deg: FloatProperty(  # type: ignore[valid-type]
        name="Swell Direction",
        description="Compass bearing the dominant swell comes FROM (0=N, 90=E, 180=S, 270=W). Only "
                    "matters when Swell Focus > 0",
        default=270.0,
        min=0.0,
        max=360.0,
    )
    coastal_talus_deg: FloatProperty(  # type: ignore[valid-type]
        name="Cliff Collapse (deg)",
        description="Talus angle for undercut cliff faces near the coast: steeper-than-this faces "
                    "collapse so tall cliffs retreat instead of standing vertical. 0 = off",
        default=0.0,
        min=0.0,
        max=89.0,
    )
    coastal_deposition: BoolProperty(  # type: ignore[valid-type]
        name="Build Beaches",
        description="Redeposit eroded sediment as beaches in sheltered shallow water (mass-"
                    "conserving). Off = pure erosion, sediment lost offshore",
        default=True,
    )

    # --- Sea floor / bathymetry + Gaea export (consistent global sea datum) ---
    enable_seafloor: BoolProperty(  # type: ignore[valid-type]
        name="Sea Floor",
        description="Fill the ocean with a realistic continental margin (shelf -> shelf break -> slope "
                    "-> abyssal floor), keeping the glacial fjords as deep troughs, and write a separate "
                    "Gaea export (<map>__gaea.png) that encodes land AND sea against one world-wide "
                    "elevation range -- so sea level is at the same brightness in every section. The "
                    "Reassembly heightmap is unaffected",
        default=False,
    )
    seafloor_shelf_depth_m: FloatProperty(  # type: ignore[valid-type]
        name="Shelf Break Depth (m)",
        description="Water depth at the shelf break, where the gentle continental shelf gives way to the "
                    "steeper slope (~130 m on Earth)",
        default=130.0,
        min=1.0,
        soft_max=600.0,
    )
    seafloor_shelf_width_km: FloatProperty(  # type: ignore[valid-type]
        name="Shelf Width (km)",
        description="Width of the continental shelf off a LOWLAND coast. Mountainous coasts get a "
                    "proportionally narrower shelf (see Shelf vs Relief)",
        default=60.0,
        min=0.0,
        soft_max=400.0,
    )
    seafloor_shelf_relief_mod: FloatProperty(  # type: ignore[valid-type]
        name="Shelf vs Relief",
        description="How strongly the bordering land's height narrows the shelf. 0 = uniform width "
                    "everywhere; 1 = mountainous coasts plunge straight to deep water (active margin), "
                    "lowlands keep a broad shallow shelf (passive margin)",
        default=0.7,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    seafloor_slope_width_km: FloatProperty(  # type: ignore[valid-type]
        name="Slope Width (km)",
        description="Width of the continental slope: the distance over which the floor drops from the "
                    "shelf break down to the abyssal plain (Ocean Floor Depth)",
        default=40.0,
        min=1.0,
        soft_max=300.0,
    )
    seafloor_bathy_filename: StringProperty(  # type: ignore[valid-type]
        name="Bathymetry Map",
        description="Optional crop name (in this section's crops/) of a painted/real depth map: white = "
                    "deepest (Ocean Floor Depth), black = shoreline. Blended over the procedural shelf and "
                    "still unioned with the fjords. Empty = fully procedural sea floor",
        default="",
    )
    seafloor_input_weight: FloatProperty(  # type: ignore[valid-type]
        name="Bathymetry Map Weight",
        description="How strongly the Bathymetry Map overrides the procedural shelf (1 = use the map, "
                    "0 = ignore it). Only matters when a Bathymetry Map is set",
        default=1.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
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
    last_gaea_sea: FloatProperty(name="gaea_sea", description="Sea-level brightness to set in Gaea (last run)", default=0.0, options=_RO)  # type: ignore[valid-type]
    last_gaea_height_m: FloatProperty(name="gaea_height", description="Gaea 'Height' value in metres (last run)", default=0.0, options=_RO)  # type: ignore[valid-type]
    last_gaea_width_scale: FloatProperty(name="gaea_width_scale", description="Terrain-width multiplier to keep height:width ratio (last run)", default=1.0, options=_RO)  # type: ignore[valid-type]


def register() -> None:
    bpy.utils.register_class(ProjectionPastaAddonPreferences)
    bpy.utils.register_class(ProjectionPastaProjectSettings)
    bpy.utils.register_class(ProjectionPastaErosionSettings)


def unregister() -> None:
    bpy.utils.unregister_class(ProjectionPastaErosionSettings)
    bpy.utils.unregister_class(ProjectionPastaProjectSettings)
    bpy.utils.unregister_class(ProjectionPastaAddonPreferences)


