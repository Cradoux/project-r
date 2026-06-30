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

from .operators import sphere_ops


def _addon_id() -> str:
    # Package name: blender_addons.project_r
    return __package__


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
        subtype="DIR_PATH",
        default="",
    )

    # Global output config
    global_width: IntProperty(  # type: ignore[valid-type]
        name="Global Width",
        default=3600,
        min=16,
    )
    global_height: IntProperty(  # type: ignore[valid-type]
        name="Global Height",
        default=1800,
        min=16,
    )

    # Hammer full canvas defaults
    hammer_full_width: IntProperty(  # type: ignore[valid-type]
        name="Hammer Full Width",
        default=8192,
        min=64,
    )
    hammer_full_height: IntProperty(  # type: ignore[valid-type]
        name="Hammer Full Height",
        default=4096,
        min=64,
    )

    crop_margin_px: IntProperty(  # type: ignore[valid-type]
        name="Crop Margin (px)",
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
        default="ProjectionSphere",
    )

    # UI-only: new section info
    new_section_name: StringProperty(  # type: ignore[valid-type]
        name="Section Name",
        default="NewSection",
    )

    # Advanced: Projection center override
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
        p = Path(bpy.path.abspath(self.project_root)).resolve()
        if not str(p).strip():
            return None
        return p

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
    section_id: StringProperty(  # type: ignore[valid-type]
        name="Section",
        description="Section id (or name) to erode. Leave empty to use the most recently created section",
        default="",
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
        default=0.5,
        min=0.0,
        soft_max=2.0,
    )
    n_sp: FloatProperty(  # type: ignore[valid-type]
        name="Slope Exponent (n)",
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
        default=1e-3,
        min=0.0,
        soft_max=1.0,
        precision=5,
        step=0.01,
    )
    dt: FloatProperty(  # type: ignore[valid-type]
        name="Timestep (dt)",
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

    # --- Wilbur multi-scale blur overlay (flat-bottomed channels) ---
    enable_overlay: BoolProperty(  # type: ignore[valid-type]
        name="Wilbur Overlay",
        description="Engrave flat-bottomed channels onto the eroded surface with a LIGHT, fine-pass "
                    "Incise-Flow overlay (preserves drainage concavity; a deep overlay would destroy it)",
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

    # --- UI state ---
    show_lem_advanced: BoolProperty(  # type: ignore[valid-type]
        name="Advanced LEM Settings",
        default=False,
    )

    # --- Last-run quality readout (filled by the operator) ---
    last_theta: FloatProperty(name="theta", default=0.0)  # type: ignore[valid-type]
    last_r2: FloatProperty(name="R2", default=0.0)  # type: ignore[valid-type]
    last_band_slope: FloatProperty(name="band_slope", default=0.0)  # type: ignore[valid-type]
    last_router: StringProperty(name="router", default="")  # type: ignore[valid-type]
    last_secs: FloatProperty(name="secs", default=0.0)  # type: ignore[valid-type]
    last_report: StringProperty(name="report", default="")  # type: ignore[valid-type]


def register() -> None:
    bpy.utils.register_class(ProjectionPastaAddonPreferences)
    bpy.utils.register_class(ProjectionPastaProjectSettings)
    bpy.utils.register_class(ProjectionPastaErosionSettings)


def unregister() -> None:
    bpy.utils.unregister_class(ProjectionPastaErosionSettings)
    bpy.utils.unregister_class(ProjectionPastaProjectSettings)
    bpy.utils.unregister_class(ProjectionPastaAddonPreferences)


