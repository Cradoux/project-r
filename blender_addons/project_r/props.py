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


def register() -> None:
    bpy.utils.register_class(ProjectionPastaAddonPreferences)
    bpy.utils.register_class(ProjectionPastaProjectSettings)


def unregister() -> None:
    bpy.utils.unregister_class(ProjectionPastaProjectSettings)
    bpy.utils.unregister_class(ProjectionPastaAddonPreferences)


