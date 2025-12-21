from __future__ import annotations

import bpy
from bpy.types import Panel

from . import is_scipy_available, is_pillow_available


class PP_PT_main(Panel):
    bl_label = "Project-R"
    bl_idname = "PP_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Project-R"

    def draw(self, context: bpy.types.Context) -> None:
        s = context.scene.projection_pasta
        layout = self.layout

        # Dependencies check
        missing_deps = []
        if not is_pillow_available():
            missing_deps.append("Pillow")
        if not is_scipy_available():
            missing_deps.append("scipy")

        if missing_deps:
            box = layout.box()
            box.alert = True
            box.label(text=f"Missing: {', '.join(missing_deps)}", icon="ERROR")
            box.operator("pp.install_dependencies", text="Install Dependencies")
            box.label(text="(Restart Blender after install)")

        box = layout.box()
        box.label(text="Project")
        box.prop(s, "project_root")
        box.prop(s, "planet_radius_km")
        row = box.row(align=True)
        row.operator("pp.init_project", text="Init Project")
        row.operator("pp.open_manifest", text="Open manifest.json")

        box = layout.box()
        box.label(text="Sphere")
        row = box.row(align=True)
        row.operator("pp.load_world_map", text="Load World Map")
        row = box.row(align=True)
        row.operator("pp.expand_selection", text="Expand")
        row.operator("pp.shrink_selection", text="Reduce")
        box.prop(s, "overlay_opacity")

        box = layout.box()
        box.label(text="Section Export")
        box.prop(s, "new_section_name")
        box.prop(s, "square_crop")
        box.prop(s, "feather_px")
        box.separator()
        box.label(text="Heightmap Tracking (optional):")
        box.prop(s, "heightmap_filename")
        box.prop(s, "max_elevation_m")
        box.operator("pp.create_section", text="Create Section from Selected Faces")

        box = layout.box()
        box.label(text="Reassembly")
        box.prop(s, "extend_edge_colors")
        box.prop(s, "normalize_heightmaps")
        row = box.row(align=True)
        row.operator("pp.validate_processed", text="Validate")
        row.operator("pp.reassemble", text="Reassemble")


def register() -> None:
    bpy.utils.register_class(PP_PT_main)


def unregister() -> None:
    bpy.utils.unregister_class(PP_PT_main)


