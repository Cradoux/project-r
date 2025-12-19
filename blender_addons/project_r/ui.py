from __future__ import annotations

import bpy
from bpy.types import Panel


class PP_PT_main(Panel):
    bl_label = "Project-R"
    bl_idname = "PP_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Project-R"

    def draw(self, context: bpy.types.Context) -> None:
        s = context.scene.projection_pasta
        layout = self.layout

        box = layout.box()
        box.label(text="Project")
        box.prop(s, "project_root")
        row = box.row(align=True)
        row.operator("pp.init_project", text="Init Project")
        row.operator("pp.open_manifest", text="Open manifest.json")

        box = layout.box()
        box.label(text="Sphere")
        box.prop(s, "sphere_object_name")
        row = box.row(align=True)
        row.operator("pp.load_world_map", text="Load World Map")
        row.operator("pp.create_sphere", text="Create Sphere")
        box.prop(s, "expand_selection_rings")
        box.operator("pp.expand_selection", text="Expand Selection")

        box = layout.box()
        box.label(text="Section Export")
        box.prop(s, "new_section_name")
        box.prop(s, "hammer_full_width")
        box.prop(s, "hammer_full_height")
        box.prop(s, "crop_margin_px")
        box.prop(s, "square_crop")
        box.prop(s, "feather_px")
        box.operator("pp.create_section", text="Create Section from Selected Faces")

        box = layout.box()
        box.label(text="Reassembly")
        row = box.row(align=True)
        row.operator("pp.validate_processed", text="Validate")
        row.operator("pp.reassemble", text="Reassemble")


def register() -> None:
    bpy.utils.register_class(PP_PT_main)


def unregister() -> None:
    bpy.utils.unregister_class(PP_PT_main)


