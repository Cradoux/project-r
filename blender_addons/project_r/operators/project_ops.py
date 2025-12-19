from __future__ import annotations

from pathlib import Path

import bpy
from bpy.types import Operator

from .. import manifest as manifest_lib


class PP_OT_init_project(Operator):
    bl_idname = "pp.init_project"
    bl_label = "Init Project"
    bl_description = "Create project folder structure and a new manifest.json if missing"

    def execute(self, context: bpy.types.Context):
        s = context.scene.projection_pasta
        root = s.project_root_path()
        if root is None:
            self.report({"ERROR"}, "Project Root is not set")
            return {"CANCELLED"}

        manifest_lib.init_project_folders(root)
        manifest_path = root / "manifest.json"

        if not manifest_path.exists():
            data = manifest_lib.default_manifest(
                global_size=(s.global_width, s.global_height),
                hammer_full_size=(s.hammer_full_width, s.hammer_full_height),
                crop_margin_px=s.crop_margin_px,
                square_crop=s.square_crop,
                blend_feather_px=s.feather_px,
            )
            manifest_lib.write_manifest(manifest_path, data)

        self.report({"INFO"}, f"Project initialized at {root}")
        return {"FINISHED"}


class PP_OT_open_manifest(Operator):
    bl_idname = "pp.open_manifest"
    bl_label = "Open manifest.json"
    bl_description = "Open the project's manifest.json in the OS file browser"

    def execute(self, context: bpy.types.Context):
        s = context.scene.projection_pasta
        mp = s.manifest_path()
        if mp is None:
            self.report({"ERROR"}, "Project Root is not set")
            return {"CANCELLED"}
        if not mp.exists():
            self.report({"ERROR"}, "manifest.json does not exist (run Init Project)")
            return {"CANCELLED"}

        bpy.ops.wm.path_open(filepath=str(mp))
        return {"FINISHED"}


_CLASSES = (
    PP_OT_init_project,
    PP_OT_open_manifest,
)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)


