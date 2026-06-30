from __future__ import annotations

import subprocess
import sys

import bpy
from bpy.types import Operator

from .. import manifest as manifest_lib


def _load_existing_project(op: Operator, context: bpy.types.Context, root, manifest_path) -> bool:
    """Read the manifest and (re)load its world map + overlay onto the sphere.
    Returns True on success. Shared by Open Project and the auto-load timer."""
    manifest = manifest_lib.read_manifest(manifest_path)
    world_map_info = manifest.get("global", {}).get("world_map", {})
    stored_world_path = world_map_info.get("path", "")

    world_map_path = manifest_lib.resolve_source_path(root, stored_world_path)
    if world_map_path is not None:
        # Loading via the resolved (existing) path makes load_world_map rewrite the
        # manifest with a portable, project-relative path, so a project copied from
        # another machine self-heals on open.
        bpy.ops.pp.load_world_map(filepath=str(world_map_path))
        op.report({"INFO"}, f"Loaded project at {root}")
        return True
    op.report(
        {"WARNING"},
        f"Opened project at {root}, but its world map is missing "
        f"(manifest path '{stored_world_path}' not found in source/). Use Load World Map.",
    )
    return False


class PP_OT_init_project(Operator):
    bl_idname = "pp.init_project"
    bl_label = "Create Project"
    bl_description = "Create the project folder structure and a fresh manifest.json in the Project Root"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        s = getattr(context.scene, "projection_pasta", None)
        if s is None or not s.project_root_path():
            cls.poll_message_set("Set a Project Root first")
            return False
        return True

    def execute(self, context: bpy.types.Context):
        s = context.scene.projection_pasta
        root = s.project_root_path()
        if root is None:
            self.report({"ERROR"}, "Project Root is not set")
            return {"CANCELLED"}

        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            # Don't clobber an existing project; just open it instead.
            _load_existing_project(self, context, root, manifest_path)
            return {"FINISHED"}

        manifest_lib.init_project_folders(root)
        data = manifest_lib.default_manifest(
            global_size=(s.global_width, s.global_height),
            hammer_full_size=(s.hammer_full_width, s.hammer_full_height),
            crop_margin_px=s.crop_margin_px,
            square_crop=s.square_crop,
            blend_feather_px=s.feather_px,
        )
        manifest_lib.write_manifest(manifest_path, data)
        self.report({"INFO"}, f"Project created at {root}. Now use Load World Map.")
        return {"FINISHED"}


class PP_OT_open_project(Operator):
    bl_idname = "pp.open_project"
    bl_label = "Open Project"
    bl_description = "Load the existing project at the Project Root (manifest, world map and overlay)"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        s = getattr(context.scene, "projection_pasta", None)
        mp = s.manifest_path() if s is not None else None
        if mp is None or not mp.exists():
            cls.poll_message_set("No manifest.json found in the Project Root")
            return False
        return True

    def execute(self, context: bpy.types.Context):
        s = context.scene.projection_pasta
        root = s.project_root_path()
        mp = s.manifest_path()
        if root is None or mp is None or not mp.exists():
            self.report({"ERROR"}, "No project found at the Project Root")
            return {"CANCELLED"}
        _load_existing_project(self, context, root, mp)
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

        # Reveal the file in the OS file browser. We deliberately avoid
        # wm.path_open / os.startfile on the .json itself: on Windows a file
        # extension with no associated application raises OSError
        # ("Application not found", WinError -2147221003) -- which is exactly
        # what was crashing here. Selecting the file in the file manager works
        # regardless of file associations.
        try:
            if sys.platform == "win32":
                # explorer returns exit code 1 even on success, so fire-and-forget.
                subprocess.Popen(["explorer", "/select,", str(mp)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(mp)])
            else:
                # Linux/other: no portable "reveal", so open the containing folder.
                bpy.ops.wm.path_open(filepath=str(mp.parent))
        except Exception as e:
            self.report({"ERROR"}, f"Could not open manifest location: {e}")
            return {"CANCELLED"}

        return {"FINISHED"}


_CLASSES = (
    PP_OT_init_project,
    PP_OT_open_project,
    PP_OT_open_manifest,
)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)


