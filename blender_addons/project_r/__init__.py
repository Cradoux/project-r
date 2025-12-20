from __future__ import annotations

import importlib
import subprocess
import sys

import bpy
from bpy.types import Operator

bl_info = {
    "name": "Project-R",
    "author": "Project-R (Kilroys Katography) + AI",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Project-R",
    "description": "Export Hammer (oblique) section crops from equirectangular maps and reassemble them back (powered by projectionpasta).",
    "category": "Import-Export",
}


def is_scipy_available() -> bool:
    try:
        import scipy.ndimage
        return True
    except ImportError:
        return False


class PP_OT_install_dependencies(Operator):
    bl_idname = "pp.install_dependencies"
    bl_label = "Install Dependencies (scipy)"
    bl_description = "Install scipy using Blender's Python pip"

    def execute(self, context):
        python = sys.executable
        try:
            # Ensure pip is available
            subprocess.check_call([python, "-m", "ensurepip", "--upgrade"])
        except Exception:
            pass  # pip may already be available

        try:
            subprocess.check_call([python, "-m", "pip", "install", "--upgrade", "scipy"])
            self.report({"INFO"}, "scipy installed successfully! Please restart Blender.")
            return {"FINISHED"}
        except subprocess.CalledProcessError as e:
            self.report({"ERROR"}, f"Failed to install scipy: {e}")
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Unexpected error: {e}")
            return {"CANCELLED"}


from . import props as _props
from . import ui as _ui
from .operators import project_ops as _project_ops
from .operators import reassemble_ops as _reassemble_ops
from .operators import section_ops as _section_ops
from .operators import sphere_ops as _sphere_ops


_MODULES = (
    _props,
    _project_ops,
    _sphere_ops,
    _section_ops,
    _reassemble_ops,
    _ui,
)


def _reload_modules_for_dev() -> None:
    # Helpful during development: Blender reloads addons without restarting,
    # but submodules can remain cached.
    for m in _MODULES:
        importlib.reload(m)


def register() -> None:
    _reload_modules_for_dev()

    bpy.utils.register_class(PP_OT_install_dependencies)

    for m in _MODULES:
        if hasattr(m, "register"):
            m.register()

    bpy.types.Scene.projection_pasta = bpy.props.PointerProperty(
        type=_props.ProjectionPastaProjectSettings
    )


def unregister() -> None:
    if hasattr(bpy.types.Scene, "projection_pasta"):
        del bpy.types.Scene.projection_pasta

    for m in reversed(_MODULES):
        if hasattr(m, "unregister"):
            m.unregister()

    bpy.utils.unregister_class(PP_OT_install_dependencies)


