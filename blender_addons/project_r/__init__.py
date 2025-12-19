from __future__ import annotations

import importlib

import bpy

bl_info = {
    "name": "Project-R",
    "author": "Project-R (Kilroys Katography) + AI",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Project-R",
    "description": "Export Hammer (oblique) section crops from equirectangular maps and reassemble them back (powered by projectionpasta).",
    "category": "Import-Export",
}

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


