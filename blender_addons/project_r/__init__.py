from __future__ import annotations

import importlib
import os

import bpy
from bpy.app.handlers import persistent

# Legacy add-on metadata. On Blender 4.2+ the extension is described by
# blender_manifest.toml (which takes precedence); bl_info is kept so the source
# still loads if installed via the legacy "Install from Disk" path and to document
# the version in one place. Keep the version in sync with blender_manifest.toml.
bl_info = {
    "name": "Project-R",
    "author": "Project-R (Kilroys Katography) + AI",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Project-R",
    "description": "Split equirectangular world maps into low-distortion Hammer sections, "
                   "erode them in-Blender, and reassemble (powered by projectionpasta).",
    "category": "Import-Export",
}


from . import deps as _deps
from . import layers as _layers  # noqa: F401  (pure helper; imported so dev-reload covers it)
from . import props as _props
from . import ui as _ui
from .operators import erode_ops as _erode_ops
from .operators import project_ops as _project_ops
from .operators import reassemble_ops as _reassemble_ops
from .operators import section_ops as _section_ops
from .operators import sphere_ops as _sphere_ops


_MODULES = (
    _deps,
    _props,
    _project_ops,
    _sphere_ops,
    _section_ops,
    _reassemble_ops,
    _erode_ops,
    _ui,
)


def _reload_modules_for_dev() -> None:
    # Development convenience (opt-in via the PROJECT_R_DEV env var). Blender
    # re-registers an addon without restarting, but `import` returns the cached
    # submodule from sys.modules, so edits to helper modules stay invisible until
    # reloaded explicitly. Reload leaf helpers first so the operator/UI modules
    # pick up fresh helper code; order matters for `from X import name` imports.
    # (The vendored projectionpasta is loaded lazily and rarely edited -- restart
    # Blender if you change it.) This used to run on every enable, including for
    # end users; gating it keeps production registration to just class registration.
    from . import imaging, geo, manifest, erosion, projection_backend, layers, deps
    for m in (imaging, geo, manifest, erosion, projection_backend, layers, deps):
        importlib.reload(m)
    for m in _MODULES:
        importlib.reload(m)


def _seed_default_root():
    """Seed the scene's project_root from the addon preference when it's empty.

    Setting the property fires its update callback, which auto-loads the project
    if that folder already has a manifest.json. Runs from a timer so it executes
    in a normal (non-restricted) context."""
    try:
        addon = bpy.context.preferences.addons.get(__package__)
        default_root = getattr(addon.preferences, "default_project_root", "") if addon else ""
        scene = getattr(bpy.context, "scene", None)
        s = getattr(scene, "projection_pasta", None) if scene is not None else None
        if s is not None and not s.project_root and default_root:
            s.project_root = default_root
    except Exception:
        pass
    return None  # one-shot


@persistent
def _on_load_post(_dummy) -> None:
    # After a .blend opens, (re)seed + auto-load via a one-shot timer.
    try:
        bpy.app.timers.register(_seed_default_root, first_interval=0.0)
    except Exception:
        pass


def register() -> None:
    if os.environ.get("PROJECT_R_DEV"):
        _reload_modules_for_dev()

    for m in _MODULES:
        if hasattr(m, "register"):
            m.register()

    bpy.types.Scene.projection_pasta = bpy.props.PointerProperty(
        type=_props.ProjectionPastaProjectSettings
    )
    bpy.types.Scene.projection_pasta_erosion = bpy.props.PointerProperty(
        type=_props.ProjectionPastaErosionSettings
    )

    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    try:
        bpy.app.timers.register(_seed_default_root, first_interval=0.0)
    except Exception:
        pass


def unregister() -> None:
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)

    # Drop any pending one-shot timers so disabling (or re-enabling) the addon can't
    # fire a stale auto-load/seed callback after its module state is gone.
    for fn in (_seed_default_root, getattr(_props, "_deferred_load_project", None)):
        try:
            if fn is not None and bpy.app.timers.is_registered(fn):
                bpy.app.timers.unregister(fn)
        except Exception:
            pass

    if hasattr(bpy.types.Scene, "projection_pasta_erosion"):
        del bpy.types.Scene.projection_pasta_erosion
    if hasattr(bpy.types.Scene, "projection_pasta"):
        del bpy.types.Scene.projection_pasta

    for m in reversed(_MODULES):
        if hasattr(m, "unregister"):
            m.unregister()
