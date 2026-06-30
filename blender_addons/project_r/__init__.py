from __future__ import annotations

import importlib
import subprocess
import sys
import site

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
        ensure_user_site_on_path()
        import scipy.ndimage
        return True
    except ImportError:
        return False


def is_pillow_available() -> bool:
    try:
        ensure_user_site_on_path()
        from PIL import Image
        # Verify the C extension actually works (this is what fails for the user)
        Image.new("RGB", (1, 1))
        return True
    except Exception:
        return False


def is_landlab_available() -> bool:
    try:
        ensure_user_site_on_path()
        import landlab  # noqa: F401
        from landlab.components import FastscapeEroder, LinearDiffuser  # noqa: F401
        return True
    except Exception:
        return False


def is_priorityflood_available() -> bool:
    """The fast GPL flow router (richdem). Optional: erosion falls back to an MIT router without it."""
    try:
        ensure_user_site_on_path()
        from landlab.components import PriorityFloodFlowRouter  # noqa: F401
        import richdem  # noqa: F401
        return True
    except Exception:
        return False


def ensure_user_site_on_path() -> None:
    try:
        user_site = site.getusersitepackages()
    except Exception:
        return

    if user_site and user_site not in sys.path:
        # Blender installs in Program Files often need user-site packages.
        sys.path.append(user_site)


class PP_OT_install_dependencies(Operator):
    bl_idname = "pp.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = "Install required packages (Pillow, scipy, landlab, richdem) using Blender's Python pip"

    def execute(self, context):
        python = sys.executable
        ensure_user_site_on_path()
        try:
            # Ensure pip is available
            subprocess.check_call([python, "-m", "ensurepip", "--upgrade"])
        except Exception:
            pass  # pip may already be available

        errors = []
        warnings = []

        def pip_install(args, label, fatal=True):
            try:
                subprocess.check_call([python, "-m", "pip", "install", "--user", *args])
                return True
            except Exception as e:
                (errors if fatal else warnings).append(f"{label}: {e}")
                return False

        # Install/reinstall Pillow (force-reinstall to fix corrupted C extensions)
        pip_install(["--upgrade", "--force-reinstall", "Pillow"], "Pillow")
        # scipy (also a landlab dependency)
        pip_install(["--upgrade", "scipy"], "scipy")
        # landlab: the erosion engine (stream-power LEM + Incise-Flow)
        pip_install(["--upgrade", "landlab"], "landlab")

        # richdem: the fast GPL PriorityFloodFlowRouter. Optional — erosion falls back to the MIT
        # DepressionFinderAndRouter without it. Try the canonical package, then the prebuilt wheel.
        if not pip_install(["--upgrade", "richdem"], "richdem", fatal=False):
            warnings[-1] += " (trying py-richdem wheel)"
            if pip_install(["py-richdem"], "py-richdem", fatal=False):
                warnings.pop()  # the wheel worked; drop the richdem failure note

        if errors:
            self.report({"ERROR"}, f"Failed to install: {'; '.join(errors)}")
            return {"CANCELLED"}

        if warnings:
            self.report(
                {"WARNING"},
                "Core deps installed; richdem unavailable (erosion will use the slower MIT router). "
                "Restart Blender. See console for details.",
            )
            for w in warnings:
                print(f"[Project-R] Dependency warning: {w}")
            return {"FINISHED"}

        self.report(
            {"INFO"},
            "Dependencies installed (user site). Please restart Blender."
        )
        return {"FINISHED"}


from . import props as _props
from . import ui as _ui
from .operators import erode_ops as _erode_ops
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
    _erode_ops,
    _ui,
)


def _reload_modules_for_dev() -> None:
    # Helpful during development: Blender reloads addons without restarting,
    # but submodules can remain cached.
    # Reload pure helper modules first so operator modules pick up their changes.
    from . import erosion as _erosion
    importlib.reload(_erosion)
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
    bpy.types.Scene.projection_pasta_erosion = bpy.props.PointerProperty(
        type=_props.ProjectionPastaErosionSettings
    )


def unregister() -> None:
    if hasattr(bpy.types.Scene, "projection_pasta_erosion"):
        del bpy.types.Scene.projection_pasta_erosion
    if hasattr(bpy.types.Scene, "projection_pasta"):
        del bpy.types.Scene.projection_pasta

    for m in reversed(_MODULES):
        if hasattr(m, "unregister"):
            m.unregister()

    bpy.utils.unregister_class(PP_OT_install_dependencies)


