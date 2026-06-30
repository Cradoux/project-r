"""Dependency management for Project-R: detection + a non-blocking installer.

Single source of truth (the availability guards used to be duplicated across
``__init__`` and ``erosion`` and had already diverged). Everything the addon needs
beyond Blender's bundled ``numpy`` -- Pillow, scipy, landlab and the optional GPL
``richdem`` flow router -- is installed *into a private per-user folder* and added
to ``sys.path`` at runtime, so:

* installs never touch Blender's bundled site-packages (no ABI roulette with the
  numpy/scipy Blender ships), and
* the freshly installed packages are importable **in the same session** -- no
  Blender restart -- because we extend ``sys.path`` + ``site.addsitedir`` +
  ``importlib.invalidate_caches`` once the install finishes.

The installer runs as a *modal* operator driving an out-of-process ``pip`` via
``subprocess.Popen``, polled on a timer, so Blender's UI stays responsive (the old
installer blocked the main thread for minutes with no feedback).
"""
from __future__ import annotations

import importlib
import os
import site
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import bpy
from bpy.types import Operator


# --- What we install -------------------------------------------------------
# numpy is intentionally NOT listed: Blender bundles it, and installing a second
# copy risks an ABI clash. We pin numpy to Blender's version via a constraints
# file at install time so scipy/landlab resolve against the SAME numpy ABI.
CORE_PACKAGES = ("scipy", "Pillow", "landlab")
# The fast GPL PriorityFloodFlowRouter. Optional: erosion falls back to the slower
# MIT router without it. richdem has no cp311 source wheel on most platforms, so we
# try the prebuilt `py-richdem` wheel (never compile from source on the user).
RICHDEM_PACKAGE = "py-richdem"


# ---------------------------------------------------------------------------
# Private libs directory + import path
# ---------------------------------------------------------------------------

def libs_dir(create: bool = False) -> Path:
    """A guaranteed-writable, per-user folder that holds our installed packages.

    Prefers the extension API (Blender 4.2+ gives each extension its own user
    data dir); falls back to the legacy user-scripts ``modules`` area when the
    addon is loaded the old way (so this still works if installed via the legacy
    "Install from Disk" path)."""
    try:
        p = bpy.utils.extension_path_user(__package__, path="libs", create=create)
        return Path(p)
    except Exception:
        base = bpy.utils.user_resource("SCRIPTS", path="modules", create=create)
        p = Path(base) / "project_r_libs"
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p


def ensure_on_path() -> None:
    """Make our private libs dir importable. Appends (does not prepend) so Blender's
    bundled packages still win, which is what keeps numpy/scipy ABI-safe. Always
    invalidates the import finder caches: a just-finished install may have added
    files to a dir that was ALREADY on sys.path (from a prior session), in which
    case the new packages stay invisible unless the finder cache is refreshed."""
    d = str(libs_dir(create=False))
    if d and os.path.isdir(d) and d not in sys.path:
        sys.path.append(d)
        site.addsitedir(d)  # process any .pth and register the dir
    importlib.invalidate_caches()


# ---------------------------------------------------------------------------
# Cached availability checks (must be cheap: the panel asks every redraw)
# ---------------------------------------------------------------------------
# The old code re-imported scipy/PIL/landlab/richdem inside the panel draw() on
# every frame. We cache the booleans and only recompute after an install (or an
# explicit refresh), so draw() just reads a dict.
_avail: Dict[str, bool] = {}


def _probe_pillow() -> bool:
    try:
        from PIL import Image  # noqa: F401
        Image.new("RGB", (1, 1))  # exercise the C extension -- the common failure
        return True
    except Exception:
        return False


def _probe_scipy() -> bool:
    try:
        import scipy.ndimage  # noqa: F401
        return True
    except Exception:
        return False


def _probe_landlab() -> bool:
    try:
        import landlab  # noqa: F401
        from landlab.components import FastscapeEroder, LinearDiffuser  # noqa: F401
        return True
    except Exception:
        return False


def _probe_priorityflood() -> bool:
    try:
        from landlab.components import PriorityFloodFlowRouter  # noqa: F401
        import richdem  # noqa: F401
        return True
    except Exception:
        return False


_PROBES: Dict[str, Callable[[], bool]] = {
    "pillow": _probe_pillow,
    "scipy": _probe_scipy,
    "landlab": _probe_landlab,
    "priorityflood": _probe_priorityflood,
}


def _cached(name: str) -> bool:
    if name not in _avail:
        ensure_on_path()
        _avail[name] = _PROBES[name]()
    return _avail[name]


def refresh() -> None:
    """Drop cached availability so the next check re-probes (call after install)."""
    _avail.clear()


def pillow_available() -> bool:
    return _cached("pillow")


def scipy_available() -> bool:
    return _cached("scipy")


def landlab_available() -> bool:
    return _cached("landlab")


def priorityflood_available() -> bool:
    return _cached("priorityflood")


def missing_required() -> List[str]:
    """Human-readable names of the required packages that aren't importable."""
    missing = []
    if not pillow_available():
        missing.append("Pillow")
    if not scipy_available():
        missing.append("scipy")
    if not landlab_available():
        missing.append("landlab")
    return missing


# ---------------------------------------------------------------------------
# Non-blocking modal installer
# ---------------------------------------------------------------------------

def _numpy_constraint_file(target: Path) -> Optional[Path]:
    """Pin numpy to Blender's bundled version so pip resolves scipy/landlab
    against the same ABI instead of pulling a newer numpy that crashes on import.
    Returns None (and logs why) if the pin can't be written -- the caller treats
    that as fatal, because installing unconstrained is the ABI hazard."""
    try:
        import numpy  # Blender always ships this
        cfile = target / "_pr_constraints.txt"
        cfile.write_text(f"numpy=={numpy.__version__}\n", encoding="utf-8")
        return cfile
    except Exception as e:
        print(f"[Project-R] Could not write numpy version constraint: {e}")
        return None


def _strip_bundled_numpy(target: Path) -> None:
    """If pip dropped a numpy into our libs dir, remove it: Blender's bundled
    numpy must remain the one in use (we only append to sys.path, but deleting the
    duplicate avoids any chance of shadowing or version confusion)."""
    try:
        for child in target.iterdir():
            n = child.name.lower()
            if n in ("numpy", "numpy.libs") or n.startswith("numpy-") or n.startswith("numpy."):
                _rmtree(child)
    except Exception:
        pass


def _rmtree(p: Path) -> None:
    import shutil
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
    except Exception:
        pass


class PP_OT_install_dependencies(Operator):
    bl_idname = "pp.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = (
        "Install Pillow, scipy, landlab (and the optional richdem fast router) into "
        "a private Project-R folder. Runs in the background -- Blender stays usable -- "
        "and no restart is needed afterwards"
    )

    _timer = None
    _proc: Optional[subprocess.Popen] = None
    _logfile: Optional[Path] = None
    _stage = 0
    _stages: List[tuple] = []
    _t0 = 0.0
    _notes: List[str] = []
    _target: Optional[Path] = None
    _constrained = False  # True only once the numpy pin was actually written

    @classmethod
    def is_running(cls) -> bool:
        return cls._proc is not None

    def execute(self, context):
        if PP_OT_install_dependencies._proc is not None:
            self.report({"WARNING"}, "An install is already running")
            return {"CANCELLED"}

        target = libs_dir(create=True)
        PP_OT_install_dependencies._target = target
        self._logfile = target / "_pr_install.log"
        self._notes = []

        # The numpy pin is load-bearing for ABI safety: scipy/landlab must resolve
        # against Blender's bundled numpy, and _finish() strips any numpy pip drops
        # into the target. Installing UNCONSTRAINED could pull a mismatched numpy
        # (e.g. 2.x vs Blender's 1.x) that segfaults on `import scipy` -- so if the
        # pin can't be written, abort loudly instead of proceeding unsafely.
        cfile = _numpy_constraint_file(target)
        if cfile is None:
            self.report(
                {"ERROR"},
                "Could not write the numpy version constraint; aborting to avoid an "
                "ABI-unsafe install. See the system console for details.",
            )
            return {"CANCELLED"}
        self._constrained = True

        py = sys.executable
        common = ["--no-input", "--upgrade", "--target", str(target), "--constraint", str(cfile)]

        # Only (re)install what's actually MISSING. Re-running (e.g. to add the
        # optional router later) must not reinstall an already-loaded scipy/landlab
        # C extension -- on Windows that file is locked and pip would fail. Re-probe
        # first so this reflects any earlier install.
        ensure_on_path()
        refresh()
        core_missing: List[str] = []
        if not scipy_available():
            core_missing.append("scipy")
        if not pillow_available():
            core_missing.append("Pillow")
        if not landlab_available():
            core_missing.append("landlab")
        want_richdem = not priorityflood_available()

        if not core_missing and not want_richdem:
            self.report({"INFO"}, "All dependencies are already installed.")
            return {"CANCELLED"}

        # Stages: (label, args, fatal). ensurepip is best-effort; core is fatal;
        # richdem is optional (MIT fallback exists). richdem installs the prebuilt
        # wheel only -- never a from-source build. The numpy constraint applies to
        # every pip stage for ABI safety.
        stages: List[tuple] = [("Preparing pip", [py, "-m", "ensurepip", "--upgrade"], False)]
        if core_missing:
            stages.append((
                f"Installing core packages ({', '.join(core_missing)})",
                [py, "-m", "pip", "install", *common, *core_missing],
                True,
            ))
        if want_richdem:
            stages.append((
                "Installing fast erosion router (richdem, optional)",
                [py, "-m", "pip", "install", *common, "--only-binary", ":all:", RICHDEM_PACKAGE],
                False,
            ))
        self._stages = stages
        self._stage = 0
        self._t0 = time.time()
        self._logf = None

        if not self._start_stage():
            return {"CANCELLED"}

        context.window.cursor_set("DEFAULT")
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.4, window=context.window)
        wm.modal_handler_add(self)
        self.report({"INFO"}, "Installing dependencies in the background...")
        return {"RUNNING_MODAL"}

    def _close_log(self) -> None:
        f = getattr(self, "_logf", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
            self._logf = None

    def _start_stage(self) -> bool:
        label, args, _fatal = self._stages[self._stage]
        try:
            self._close_log()  # close the previous stage's handle before reopening
            logf = open(self._logfile, "a", encoding="utf-8", errors="replace")
            logf.write(f"\n=== {label} ===\n{' '.join(args)}\n")
            logf.flush()
            self._logf = logf
            PP_OT_install_dependencies._proc = subprocess.Popen(
                args, stdout=logf, stderr=subprocess.STDOUT, text=True
            )
            return True
        except Exception as e:
            self._notes.append(f"{label}: failed to start ({e})")
            PP_OT_install_dependencies._proc = None
            return False

    def modal(self, context, event):
        if event.type == "ESC":
            self._terminate()
            return self._finish(context, cancelled=True)

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        proc = PP_OT_install_dependencies._proc
        if proc is None:
            return self._finish(context, cancelled=True)

        rc = proc.poll()
        # Keep the header/sidebar repainting so the elapsed-time readout advances.
        for area in context.screen.areas:
            area.tag_redraw()

        if rc is None:
            elapsed = int(time.time() - self._t0)
            label = self._stages[self._stage][0]
            context.workspace.status_text_set(f"Project-R: {label}  ({elapsed}s)  [Esc to cancel]")
            return {"RUNNING_MODAL"}

        # Stage finished.
        label, _args, fatal = self._stages[self._stage]
        PP_OT_install_dependencies._proc = None
        self._close_log()  # release the log file handle (stays locked on Windows otherwise)
        if rc != 0:
            if fatal:
                self._notes.append(f"{label}: pip exited {rc}")
                return self._finish(context, cancelled=False, failed=True)
            else:
                self._notes.append(f"{label}: skipped (exit {rc}); using fallback")

        self._stage += 1
        if self._stage >= len(self._stages):
            return self._finish(context, cancelled=False, failed=False)

        if not self._start_stage():
            return self._finish(context, cancelled=False, failed=True)
        return {"RUNNING_MODAL"}

    def _terminate(self) -> None:
        proc = PP_OT_install_dependencies._proc
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        PP_OT_install_dependencies._proc = None
        self._close_log()

    def _finish(self, context, *, cancelled: bool, failed: bool = False):
        wm = context.window_manager
        self._close_log()
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        try:
            context.workspace.status_text_set(None)
        except Exception:
            pass

        # Only strip the target's numpy when the pin was actually applied: with the
        # constraint, the target numpy == Blender's, so removing the duplicate is
        # safe. Without it we must NOT strip (a mismatched numpy may be what
        # scipy/landlab were built against).
        target = PP_OT_install_dependencies._target
        if target is not None and self._constrained:
            _strip_bundled_numpy(target)

        # Make the freshly installed packages importable NOW -- no restart.
        ensure_on_path()
        refresh()

        if cancelled:
            self.report({"WARNING"}, "Dependency install cancelled")
            return {"CANCELLED"}

        still_missing = missing_required()
        if failed or still_missing:
            detail = "; ".join(self._notes) or "see log"
            log_hint = f" Log: {self._logfile}" if self._logfile else ""
            if still_missing:
                self.report(
                    {"ERROR"},
                    f"Install incomplete -- still missing: {', '.join(still_missing)}. "
                    f"{detail}.{log_hint}",
                )
            else:
                self.report({"ERROR"}, f"Install reported errors: {detail}.{log_hint}")
            return {"CANCELLED"}

        if not priorityflood_available():
            self.report(
                {"WARNING"},
                "Core dependencies installed. richdem (fast router) unavailable -- "
                "erosion will use the slower MIT router. No restart needed.",
            )
        else:
            self.report({"INFO"}, "All dependencies installed. No restart needed.")
        return {"FINISHED"}


_CLASSES = (PP_OT_install_dependencies,)


def register() -> None:
    ensure_on_path()
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
