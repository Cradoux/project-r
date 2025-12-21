from __future__ import annotations

# Vendored from https://github.com/hersfeldtn/projectionpasta
# We wrap the single-file module as a package so the Blender addon can import it
# via relative imports.
#
# IMPORTANT: We use lazy imports here because projectionpasta.py imports PIL at
# module level. If PIL isn't installed, we want the addon to still load so the
# user can click "Install Dependencies". The actual import happens when functions
# are first accessed.

__all__ = [
    "Proj_Image",
    "Proj_Array",
    "Find_index",
    "Rotate_to",
    "Rotate_from",
    "Quickvis",
    "pad",
    "def_opts",
    "posl",
]

# Lazy loading: these will be populated on first access
_module = None


def _ensure_loaded():
    global _module
    if _module is None:
        from . import projectionpasta as _pp  # type: ignore
        _module = _pp
    return _module


def __getattr__(name: str):
    """Lazy load projectionpasta symbols on first access."""
    if name in __all__:
        mod = _ensure_loaded()
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


