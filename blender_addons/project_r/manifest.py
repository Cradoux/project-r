from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Bump whenever the on-disk manifest.json schema changes in a way readers must
# know about. 0.2 records per-section size_info + erosion elevation_info that 0.1
# manifests lacked. read_manifest() stamps/migrates older manifests on load.
MANIFEST_VERSION = "0.2"


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def default_manifest(
    *,
    global_size: Tuple[int, int],
    hammer_full_size: Tuple[int, int],
    crop_margin_px: int,
    square_crop: bool,
    blend_feather_px: int,
) -> Dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "project": {},
        "global": {
            "projection": "Equirectangular",
            "size": [int(global_size[0]), int(global_size[1])],
            "layers": [],
        },
        "defaults": {
            "hammer_full_size": [int(hammer_full_size[0]), int(hammer_full_size[1])],
            "crop_margin_px": int(crop_margin_px),
            "square_crop": bool(square_crop),
            "blend_feather_px": int(blend_feather_px),
        },
        "sections": [],
    }


def read_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _migrate(data)


def _migrate(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise an on-disk manifest to the current shape in memory.

    Kept deliberately tolerant: the schema has only ever *added* optional fields
    (size_info, elevation_info, overlay, ...), so migration is mostly stamping the
    version and guaranteeing the top-level containers exist. This is the single
    place to add real field migrations if the schema ever changes incompatibly.
    """
    if not isinstance(data, dict):
        return data
    ver = str(data.get("version", "0.1"))
    data.setdefault("global", {})
    data.setdefault("defaults", {})
    data.setdefault("sections", [])
    if ver != MANIFEST_VERSION:
        # No structural rewrite needed from 0.1 -> 0.2 (additive only); stamp it so
        # the file converges to current on the next write.
        data["version"] = MANIFEST_VERSION
    return data


def section_choices(project_root: Path) -> List[Tuple[str, str]]:
    """(section_id, display_name) for every section in the project's manifest.

    Used to populate the erosion target dropdown. Tolerant of a missing or
    corrupt manifest (returns an empty list rather than raising), because it is
    called from a property callback that must never throw.
    """
    try:
        mp = project_root / "manifest.json"
        if not mp.exists():
            return []
        data = read_manifest(mp)
    except Exception:
        return []
    out: List[Tuple[str, str]] = []
    for sec in data.get("sections", []) or []:
        sid = str(sec.get("id", "")).strip()
        if not sid:
            continue
        out.append((sid, str(sec.get("name", sid))))
    return out


def write_manifest(path: Path, data: Dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def init_project_folders(project_root: Path) -> None:
    (project_root / "source").mkdir(parents=True, exist_ok=True)
    (project_root / "sections").mkdir(parents=True, exist_ok=True)
    (project_root / "processed").mkdir(parents=True, exist_ok=True)
    (project_root / "reassembled").mkdir(parents=True, exist_ok=True)


def add_layer(
    manifest: Dict[str, Any],
    *,
    layer_id: str,
    path: str,
    datatype: str,
    file_format: str,
    interp: str,
) -> None:
    layers: List[Dict[str, Any]] = manifest["global"]["layers"]
    layers.append(
        {
            "id": layer_id,
            "path": path,
            "datatype": datatype,  # continuous | categorical
            "format": file_format,  # OPEN_EXR | PNG16 | PNG | JPG | TIF16 | ...
            "interp": interp,  # linear | nearest
        }
    )


def relativize_path(project_root: Path, path: str | Path) -> str:
    """Express `path` relative to `project_root` (posix-style) when it lives
    inside the project, so manifests stay portable across machines and user
    accounts. Falls back to the absolute string for paths outside the project.
    """
    p = Path(path)
    try:
        abs_p = p if p.is_absolute() else (project_root / p)
        return abs_p.resolve().relative_to(project_root.resolve()).as_posix()
    except (ValueError, OSError):
        return str(p)


def resolve_source_path(
    project_root: Path, stored: Optional[str | Path]
) -> Optional[Path]:
    """Resolve a manifest-stored source path to a file that actually exists,
    tolerating projects authored elsewhere (stale absolute paths from another
    machine or user account). Tries, in order: the path as stored (absolute, or
    relative to the project root), then ``<root>/source/<name>``, then
    ``<root>/<name>``. Returns the first existing candidate, or None.
    """
    if not stored:
        return None
    p = Path(stored)
    candidates: List[Path] = [p if p.is_absolute() else (project_root / p)]
    candidates.append(project_root / "source" / p.name)
    candidates.append(project_root / p.name)
    for c in candidates:
        try:
            if c.exists():
                return c
        except OSError:
            continue
    return None


