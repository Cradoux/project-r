from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MANIFEST_VERSION = "0.1"


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
        return json.load(f)


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


