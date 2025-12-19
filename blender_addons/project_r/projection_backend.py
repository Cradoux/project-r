from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np

from . import imaging

Interp = Literal["nearest", "linear"]


@dataclass(frozen=True)
class ProjectionParams:
    center_lon_deg: float
    center_lat_deg: float
    rot_deg: float = 0.0


class ProjectionBackendError(RuntimeError):
    pass


def _import_projectionpasta():
    # vendored under blender_addons/projection_pasta/vendor/projectionpasta
    try:
        from .vendor import projectionpasta  # type: ignore

        return projectionpasta
    except Exception as e:  # noqa: BLE001 - surface clean error to Blender UI
        raise ProjectionBackendError(
            "projectionpasta import failed (vendor missing or bad install)."
        ) from e


def _opts_for(dst_size: Tuple[int, int]) -> dict:
    projectionpasta = _import_projectionpasta()
    opts = dict(projectionpasta.def_opts)
    # Force output size (width,height) via force_scale (projectionpasta expects this).
    opts["force_scale"] = (int(dst_size[0]), int(dst_size[1]))
    # Avoid SciPy dependency; we'll do nearest or our own bilinear sampling.
    opts["interp_type"] = "none"
    opts["proj_direction"] = "backward"
    opts["truncate_in"] = True
    opts["truncate_out"] = True
    opts["use_sym"] = True
    opts["avoid_seam"] = True
    return opts


def _to_file_format(
    path: Path,
    *,
    treat_as_color: bool,
) -> Tuple[imaging.ImageFormat, str | None]:
    suf = path.suffix.lower()
    if suf == ".exr":
        return ("OPEN_EXR", "32")
    if suf == ".png":
        # Preserve typical bit-depth expectations:
        # - color maps: 8-bit (sRGB encoded)
        # - non-color (masks/height-ish): 16-bit
        return ("PNG", "8" if treat_as_color else "16")
    if suf in (".jpg", ".jpeg"):
        return ("JPEG", None)
    # default: try PNG
    return ("PNG", "8" if treat_as_color else "16")


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4).astype(np.float32)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1.0 + a) * (x ** (1.0 / 2.4)) - a).astype(np.float32)


def _sample_nearest(data_in: np.ndarray, xpix: np.ndarray, ypix: np.ndarray) -> np.ndarray:
    xi = np.clip(np.rint(xpix), 0, data_in.shape[1] - 1).astype(np.int64)
    yi = np.clip(np.rint(ypix), 0, data_in.shape[0] - 1).astype(np.int64)
    return data_in[yi, xi]


def _sample_bilinear(data_in: np.ndarray, xpix: np.ndarray, ypix: np.ndarray) -> np.ndarray:
    x0 = np.floor(xpix).astype(np.int64)
    y0 = np.floor(ypix).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, data_in.shape[1] - 1)
    x1 = np.clip(x1, 0, data_in.shape[1] - 1)
    y0 = np.clip(y0, 0, data_in.shape[0] - 1)
    y1 = np.clip(y1, 0, data_in.shape[0] - 1)

    wx = (xpix - x0).astype(np.float32)
    wy = (ypix - y0).astype(np.float32)

    # Expand dims for broadcasting into channels if needed.
    if data_in.ndim == 3:
        wx = wx[..., None]
        wy = wy[..., None]

    c00 = data_in[y0, x0].astype(np.float32)
    c10 = data_in[y0, x1].astype(np.float32)
    c01 = data_in[y1, x0].astype(np.float32)
    c11 = data_in[y1, x1].astype(np.float32)

    c0 = c00 * (1.0 - wx) + c10 * wx
    c1 = c01 * (1.0 - wx) + c11 * wx
    return c0 * (1.0 - wy) + c1 * wy


def _reproject_array(
    *,
    data_in: np.ndarray,
    proj_in: str,
    proj_out: str,
    dst_size: Tuple[int, int],
    aspect_in_deg: Tuple[float, float, float],
    aspect_out_deg: Tuple[float, float, float],
    interp: Interp,
    treat_as_color: bool,
) -> np.ndarray:
    projectionpasta = _import_projectionpasta()
    opts = _opts_for(dst_size)

    # For color images: do interpolation in linear, but keep file encoding sRGB.
    if treat_as_color and data_in.ndim == 3 and data_in.shape[2] >= 3:
        data_in = data_in.copy()
        data_in[..., 0:3] = _srgb_to_linear(data_in[..., 0:3])

    # Construct coordinate grids the same way projectionpasta does.
    h0, w0 = data_in.shape[0], data_in.shape[1]
    x_in = np.linspace(-1, 1, w0, False) + 1 / w0
    y_in = np.linspace(1, -1, h0, False) - 1 / h0

    # Output size is forced by opts.force_scale (width,height)
    w1, h1 = int(dst_size[0]), int(dst_size[1])
    x_out = np.linspace(-1, 1, w1, False) + 1 / w1
    y_out = np.linspace(1, -1, h1, False) - 1 / h1
    x_out, y_out = np.meshgrid(x_out, y_out)

    # Compute backward index from output pixels to input projection coords.
    x_ind, y_ind = projectionpasta.Find_index(
        x_out,
        y_out,
        proj_out,
        proj_in,
        aspect_out_deg,
        aspect_in_deg,
        opts,
        deg=True,
        get_lon=False,
    )

    # Pad input edges to avoid seams (same as projectionpasta; padding size fixed at 2).
    if opts["avoid_seam"] and opts["truncate_in"]:
        x_in1, y_in1 = np.meshgrid(x_in, y_in)
        vis_in, far = projectionpasta.Quickvis(
            proj_in, x_in1, y_in1, None, None, opts={**opts, "in": True}, get_far=True
        )
        # Determine padding type for equirect/hammers etc. For our v0.1 we only need
        # robust wrap for rectangular globals. projectionpasta uses `wrapl`; we fall back.
        ptype = getattr(projectionpasta, "wrapl", {}).get(proj_in, "rect")
        data_in, x_in, y_in = projectionpasta.pad(ptype, proj_in, vis_in, data_in, x_in, y_in, pd=2, opts=opts, far=far)

        # Offset accounts for padding width/height relative to original.
        x_off = round((data_in.shape[1] - w0) / 2)
        y_off = round((data_in.shape[0] - h0) / 2)
    else:
        x_off = 0
        y_off = 0

    # Convert projection-space indices (-1..1) to pixel coordinates on padded input.
    xpix = (x_ind + 1.0) / 2.0 * data_in.shape[1] - 0.5 + x_off
    ypix = (1.0 - y_ind) / 2.0 * data_in.shape[0] - 0.5 + y_off

    if interp == "nearest":
        out = _sample_nearest(data_in, xpix, ypix)
    else:
        out = _sample_bilinear(data_in, xpix, ypix)

    if treat_as_color and out.ndim == 3 and out.shape[2] >= 3:
        out = out.astype(np.float32, copy=False)
        out[..., 0:3] = _linear_to_srgb(out[..., 0:3])

    # Preserve dtype-ish: masks might want integer-ish results.
    return out


def project_equirect_array_to_hammer(
    *,
    data_in: np.ndarray,
    dst_size: Tuple[int, int],
    params: ProjectionParams,
    interp: Interp,
    treat_as_color: bool = False,
) -> np.ndarray:
    return _reproject_array(
        data_in=data_in,
        proj_in="Equirectangular",
        proj_out="Hammer",
        dst_size=dst_size,
        aspect_in_deg=(0.0, 0.0, 0.0),
        aspect_out_deg=(params.center_lon_deg, params.center_lat_deg, params.rot_deg),
        interp=interp,
        treat_as_color=treat_as_color,
    )


def project_hammer_array_to_equirect(
    *,
    data_in: np.ndarray,
    dst_size: Tuple[int, int],
    params: ProjectionParams,
    interp: Interp,
    treat_as_color: bool = False,
) -> np.ndarray:
    return _reproject_array(
        data_in=data_in,
        proj_in="Hammer",
        proj_out="Equirectangular",
        dst_size=dst_size,
        aspect_in_deg=(params.center_lon_deg, params.center_lat_deg, params.rot_deg),
        aspect_out_deg=(0.0, 0.0, 0.0),
        interp=interp,
        treat_as_color=treat_as_color,
    )


def project_equirect_to_hammer(
    *,
    src_path: Path,
    dst_path: Path,
    dst_size: Tuple[int, int],
    params: ProjectionParams,
    interp: Interp,
    treat_as_color: bool = False,
) -> None:
    buf = imaging.load_image(src_path)
    data_in = buf.pixels
    out = _reproject_array(
        data_in=data_in,
        proj_in="Equirectangular",
        proj_out="Hammer",
        dst_size=dst_size,
        aspect_in_deg=(0.0, 0.0, 0.0),
        aspect_out_deg=(params.center_lon_deg, params.center_lat_deg, params.rot_deg),
        interp=interp,
        treat_as_color=treat_as_color,
    )

    out_buf = imaging.ImageBuffer(
        width=int(dst_size[0]),
        height=int(dst_size[1]),
        channels=buf.channels,
        pixels=out.astype(np.float32),
    )
    fmt, depth = _to_file_format(dst_path, treat_as_color=treat_as_color)
    imaging.save_image(out_buf, dst_path, fmt, color_depth=depth)


def project_hammer_to_equirect(
    *,
    src_path: Path,
    dst_path: Path,
    dst_size: Tuple[int, int],
    params: ProjectionParams,
    interp: Interp,
    treat_as_color: bool = False,
) -> None:
    buf = imaging.load_image(src_path)
    data_in = buf.pixels
    out = _reproject_array(
        data_in=data_in,
        proj_in="Hammer",
        proj_out="Equirectangular",
        dst_size=dst_size,
        aspect_in_deg=(params.center_lon_deg, params.center_lat_deg, params.rot_deg),
        aspect_out_deg=(0.0, 0.0, 0.0),
        interp=interp,
        treat_as_color=treat_as_color,
    )

    out_buf = imaging.ImageBuffer(
        width=int(dst_size[0]),
        height=int(dst_size[1]),
        channels=buf.channels,
        pixels=out.astype(np.float32),
    )
    fmt, depth = _to_file_format(dst_path, treat_as_color=treat_as_color)
    imaging.save_image(out_buf, dst_path, fmt, color_depth=depth)


