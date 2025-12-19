from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import bpy
import numpy as np

ImageFormat = Literal["OPEN_EXR", "PNG", "JPEG"]


@dataclass(frozen=True)
class ImageBuffer:
    width: int
    height: int
    channels: int
    pixels: np.ndarray  # float32, shape (H, W, C)


def load_image(path: Path) -> ImageBuffer:
    # For PNG/JPG we use Pillow to avoid Blender color-management altering pixel values.
    suf = path.suffix.lower()
    if suf in (".png", ".jpg", ".jpeg"):
        try:
            from PIL import Image as PILImage  # type: ignore
        except Exception:
            PILImage = None

        if PILImage is not None:
            im = PILImage.open(path)
            im.load()

            # Normalize to float32 [0..1], top-to-bottom.
            arr_u = np.asarray(im)
            if arr_u.ndim == 2:
                arr_u = arr_u[:, :, None]

            if arr_u.dtype == np.uint16:
                arr = arr_u.astype(np.float32) / 65535.0
            else:
                arr = arr_u.astype(np.float32) / 255.0

            return ImageBuffer(
                width=int(im.size[0]),
                height=int(im.size[1]),
                channels=int(arr.shape[2]),
                pixels=arr,
            )

    # Fallback: Blender loader (EXR, or if PIL unavailable)
    img = bpy.data.images.load(str(path), check_existing=False)
    try:
        w, h = img.size
        c = img.channels
        arr = np.array(img.pixels[:], dtype=np.float32)
        # Blender's Image.pixels is stored bottom-to-top.
        arr = arr.reshape((h, w, c))
        arr = np.flip(arr, axis=0)
        return ImageBuffer(width=w, height=h, channels=c, pixels=arr)
    finally:
        bpy.data.images.remove(img)


def save_image(
    buf: ImageBuffer,
    path: Path,
    file_format: ImageFormat,
    *,
    color_depth: Literal["8", "16", "32"] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # For PNG/JPG, prefer Pillow to preserve exact encoding/bit-depth without Blender color management.
    if file_format in ("PNG", "JPEG"):
        try:
            from PIL import Image as PILImage  # type: ignore
        except Exception:
            PILImage = None

        if PILImage is not None:
            px = buf.pixels.astype(np.float32)
            if px.ndim == 2:
                px = px[:, :, None]

            # Convert to desired channel count for writing.
            c = px.shape[2]
            if c == 1:
                out = np.clip(px[:, :, 0], 0.0, 1.0)
                if file_format == "PNG" and color_depth == "16":
                    arr_u = (out * 65535.0 + 0.5).astype(np.uint16)
                    im = PILImage.fromarray(arr_u, mode="I;16")
                else:
                    arr_u = (out * 255.0 + 0.5).astype(np.uint8)
                    im = PILImage.fromarray(arr_u, mode="L")
            else:
                # Write RGB or RGBA
                rgb = np.clip(px[:, :, :3], 0.0, 1.0)
                if file_format == "PNG" and color_depth == "16":
                    arr_u = (rgb * 65535.0 + 0.5).astype(np.uint16)
                    # Pillow doesn't have a great native "RGB;16" mode across all versions;
                    # store as 8-bit for RGB if needed in practice.
                    # If user asks for true 16-bit RGB later, we can switch to TIFF.
                    arr_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
                    im = PILImage.fromarray(arr_u8, mode="RGB")
                else:
                    arr_u = (rgb * 255.0 + 0.5).astype(np.uint8)
                    im = PILImage.fromarray(arr_u, mode="RGB")

                if c >= 4 and file_format == "PNG":
                    a = np.clip(px[:, :, 3], 0.0, 1.0)
                    a_u = (a * 255.0 + 0.5).astype(np.uint8)
                    im.putalpha(PILImage.fromarray(a_u, mode="L"))

            im.save(path)
            return

    # Blender stores `Image.pixels` as RGBA (4 floats per pixel) even when an
    # image is conceptually single-channel. Always write RGBA to avoid
    # `foreach_set` length mismatches.
    img = bpy.data.images.new(
        name="PP_Temp",
        width=buf.width,
        height=buf.height,
        alpha=True,
        float_buffer=True,
    )
    try:
        px = buf.pixels.astype(np.float32)
        if px.ndim == 2:
            px = px[:, :, None]
        if px.shape[2] == 4:
            rgba = px
        elif px.shape[2] == 3:
            a = np.ones((buf.height, buf.width, 1), dtype=np.float32)
            rgba = np.concatenate([px, a], axis=2)
        elif px.shape[2] == 2:
            # Treat as RG, set B=0, A=1
            b = np.zeros((buf.height, buf.width, 1), dtype=np.float32)
            a = np.ones((buf.height, buf.width, 1), dtype=np.float32)
            rgba = np.concatenate([px, b, a], axis=2)
        elif px.shape[2] == 1:
            a = np.ones((buf.height, buf.width, 1), dtype=np.float32)
            rgba = np.concatenate([px, px, px, a], axis=2)
        else:
            # Fallback: take first 3 channels, pad alpha=1
            rgb = px[:, :, :3] if px.shape[2] >= 3 else np.zeros((buf.height, buf.width, 3), dtype=np.float32)
            a = np.ones((buf.height, buf.width, 1), dtype=np.float32)
            rgba = np.concatenate([rgb, a], axis=2)

        # Convert from our normalized top-to-bottom back to Blender's bottom-to-top.
        rgba = np.flip(rgba, axis=0)

        flat = rgba.reshape((-1,))
        img.pixels.foreach_set(flat)
        img.filepath_raw = str(path)
        img.file_format = file_format

        if color_depth is not None:
            scene = bpy.context.scene
            old_depth = scene.render.image_settings.color_depth
            scene.render.image_settings.color_depth = color_depth
            try:
                img.save()
            finally:
                scene.render.image_settings.color_depth = old_depth
        else:
            img.save()
    finally:
        bpy.data.images.remove(img)


def crop(buf: ImageBuffer, x: int, y: int, w: int, h: int) -> ImageBuffer:
    x2 = max(x, 0)
    y2 = max(y, 0)
    w2 = max(1, min(w, buf.width - x2))
    h2 = max(1, min(h, buf.height - y2))
    return ImageBuffer(
        width=w2,
        height=h2,
        channels=buf.channels,
        pixels=buf.pixels[y2 : y2 + h2, x2 : x2 + w2, :].copy(),
    )


def paste_into(
    *,
    dst_size: Tuple[int, int],
    dst_channels: int,
    src: ImageBuffer,
    rect_xywh: Tuple[int, int, int, int],
) -> ImageBuffer:
    dst_w, dst_h = dst_size
    x, y, w, h = rect_xywh
    dst = np.zeros((dst_h, dst_w, dst_channels), dtype=np.float32)
    # If channels mismatch, copy min(C)
    c = min(dst_channels, src.channels)
    dst[y : y + h, x : x + w, :c] = src.pixels[:h, :w, :c]
    if dst_channels == 4 and src.channels < 4:
        dst[..., 3] = 1.0
    return ImageBuffer(width=dst_w, height=dst_h, channels=dst_channels, pixels=dst)


def make_feather_weight_mask(width: int, height: int, feather_px: int) -> ImageBuffer:
    feather_px = max(0, int(feather_px))
    if feather_px == 0:
        m = np.ones((height, width, 1), dtype=np.float32)
        return ImageBuffer(width=width, height=height, channels=1, pixels=m)

    yy, xx = np.mgrid[0:height, 0:width]
    d_left = xx
    d_right = (width - 1) - xx
    d_top = yy
    d_bot = (height - 1) - yy
    d = np.minimum(np.minimum(d_left, d_right), np.minimum(d_top, d_bot)).astype(np.float32)
    w = np.clip(d / float(feather_px), 0.0, 1.0)
    return ImageBuffer(width=width, height=height, channels=1, pixels=w[..., None])


def load_or_create_overlay_rgba_u8(path: Path, size: Tuple[int, int]) -> "np.ndarray":
    """
    Load an RGBA overlay PNG (uint8) or create a transparent one if missing.
    Array is returned top-to-bottom with shape (H, W, 4).
    """
    w, h = int(size[0]), int(size[1])
    try:
        from PIL import Image as PILImage  # type: ignore
    except Exception as e:
        raise RuntimeError("Pillow is required for overlay rendering") from e

    if path.exists():
        im = PILImage.open(path).convert("RGBA")
        if im.size != (w, h):
            # Size mismatch: recreate to avoid misaligned UV mapping.
            im = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
    else:
        im = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))

    arr = np.asarray(im, dtype=np.uint8)
    return arr


def save_overlay_rgba_u8(path: Path, rgba: "np.ndarray") -> None:
    """
    Save an RGBA uint8 array (H,W,4) as PNG.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image as PILImage  # type: ignore
    except Exception as e:
        raise RuntimeError("Pillow is required for overlay rendering") from e

    im = PILImage.fromarray(rgba, mode="RGBA")
    im.save(path)


def paint_uv_triangles_on_overlay(
    overlay_rgba_u8: "np.ndarray",
    *,
    triangles_uv: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    color_rgba_u8: tuple[int, int, int, int],
) -> "np.ndarray":
    """
    Paint UV triangles onto an RGBA overlay.

    - overlay is top-to-bottom (image coordinates) shape (H,W,4), uint8.
    - UVs are in [0,1] with v=0 at bottom (Blender UV convention).
    - Handles U seam by drawing wrapped triangles when span crosses the seam.
    """
    try:
        from PIL import Image as PILImage  # type: ignore
        from PIL import ImageDraw  # type: ignore
    except Exception as e:
        raise RuntimeError("Pillow is required for overlay rendering") from e

    h, w = int(overlay_rgba_u8.shape[0]), int(overlay_rgba_u8.shape[1])
    base = PILImage.fromarray(overlay_rgba_u8, mode="RGBA")
    draw = ImageDraw.Draw(base, "RGBA")

    def uv_to_xy(u: float, v: float) -> tuple[float, float]:
        # PIL image coords: origin top-left; Blender UV v=0 is bottom.
        x = u * w
        y = (1.0 - v) * h
        return (x, y)

    for tri in triangles_uv:
        (u0, v0), (u1, v1), (u2, v2) = tri
        us = [u0, u1, u2]
        span = max(us) - min(us)

        if span > 0.5:
            # Draw both shifted variants to cover across the seam.
            # Variant A: shift low-u up by +1 (draws on right side)
            tri_a = []
            for (u, v) in tri:
                if u < 0.5:
                    u += 1.0
                tri_a.append(uv_to_xy(u, v))
            draw.polygon(tri_a, fill=color_rgba_u8)

            # Variant B: shift high-u down by -1 (draws on left side)
            tri_b = []
            for (u, v) in tri:
                if u > 0.5:
                    u -= 1.0
                tri_b.append(uv_to_xy(u, v))
            draw.polygon(tri_b, fill=color_rgba_u8)
        else:
            pts = [uv_to_xy(u0, v0), uv_to_xy(u1, v1), uv_to_xy(u2, v2)]
            draw.polygon(pts, fill=color_rgba_u8)

    out = np.asarray(base, dtype=np.uint8)
    return out


def paint_uv_circles_on_overlay(
    overlay_rgba_u8: "np.ndarray",
    *,
    centers_uv: list[tuple[float, float]],
    radius_px: int,
    color_rgba_u8: tuple[int, int, int, int],
) -> "np.ndarray":
    """
    Paint circles (markers) centered at UVs onto an RGBA overlay.

    - overlay is top-to-bottom (image coordinates) shape (H,W,4), uint8.
    - UVs are in [0,1] with v=0 at bottom (Blender UV convention).
    - Handles U seam by duplicating circles that cross the seam.
    """
    try:
        from PIL import Image as PILImage  # type: ignore
        from PIL import ImageDraw  # type: ignore
    except Exception as e:
        raise RuntimeError("Pillow is required for overlay rendering") from e

    radius_px = max(1, int(radius_px))
    h, w = int(overlay_rgba_u8.shape[0]), int(overlay_rgba_u8.shape[1])
    base = PILImage.fromarray(overlay_rgba_u8, mode="RGBA")
    draw = ImageDraw.Draw(base, "RGBA")

    def draw_one(u: float, v: float) -> None:
        x = u * w
        y = (1.0 - v) * h
        r = radius_px
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color_rgba_u8)

    seam_u = radius_px / float(w)
    for (u, v) in centers_uv:
        u = float(u) % 1.0
        v = float(v)
        draw_one(u, v)
        # Duplicate across seam if circle would wrap.
        if u < seam_u:
            draw_one(u + 1.0, v)
        elif u > 1.0 - seam_u:
            draw_one(u - 1.0, v)

    out = np.asarray(base, dtype=np.uint8)
    return out


