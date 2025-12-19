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
    img = bpy.data.images.load(str(path), check_existing=False)
    try:
        w, h = img.size
        c = img.channels
        arr = np.array(img.pixels[:], dtype=np.float32)
        arr = arr.reshape((h, w, c))
        return ImageBuffer(width=w, height=h, channels=c, pixels=arr)
    finally:
        # Avoid accumulating data blocks
        bpy.data.images.remove(img)


def save_image(
    buf: ImageBuffer,
    path: Path,
    file_format: ImageFormat,
    *,
    color_depth: Literal["8", "16", "32"] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

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


