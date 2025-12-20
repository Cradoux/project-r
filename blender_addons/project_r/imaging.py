from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import bpy
import numpy as np

# SciPy is optional; we provide a fallback for distance transforms.
try:
    from scipy.ndimage import distance_transform_edt as _scipy_edt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    _scipy_edt = None  # type: ignore

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


def create_solid_mask_equirect(
    *,
    width: int,
    height: int,
    triangles_uv: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
) -> "np.ndarray":
    """
    Create a solid grayscale mask in equirectangular space from UV triangles.

    - Returns (H, W) float32 array with 1.0 where triangles cover, 0.0 elsewhere.
    - UVs are in [0,1] with v=0 at bottom (Blender UV convention).
    - Handles U seam by drawing wrapped triangles.
    """
    try:
        from PIL import Image as PILImage  # type: ignore
        from PIL import ImageDraw  # type: ignore
    except Exception as e:
        raise RuntimeError("Pillow is required for mask generation") from e

    h, w = int(height), int(width)
    # Create grayscale image
    mask_img = PILImage.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)

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
            # Variant A: shift low-u up by +1
            tri_a = []
            for (u, v) in tri:
                if u < 0.5:
                    u += 1.0
                tri_a.append(uv_to_xy(u, v))
            draw.polygon(tri_a, fill=255)

            # Variant B: shift high-u down by -1
            tri_b = []
            for (u, v) in tri:
                if u > 0.5:
                    u -= 1.0
                tri_b.append(uv_to_xy(u, v))
            draw.polygon(tri_b, fill=255)
        else:
            pts = [uv_to_xy(u0, v0), uv_to_xy(u1, v1), uv_to_xy(u2, v2)]
            draw.polygon(pts, fill=255)

    # Convert to float32 normalized
    out = np.asarray(mask_img, dtype=np.float32) / 255.0
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


# ---------------------------------------------------------------------------
# Distance Transform & Nearest-Valid Extension
# ---------------------------------------------------------------------------

def _distance_transform_edt_fallback(mask: np.ndarray, return_indices: bool = False):
    """
    Pure-numpy fallback for distance_transform_edt.
    Uses a simple brute-force O(n^2) approach - slow but works without scipy.
    For large images, scipy is strongly recommended.
    """
    h, w = mask.shape[:2]
    dist = np.full((h, w), np.inf, dtype=np.float32)
    indices = np.zeros((2, h, w), dtype=np.int64) if return_indices else None

    # Find all valid (mask == 1) pixel coordinates
    valid_ys, valid_xs = np.where(mask > 0.5)
    if len(valid_ys) == 0:
        # No valid pixels; return zeros
        if return_indices:
            return np.zeros((h, w), dtype=np.float32), np.zeros((2, h, w), dtype=np.int64)
        return np.zeros((h, w), dtype=np.float32)

    # For each pixel, find distance to nearest valid pixel
    # This is O(n*m) where n=total pixels, m=valid pixels
    yy, xx = np.mgrid[0:h, 0:w]

    # Process in chunks to limit memory for large images
    chunk_size = 1000
    for i in range(0, len(valid_ys), chunk_size):
        vy = valid_ys[i : i + chunk_size, None, None]
        vx = valid_xs[i : i + chunk_size, None, None]
        d2 = (yy[None, :, :] - vy) ** 2 + (xx[None, :, :] - vx) ** 2
        min_idx = np.argmin(d2, axis=0)
        min_d2 = np.take_along_axis(d2, min_idx[None, :, :], axis=0)[0]
        update = min_d2 < dist**2
        dist = np.where(update, np.sqrt(min_d2), dist)
        if return_indices:
            indices[0] = np.where(update, valid_ys[i + min_idx], indices[0])
            indices[1] = np.where(update, valid_xs[i + min_idx], indices[1])

    if return_indices:
        return dist.astype(np.float32), indices
    return dist.astype(np.float32)


def distance_transform_edt(mask: np.ndarray, return_indices: bool = False):
    """
    Compute Euclidean distance transform. Uses scipy if available, else fallback.
    
    Args:
        mask: Binary mask (H, W), nonzero = valid pixels
        return_indices: If True, also return indices of nearest valid pixel
    
    Returns:
        dist: Distance to nearest valid pixel (0 for valid pixels themselves)
        indices: (optional) (2, H, W) array of [y, x] indices of nearest valid pixel
    """
    mask_2d = mask.squeeze() if mask.ndim > 2 else mask
    # Invert: we want distance to nearest valid (1) pixel, not background
    # scipy measures from 0s to 1s, so we pass the inverted mask
    bg_mask = (mask_2d < 0.5).astype(np.uint8)

    if HAS_SCIPY and _scipy_edt is not None:
        if return_indices:
            dist, indices = _scipy_edt(bg_mask, return_indices=True)
            return dist.astype(np.float32), indices
        else:
            dist = _scipy_edt(bg_mask)
            return dist.astype(np.float32)
    else:
        # Fallback: compute distance from each invalid pixel to nearest valid
        # We invert the logic: pass the valid mask
        valid_mask = (mask_2d >= 0.5).astype(np.uint8)
        return _distance_transform_edt_fallback(valid_mask, return_indices=return_indices)


def generate_effective_mask(
    coverage: np.ndarray,
    feather_px: int,
) -> np.ndarray:
    """
    Generate an effective blend mask from a binary coverage mask.
    
    This is the SINGLE mask used for reassembly blending.
    
    - 1.0 in the interior (well inside selection)
    - Fades from 1.0 → 0.0 approaching selection boundary
    - Additionally fades at image edges
    - 0.0 outside selection
    
    Args:
        coverage: Binary mask (H, W) or (H, W, 1), 1=inside selection, 0=outside
        feather_px: Feather width in pixels
    
    Returns:
        (H, W) float32 array with values 0..1
    """
    feather_px = max(1, int(feather_px))
    h, w = coverage.shape[:2]
    cov_2d = coverage.squeeze() if coverage.ndim > 2 else coverage
    inside = (cov_2d >= 0.5).astype(np.uint8)
    
    inside_count = int(np.sum(inside))
    print(f"[Project-R] generate_effective_mask: {inside_count} inside pixels, feather={feather_px}px, scipy={HAS_SCIPY}")

    # If no inside pixels, return zeros
    if inside_count == 0:
        return np.zeros((h, w), dtype=np.float32)

    # Distance from INSIDE pixels to nearest OUTSIDE pixel (boundary)
    # scipy.distance_transform_edt(mask) computes, for each FOREGROUND (non-zero) pixel,
    # the distance to the nearest BACKGROUND (zero) pixel.
    # So we pass `inside` (1=inside, 0=outside): inside pixels get distance to nearest outside.
    outside_count = int(np.sum(1 - inside))
    
    if outside_count == 0:
        # No outside pixels in crop - selection fills entire crop area
        # Fall back to very large distance (will be limited by edge distance)
        print(f"[Project-R] No outside pixels in crop - using edge-only feathering")
        dist_to_boundary = np.full((h, w), float(max(h, w)), dtype=np.float32)
    elif HAS_SCIPY and _scipy_edt is not None:
        # scipy.distance_transform_edt on `inside`:
        # - For inside=1 (foreground): distance to nearest inside=0 (background/outside)
        # - For inside=0 (background): 0
        dist_to_boundary = _scipy_edt(inside).astype(np.float32)
    else:
        # Fallback: pass inverted mask so it finds distance TO outside pixels
        dist_to_boundary = _distance_transform_edt_fallback(1 - inside, return_indices=False)

    print(f"[Project-R] dist_to_boundary: min={np.min(dist_to_boundary):.2f}, max={np.max(dist_to_boundary):.2f}, outside_px={outside_count}")

    # Distance to image edges
    yy, xx = np.mgrid[0:h, 0:w]
    dist_to_edge = np.minimum(
        np.minimum(xx, (w - 1) - xx),
        np.minimum(yy, (h - 1) - yy),
    ).astype(np.float32)

    print(f"[Project-R] dist_to_edge: min={np.min(dist_to_edge):.2f}, max={np.max(dist_to_edge):.2f}")

    # Check dist_to_boundary for INSIDE pixels only
    inside_mask = inside > 0.5
    if np.any(inside_mask):
        dist_inside = dist_to_boundary[inside_mask]
        print(f"[Project-R] dist_to_boundary (inside only): min={np.min(dist_inside):.2f}, max={np.max(dist_inside):.2f}")

    # Combine: take minimum of both distances, normalize by feather width
    combined = np.minimum(dist_to_boundary, dist_to_edge)
    
    if np.any(inside_mask):
        comb_inside = combined[inside_mask]
        print(f"[Project-R] combined (inside only): min={np.min(comb_inside):.2f}, max={np.max(comb_inside):.2f}")
    
    mask = np.clip(combined / float(feather_px), 0.0, 1.0)

    # Zero out outside coverage
    mask = mask * inside.astype(np.float32)
    
    print(f"[Project-R] effective_mask: min={np.min(mask):.4f}, max={np.max(mask):.4f}")

    return mask.astype(np.float32)


def extend_nearest_valid(
    image: np.ndarray,
    coverage: np.ndarray,
) -> np.ndarray:
    """
    Fill invalid regions (coverage==0) by sampling from nearest valid pixel.
    
    This prevents black/undefined pixels from contaminating blends.
    
    Args:
        image: (H, W, C) image array
        coverage: (H, W) or (H, W, 1) binary mask, 1=valid, 0=invalid
    
    Returns:
        Filled image with same shape as input
    """
    cov_2d = coverage.squeeze() if coverage.ndim > 2 else coverage
    valid_mask = (cov_2d >= 0.5).astype(np.uint8)

    # If all valid, nothing to do
    if np.all(valid_mask):
        return image.copy()

    # If none valid, return as-is (can't extend from nothing)
    if not np.any(valid_mask):
        return image.copy()

    # Get indices of nearest valid pixel for each position
    _, indices = distance_transform_edt(valid_mask, return_indices=True)

    # Sample from those indices
    out = image[indices[0], indices[1]]
    return out.astype(image.dtype)


def resize_half(arr: np.ndarray) -> np.ndarray:
    """Downsample array by 2x using simple averaging."""
    h, w = arr.shape[:2]
    h2, w2 = h // 2, w // 2
    if arr.ndim == 2:
        return arr[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).mean(axis=(1, 3)).astype(arr.dtype)
    else:
        return arr[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, -1).mean(axis=(1, 3)).astype(arr.dtype)


def resize_double_bilinear(arr: np.ndarray) -> np.ndarray:
    """Upsample array by 2x using bilinear interpolation."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        # Fallback: simple duplication
        if arr.ndim == 2:
            return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
        else:
            return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)

    h, w = arr.shape[:2]
    if arr.ndim == 2 or arr.shape[2] == 1:
        squeezed = arr.squeeze()
        # Scale to uint16 for precision
        if squeezed.dtype == np.float32 or squeezed.dtype == np.float64:
            scaled = (np.clip(squeezed, 0, 1) * 65535).astype(np.uint16)
            im = PILImage.fromarray(scaled, mode="I;16")
            im_up = im.resize((w * 2, h * 2), resample=PILImage.BILINEAR)
            return (np.asarray(im_up).astype(np.float32) / 65535.0)[..., None]
        else:
            im = PILImage.fromarray(squeezed)
            im_up = im.resize((w * 2, h * 2), resample=PILImage.BILINEAR)
            return np.asarray(im_up)[..., None]
    else:
        # Multi-channel: resize each channel
        channels = []
        for c in range(arr.shape[2]):
            ch = arr[:, :, c]
            if ch.dtype == np.float32 or ch.dtype == np.float64:
                scaled = (np.clip(ch, 0, 1) * 65535).astype(np.uint16)
                im = PILImage.fromarray(scaled, mode="I;16")
                im_up = im.resize((w * 2, h * 2), resample=PILImage.BILINEAR)
                channels.append(np.asarray(im_up).astype(np.float32) / 65535.0)
            else:
                im = PILImage.fromarray(ch)
                im_up = im.resize((w * 2, h * 2), resample=PILImage.BILINEAR)
                channels.append(np.asarray(im_up))
        return np.stack(channels, axis=-1)


