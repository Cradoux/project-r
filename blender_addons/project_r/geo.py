from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class LonLat:
    lon: float  # radians, [-pi, pi]
    lat: float  # radians, [-pi/2, pi/2]


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __truediv__(self, s: float) -> "Vec3":
        return Vec3(self.x / s, self.y / s, self.z / s)


def uv_to_lonlat(u: float, v: float) -> LonLat:
    lon = (u * 2.0 * math.pi) - math.pi
    lat = (v * math.pi) - (math.pi / 2.0)
    return LonLat(lon=lon, lat=lat)


def lonlat_to_unit_vec(p: LonLat) -> Vec3:
    cl = math.cos(p.lat)
    return Vec3(
        x=cl * math.cos(p.lon),
        y=cl * math.sin(p.lon),
        z=math.sin(p.lat),
    )


def normalize(v: Vec3) -> Vec3:
    n = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if n == 0.0:
        return Vec3(1.0, 0.0, 0.0)
    return Vec3(v.x / n, v.y / n, v.z / n)


def mean_center_lonlat(points: Sequence[LonLat]) -> LonLat:
    if not points:
        return LonLat(0.0, 0.0)
    acc = Vec3(0.0, 0.0, 0.0)
    for p in points:
        acc = acc + lonlat_to_unit_vec(p)
    m = normalize(acc / float(len(points)))
    lon = math.atan2(m.y, m.x)
    lat = math.asin(max(-1.0, min(1.0, m.z)))
    return LonLat(lon=lon, lat=lat)


@dataclass(frozen=True)
class RectI:
    x: int
    y: int
    w: int
    h: int


def expand_rect_square(rect: RectI, bounds_w: int, bounds_h: int) -> RectI:
    side = max(rect.w, rect.h)
    cx = rect.x + rect.w / 2.0
    cy = rect.y + rect.h / 2.0
    x = int(round(cx - side / 2.0))
    y = int(round(cy - side / 2.0))
    x = max(0, min(x, bounds_w - side))
    y = max(0, min(y, bounds_h - side))
    side = max(1, min(side, bounds_w, bounds_h))
    return RectI(x=x, y=y, w=side, h=side)


def _rot_z(v: Vec3, ang: float) -> Vec3:
    c = math.cos(ang)
    s = math.sin(ang)
    return Vec3(x=v.x * c - v.y * s, y=v.x * s + v.y * c, z=v.z)


def _rot_y(v: Vec3, ang: float) -> Vec3:
    c = math.cos(ang)
    s = math.sin(ang)
    return Vec3(x=v.x * c + v.z * s, y=v.y, z=-v.x * s + v.z * c)


def _rot_x(v: Vec3, ang: float) -> Vec3:
    c = math.cos(ang)
    s = math.sin(ang)
    return Vec3(x=v.x, y=v.y * c - v.z * s, z=v.y * s + v.z * c)


def rotate_to_aspect(p: LonLat, *, center: LonLat, rot_rad: float) -> LonLat:
    """
    Rotate a lon/lat point so that `center` becomes lon=0, lat=0 (map center),
    then apply an additional roll about the new x-axis by `rot_rad`.
    """
    v = lonlat_to_unit_vec(p)
    v = _rot_z(v, -center.lon)
    v = _rot_y(v, -center.lat)
    if rot_rad != 0.0:
        v = _rot_x(v, -rot_rad)
    lon = math.atan2(v.y, v.x)
    lat = math.asin(max(-1.0, min(1.0, v.z)))
    return LonLat(lon=lon, lat=lat)


def hammer_xy_unit(p_rot: LonLat) -> Tuple[float, float]:
    """
    Hammer projection for rotated lon/lat (radians), returning normalized x,y in [-1,1].
    Normalization matches the common Hammer extents (x in [-2*sqrt(2), 2*sqrt(2)], y in [-sqrt(2), sqrt(2)]).
    """
    lon = p_rot.lon
    lat = p_rot.lat
    denom = math.sqrt(1.0 + math.cos(lat) * math.cos(lon / 2.0))
    if denom == 0.0:
        return 0.0, 0.0
    x = (2.0 * math.sqrt(2.0) * math.cos(lat) * math.sin(lon / 2.0)) / denom
    y = (math.sqrt(2.0) * math.sin(lat)) / denom
    # Normalize to [-1,1]
    return x / (2.0 * math.sqrt(2.0)), y / (math.sqrt(2.0))


def unit_xy_to_pixel(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    """
    Convert normalized projection coords x,y in [-1,1] to pixel coords (float).
    """
    px = (x + 1.0) / 2.0 * width - 0.5
    py = (1.0 - y) / 2.0 * height - 0.5
    return px, py


def compute_section_extent_km(
    center_lon_deg: float,
    center_lat_deg: float,
    lon_range_deg: float,
    lat_range_deg: float,
    planet_radius_km: float,
) -> Tuple[float, float]:
    """
    Compute approximate width x height in km for a section.

    Args:
        center_lon_deg: Center longitude in degrees
        center_lat_deg: Center latitude in degrees
        lon_range_deg: Total longitude span in degrees
        lat_range_deg: Total latitude span in degrees
        planet_radius_km: Planet radius in kilometers

    Returns:
        (width_km, height_km) tuple
    """
    lat_rad = math.radians(center_lat_deg)

    # Height (N-S): arc length along meridian (same at any longitude)
    height_km = math.radians(lat_range_deg) * planet_radius_km

    # Width (E-W): arc length at center latitude (shrinks toward poles)
    width_km = math.radians(lon_range_deg) * planet_radius_km * math.cos(lat_rad)

    return (abs(width_km), abs(height_km))


def compute_lonlat_bounds(points: Sequence[LonLat]) -> Tuple[float, float, float, float]:
    """
    Compute the bounding box of lon/lat points in degrees.

    Returns:
        (min_lon_deg, max_lon_deg, min_lat_deg, max_lat_deg)
    """
    if not points:
        return (0.0, 0.0, 0.0, 0.0)

    lons = [math.degrees(p.lon) for p in points]
    lats = [math.degrees(p.lat) for p in points]

    return (min(lons), max(lons), min(lats), max(lats))


