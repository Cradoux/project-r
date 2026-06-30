"""Physically-based + Wilbur-style erosion for Project-R section heightmaps.

Pure module (NO bpy) so it can be unit-tested / validated outside Blender. Operates on
float32 ``(H, W)`` arrays in **metres** at a known ground ``cell`` size (metres/pixel).

Ported from the Terrascope erosion research toolkit (see the erosion handover, §2-§4 + §9-§10):

- **LEM (landscape evolution):** the engine that carves crisp dendritic drainage. Stream-power
  ``E = K (R A)^m S^n`` via ``FastscapeEroder`` on ``surface_water__discharge`` + ``LinearDiffuser``
  hillslopes, routed by a depression-filling flow router, ~200 steps. Scored on slope-area
  concavity theta (~0.5), NOT spectral beta (beta is a noise artifact).

- **THE GPL WIN:** Project-R is GPL, so the flow router is the fast near-linear GPL
  ``PriorityFloodFlowRouter`` (richdem) — Terrascope (MIT) was forbidden it and had to use the
  superlinear ``DepressionFinderAndRouter``, which explodes on the pits erosion creates. We prefer
  PriorityFlood and fall back to the MIT router only if richdem is unavailable.

- **Seed conditioning (D8 grid-bias fix):** add fine smoothed-gauss / fractal noise (amp ~0.5-0.6)
  to the seed BEFORE eroding so channels meander off-axis. White noise does NOT work; resolution
  and domain-warp do NOT fix grid-bias — only fine sub-grid texture does.

- **Climate forcing:** a per-node rainfall field R drives ``runoff_rate`` so incision concentrates
  where it rains (orographic wet/dry contrast).

- **Wilbur multi-scale blur OVERLAY:** an Incise-Flow channel engraver that blurs the *removed
  material* (the incision field) — not the heightmap — so valleys widen and floors flatten while
  ridges stay crisp. Coarse-to-fine schedule sized automatically from ``(tile_km, res_px)``. Used
  as a LIGHT (shallow, fine-passes-only) overlay onto the LEM surface: it sharpens channels while
  PRESERVING theta. A full-depth overlay DESTROYS theta — keep it shallow.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# One "amplitude unit" of seed-conditioning noise, in metres (matches the toolkit's UNIT).
UNIT_M = 30.0

# Macro smoothing scale as a fraction of grid edge (scale-invariant vs the original 512 grid),
# used only by the synthetic-seed helper for validation/preview.
_MACRO_SIGMA_FRAC = 25.0 / 512.0


# ---------------------------------------------------------------------------
# Erosion presets (scale band x intensity)
# ---------------------------------------------------------------------------
# Scale sets base physics appropriate to the section's real-world size; intensity
# scales how developed the drainage network gets (steps + erodibility). These are
# starting points -- the Custom scale exposes every parameter for manual tuning.
_SCALE_BANDS = (
    ("LOCAL",       500.0,        dict(k_sp=4.0e-5, diffusivity=0.4, uplift=1.0e-3, dt=1000.0, steps=120, w_macro_km=2.0,
                                       c_rate_m=2.0, c_notch_m=15.0, c_fetch_km=15.0, c_steps=20)),
    ("REGIONAL",    1500.0,       dict(k_sp=3.0e-5, diffusivity=0.5, uplift=1.0e-3, dt=1000.0, steps=140, w_macro_km=5.0,
                                       c_rate_m=3.0, c_notch_m=20.0, c_fetch_km=30.0, c_steps=25)),
    ("CONTINENTAL", 4000.0,       dict(k_sp=2.5e-5, diffusivity=0.6, uplift=8.0e-4, dt=1500.0, steps=160, w_macro_km=10.0,
                                       c_rate_m=5.0, c_notch_m=30.0, c_fetch_km=60.0, c_steps=30)),
    ("SUPER",       float("inf"), dict(k_sp=2.0e-5, diffusivity=0.8, uplift=6.0e-4, dt=2000.0, steps=180, w_macro_km=20.0,
                                       c_rate_m=8.0, c_notch_m=45.0, c_fetch_km=120.0, c_steps=35)),
)
_INTENSITY = {
    "GENTLE":   dict(steps_mult=0.45, k_mult=0.6),
    "MODERATE": dict(steps_mult=1.0,  k_mult=1.0),
    "STRONG":   dict(steps_mult=1.4,  k_mult=1.5),
}


STANDARD_RESOLUTIONS = (512, 1024, 2048, 4096, 8192)
AUTO_RES_CAP = 1024  # AUTO stays responsive; choose a fixed size for more detail


def suggest_resolution(native_px: float, cap: int = AUTO_RES_CAP) -> int:
    """A balanced 'optimal' longest-edge size for a section: the largest standard
    resolution that does NOT exceed the crop's native size (so we never upscale and
    invent source detail), floored at 512 and capped (default 1024) so AUTO erosion
    stays responsive. Erosion cost scales ~linearly with pixels, so pick a fixed size
    (up to 8192) for finer synthesized detail at the cost of time."""
    n = max(int(round(native_px)), 1)
    chosen = STANDARD_RESOLUTIONS[0]  # 512 floor
    for s in STANDARD_RESOLUTIONS:
        if s <= n:
            chosen = s
        else:
            break
    return min(chosen, cap)


def resolve_resolution(choice: str, native_px: float) -> int:
    """Map an output_resolution enum value ('AUTO' or a pixel number) to a concrete
    longest-edge size for the section."""
    c = (choice or "AUTO").upper()
    if c in ("AUTO", "", "NATIVE"):
        return suggest_resolution(native_px)
    try:
        return max(64, int(c))
    except (TypeError, ValueError):
        return suggest_resolution(native_px)


def scale_band_for_extent(extent_km: float) -> str:
    """Map a section's width in km to a scale band (the AUTO scale)."""
    e = max(float(extent_km), 1.0)
    for name, hi, _ in _SCALE_BANDS:
        if e < hi:
            return name
    return "SUPER"


def lem_preset(extent_km: float, scale: str = "AUTO", intensity: str = "MODERATE") -> Dict:
    """Resolve (scale, intensity) -> a concrete LEM parameter dict. ``scale='AUTO'``
    (or 'CUSTOM') derives the band from ``extent_km``; intensity scales steps + K."""
    scale = (scale or "AUTO").upper()
    if scale in ("AUTO", "", "CUSTOM"):
        scale = scale_band_for_extent(extent_km)
    base = next((b for n, _, b in _SCALE_BANDS if n == scale), _SCALE_BANDS[1][2])
    inten = _INTENSITY.get((intensity or "MODERATE").upper(), _INTENSITY["MODERATE"])
    return dict(
        k_sp=base["k_sp"] * inten["k_mult"],
        m_sp=0.5,
        n_sp=1.0,
        diffusivity=base["diffusivity"],
        uplift=base["uplift"],
        dt=base["dt"],
        steps=max(20, int(round(base["steps"] * inten["steps_mult"]))),
        overlay_w_macro_km=base["w_macro_km"],
        scale_band=scale,
    )


def coastal_preset(extent_km: float, scale: str = "AUTO", intensity: str = "MODERATE") -> Dict:
    """Resolve (scale, intensity) -> concrete coastal-erosion params, auto-sized to the section the
    same way ``lem_preset`` sizes the river physics. Bigger sections get longer fetch + a taller
    wave-attack band + faster cliff retreat; intensity scales rate (k_mult) and steps (steps_mult).
    ``scale='AUTO'``/'CUSTOM' derives the band from ``extent_km``."""
    scale = (scale or "AUTO").upper()
    if scale in ("AUTO", "", "CUSTOM"):
        scale = scale_band_for_extent(extent_km)
    base = next((b for n, _, b in _SCALE_BANDS if n == scale), _SCALE_BANDS[1][2])
    inten = _INTENSITY.get((intensity or "MODERATE").upper(), _INTENSITY["MODERATE"])
    return dict(
        rate_m=base["c_rate_m"] * inten["k_mult"],
        notch_m=base["c_notch_m"],
        max_fetch_km=base["c_fetch_km"],
        steps=max(8, int(round(base["c_steps"] * inten["steps_mult"]))),
        scale_band=scale,
    )


# ---------------------------------------------------------------------------
# Availability guards (mirror the scipy/Pillow guards in __init__.py)
# ---------------------------------------------------------------------------

def is_landlab_available() -> bool:
    """True if landlab (the erosion engine) can be imported."""
    try:
        import landlab  # noqa: F401
        from landlab.components import FastscapeEroder, LinearDiffuser  # noqa: F401
        return True
    except Exception:
        return False


def is_priorityflood_available() -> bool:
    """True if the fast GPL PriorityFloodFlowRouter (richdem) is importable."""
    try:
        from landlab.components import PriorityFloodFlowRouter  # noqa: F401
        import richdem  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Small numpy/scipy helpers
# ---------------------------------------------------------------------------

def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur via scipy if present, else a separable numpy fallback."""
    if sigma <= 0:
        return arr
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr, sigma)
    except Exception:
        # Separable Gaussian fallback (reflect padding), good enough for previews.
        radius = max(1, int(round(sigma * 3.0)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        k = np.exp(-(x * x) / (2.0 * sigma * sigma))
        k /= k.sum()
        out = arr.astype(np.float64)
        out = np.apply_along_axis(lambda m: np.convolve(np.pad(m, radius, mode="reflect"), k, mode="valid"), 0, out)
        out = np.apply_along_axis(lambda m: np.convolve(np.pad(m, radius, mode="reflect"), k, mode="valid"), 1, out)
        return out.astype(arr.dtype)


def _maximum_filter(arr: np.ndarray, size: int) -> np.ndarray:
    try:
        from scipy.ndimage import maximum_filter
        return maximum_filter(arr, size=size)
    except Exception:
        # Crude fallback via repeated dilation-by-shift (square structuring element).
        out = arr.copy()
        r = size // 2
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                out = np.maximum(out, np.roll(np.roll(arr, dy, axis=0), dx, axis=1))
        return out


def _minimum_filter(arr: np.ndarray, size: int) -> np.ndarray:
    try:
        from scipy.ndimage import minimum_filter
        return minimum_filter(arr, size=size)
    except Exception:
        out = arr.copy()
        r = size // 2
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                out = np.minimum(out, np.roll(np.roll(arr, dy, axis=0), dx, axis=1))
        return out


# ---------------------------------------------------------------------------
# Seed conditioning (fine noise -> meandering channels; breaks D8 grid-bias)
# ---------------------------------------------------------------------------

def make_noise(kind: str, size: Tuple[int, int], seed: int = 7) -> np.ndarray:
    """Unit-std, mean-0 conditioning texture of the requested ``kind``.

    kind: ``none`` -> zeros; ``white`` -> per-pixel (does NOT break grid-bias, here for parity);
    ``gauss`` -> smoothed-gauss (the proven lever); ``fractal`` -> 1/f-ish multi-scale fBm.
    Multiply by ``amp * UNIT_M`` before adding to a heightmap in metres.
    """
    h, w = int(size[0]), int(size[1])
    if kind == "none":
        return np.zeros((h, w), dtype=np.float32)
    rng = np.random.default_rng(seed)
    if kind == "white":
        tex = rng.random((h, w))
    elif kind == "gauss":
        tex = _gaussian_blur(rng.random((h, w)), 1.2)
    elif kind == "fractal":
        tex = np.zeros((h, w), dtype=np.float64)
        a = 1.0
        for octv in (16, 8, 4, 2, 1):  # coarse -> fine, 1/f-ish
            tex += a * _gaussian_blur(rng.random((h, w)), octv)
            a *= 0.5
    else:
        raise ValueError(f"unknown noise kind: {kind!r}")
    std = tex.std()
    tex = (tex - tex.mean()) / (std if std > 1e-12 else 1.0)
    return tex.astype(np.float32)


def condition_seed(height_m: np.ndarray, kind: str, amp: float, seed: int = 7) -> np.ndarray:
    """Return ``height_m`` plus fine conditioning noise (amp in 30 m units, ~0.5-0.6 typical)."""
    if kind == "none" or amp == 0.0:
        return height_m.astype(np.float32, copy=True)
    tex = make_noise(kind, height_m.shape, seed=seed)
    return (height_m + amp * UNIT_M * tex).astype(np.float32)


# ---------------------------------------------------------------------------
# Climate / rainfall forcing field (the right erosion in the right place)
# ---------------------------------------------------------------------------

def climate_field(kind: str, size: Tuple[int, int], strength: float = 1.0) -> np.ndarray:
    """Per-node runoff multiplier field, mean ~1. ``strength`` (0..1+) scales the contrast away
    from uniform: 0 -> flat, 1 -> the full pattern. Patterns ported from run_climate.py.

    kind: ``uniform`` | ``tropical`` (Hadley wet bands) | ``gradient`` (wet west->dry east) |
    ``orographic`` (wet windward + sharp rain shadow).
    """
    h, w = int(size[0]), int(size[1])
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    y /= max(h, 1)
    x /= max(w, 1)
    if kind == "uniform":
        r = np.ones((h, w))
    elif kind == "tropical":
        lat = (y - 0.5) * 2.0
        r = 0.3 + 3.0 * np.exp(-((lat - 0.0) ** 2) / 0.03)
        r += 1.2 * np.exp(-((np.abs(lat) - 0.7) ** 2) / 0.02)
    elif kind == "gradient":
        r = 0.25 + 3.75 * (1.0 - x)
    elif kind == "orographic":
        r = 0.25 + 3.5 / (1.0 + np.exp((x - 0.5) * 25.0))
    else:
        raise ValueError(f"unknown climate kind: {kind!r}")
    r = r * (1.0 / max(float(r.mean()), 1e-9))  # normalize mean to 1
    if strength != 1.0:
        r = 1.0 + float(strength) * (r - 1.0)
        r = np.clip(r, 0.01, None)
        r = r * (1.0 / max(float(r.mean()), 1e-9))
    return r.astype(np.float64)


# ---------------------------------------------------------------------------
# Flow router: prefer the fast GPL PriorityFloodFlowRouter, fall back to MIT
# ---------------------------------------------------------------------------

def _make_router(grid, runoff: np.ndarray, flow_metric: str = "D8") -> Tuple[object, str]:
    """Return ``(router_component, name)``. The component's ``run_one_step()`` fills depressions,
    routes flow, and (re)populates ``drainage_area`` + ``surface_water__discharge`` each call.

    Prefers the GPL ``PriorityFloodFlowRouter`` (richdem; near-linear, robust to the pits erosion
    creates). Falls back to the MIT ``FlowAccumulator`` + ``DepressionFinderAndRouter`` if richdem
    is unavailable.
    """
    runoff = np.asarray(runoff, dtype=np.float64).ravel()
    try:
        from landlab.components import PriorityFloodFlowRouter
        router = PriorityFloodFlowRouter(
            grid,
            surface="topographic__elevation",
            flow_metric=flow_metric,
            runoff_rate=runoff,
            update_flow_depressions=True,
            depression_handler="fill",
            accumulate_flow=True,
            suppress_out=True,
        )
        return router, "PriorityFloodFlowRouter"
    except Exception:
        from landlab.components import FlowAccumulator
        router = FlowAccumulator(
            grid,
            flow_director="FlowDirectorD8",
            depression_finder="DepressionFinderAndRouter",
            runoff_rate=runoff,
        )
        return router, "DepressionFinderAndRouter"


def _new_grid(seed_m: np.ndarray, cell_m: float,
              sea_mask: Optional[np.ndarray] = None, sea_level: float = 0.0):
    """Build a RasterModelGrid seeded with ``topographic__elevation`` (open edges).

    When ``sea_mask`` is given, ocean cells are pinned at ``sea_level`` and marked as
    FIXED_VALUE boundary nodes: they become the base-level OUTLET the land drains
    into, and -- because only core nodes evolve -- the sea neither uplifts nor
    erodes. That keeps the coastline/landmass shape and leaves the ocean flat, while
    rivers still terminate correctly at the coast.
    """
    from landlab import RasterModelGrid
    size_y, size_x = seed_m.shape
    g = RasterModelGrid((int(size_y), int(size_x)), xy_spacing=float(cell_m))
    z = g.add_zeros("topographic__elevation", at="node")
    z[:] = seed_m.astype(np.float64).ravel()
    g.set_closed_boundaries_at_grid_edges(False, False, False, False)  # all edges open
    if sea_mask is not None:
        ocean = np.asarray(sea_mask, dtype=bool).ravel()
        if ocean.any() and not ocean.all():
            z[ocean] = float(sea_level)
            g.status_at_node[ocean] = g.BC_NODE_IS_FIXED_VALUE
    return g, z


# ---------------------------------------------------------------------------
# LEM: stream-power + diffusion landscape evolution (carves crisp drainage)
# ---------------------------------------------------------------------------

def lem_erode(
    height_m: np.ndarray,
    cell_m: float,
    *,
    rainfall: Optional[np.ndarray] = None,
    k_sp: float = 3e-5,
    m_sp: float = 0.5,
    n_sp: float = 1.0,
    diffusivity: float = 0.5,
    uplift: float = 1e-3,
    dt: float = 1000.0,
    steps: int = 200,
    flow_metric: str = "D8",
    sea_mask: Optional[np.ndarray] = None,
    sea_level: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Evolve ``height_m`` under stream-power incision + linear hillslope diffusion.

    Returns ``(z_metres, drainage_area_m2, info)``. ``rainfall`` is an optional per-node runoff
    multiplier (mean ~1); when None a uniform field is used, making ``E = K A^m S^n``. With a
    rainfall field the discharge ``Q = R*A`` drives ``E = K Q^m S^n`` (climate-driven incision).
    ``sea_mask`` (cells at/below ``sea_level``) pins the ocean as a fixed base-level outlet so the
    land erodes toward the coast without the sea uplifting or the coastline being reworked.
    """
    from landlab.components import FastscapeEroder, LinearDiffuser

    seed = height_m.astype(np.float64)
    if rainfall is None:
        runoff = np.ones(seed.size, dtype=np.float64)
    else:
        runoff = np.asarray(rainfall, dtype=np.float64).ravel()

    g, z = _new_grid(seed, cell_m, sea_mask=sea_mask, sea_level=sea_level)
    router, router_name = _make_router(g, runoff, flow_metric=flow_metric)
    sp = FastscapeEroder(g, K_sp=k_sp, m_sp=m_sp, n_sp=n_sp,
                         discharge_field="surface_water__discharge")
    ld = LinearDiffuser(g, linear_diffusivity=diffusivity)
    core = g.core_nodes

    t0 = time.perf_counter()
    for _ in range(int(steps)):
        z[core] += uplift * dt
        router.run_one_step()
        sp.run_one_step(dt)
        ld.run_one_step(dt)
    secs = time.perf_counter() - t0

    info = {"router": router_name, "secs": round(secs, 2), "steps": int(steps)}
    out = z.reshape(seed.shape).astype(np.float32)
    dr = g.at_node["drainage_area"].reshape(seed.shape).astype(np.float32)
    return out, dr, info


# ---------------------------------------------------------------------------
# Wilbur multi-scale blur overlay (Incise Flow) + the resolution formula
# ---------------------------------------------------------------------------

def blur_schedule_from_resolution(
    tile_km: float,
    res_px: float,
    *,
    w_macro_km: float = 8.0,
    r: float = 0.4,
    blur_min_px: float = 0.5,
    depth_macro_m: float = 600.0,
) -> Tuple[List[Dict], float]:
    """The blur-from-(tile_km, res_px) schedule (HANDOFF §10c). Returns ``(schedule, cell_m)``.

    blur_0 = W_macro / (2*cell); blur_k = blur_0 * r^k down to ~blur_min_px. flow_exp 0.40->0.10,
    effect_blend 0.25->0.10, amount(rel) 3.5->0.6 ramp macro->detail, min_area large->small.
    Scales automatically with tile size and resolution.
    """
    cell = float(tile_km) * 1000.0 / max(float(res_px), 1.0)
    blur0 = float(w_macro_km) * 1000.0 / (2.0 * cell)
    blurs: List[float] = []
    b = blur0
    while b >= blur_min_px and len(blurs) < 8:
        blurs.append(b)
        b *= r
    if not blurs:
        blurs = [max(blur0, blur_min_px)]
    n = len(blurs)
    sched: List[Dict] = []
    for k, bl in enumerate(blurs):
        frac = k / (n - 1) if n > 1 else 0.0  # 0 macro -> 1 detail
        sched.append(dict(
            blur_px=round(bl, 3),
            flow_exp=round(0.40 + (0.10 - 0.40) * frac, 3),
            amount=round(depth_macro_m * ((3.5 + (0.6 - 3.5) * frac) / 3.5), 1),
            effect_blend=round(0.25 + (0.10 - 0.25) * frac, 3),
            min_area_m2=50e6 * ((1e6 / 50e6) ** frac),
            iters=2,
        ))
    return sched, cell


def incise_schedule(
    seed_m: np.ndarray,
    runoff: np.ndarray,
    schedule: List[Dict],
    cell_m: float,
    *,
    base: float = 0.0,
    clip: bool = False,
    flow_metric: str = "D8",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Wilbur Incise-Flow carve over a pass-schedule (HANDOFF §10). For each pass: route/fill,
    build incision field ``f = (Q/q99)^flow_exp`` (optionally area-gated + blurred), then
    ``z = max(z - effect_blend*amount*f, base)``. The blur acts on the REMOVED material ``f`` (not
    z), so valleys get flat feathered floors while ridges stay crisp. Returns ``(z, dr, secs)``.
    """
    seed = seed_m.astype(np.float64)
    runoff = np.asarray(runoff, dtype=np.float64).ravel()
    g, z = _new_grid(seed, cell_m)
    router, _ = _make_router(g, runoff, flow_metric=flow_metric)
    core = g.core_nodes
    shape = seed.shape

    t0 = time.perf_counter()
    for p in schedule:
        for _ in range(int(p.get("iters", 1))):
            router.run_one_step()
            q = g.at_node["surface_water__discharge"]
            area = g.at_node["drainage_area"]
            ratio = q / max(np.percentile(q, 99), 1e-9)
            if clip:
                ratio = np.clip(ratio, 0.0, 1.0)
            f = ratio ** p["flow_exp"]
            if p.get("min_area_m2"):
                f = f * (area >= p["min_area_m2"])
            if p["blur_px"] > 0:
                f = _gaussian_blur(f.reshape(shape), p["blur_px"]).ravel()
            f = np.clip(f / max(np.percentile(f, 99), 1e-12), 0.0, 1.0)
            z[core] = np.maximum(z[core] - p["effect_blend"] * p["amount"] * f[core], base)
    secs = time.perf_counter() - t0

    out = z.reshape(shape).astype(np.float32)
    dr = g.at_node["drainage_area"].reshape(shape).astype(np.float32)
    return out, dr, secs


def wilbur_overlay(
    height_m: np.ndarray,
    cell_m: float,
    tile_km: float,
    res_px: float,
    *,
    rainfall: Optional[np.ndarray] = None,
    depth_macro_m: float = 200.0,
    w_macro_km: float = 8.0,
    r: float = 0.4,
    skip_macro: bool = True,
    base: float = 0.0,
    flow_metric: str = "D8",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """LIGHT Incise-Flow overlay onto an (already LEM-evolved) surface: sharpens climate-aware
    flat-bottomed channels while PRESERVING the LEM's drainage concavity. Builds a shallow
    schedule from the resolution formula and (by default) drops the macro pass since the LEM
    already supplies broad valleys. A deep/full overlay DESTROYS theta — keep ``depth_macro_m``
    small and ``skip_macro`` on. Returns ``(z, dr, secs)``.
    """
    sched, _ = blur_schedule_from_resolution(
        tile_km, res_px, w_macro_km=w_macro_km, r=r, depth_macro_m=depth_macro_m
    )
    if skip_macro and len(sched) > 1:
        sched = sched[1:]
    if rainfall is None:
        runoff = np.ones(height_m.size, dtype=np.float64)
    else:
        runoff = np.asarray(rainfall, dtype=np.float64).ravel()
    return incise_schedule(height_m, runoff, sched, cell_m, base=base, flow_metric=flow_metric)


# ---------------------------------------------------------------------------
# Coastal (wave) erosion overlay — fetch-driven shoreline reworking
# ---------------------------------------------------------------------------
# The LEM/Wilbur passes deliberately FREEZE the coast (the ocean is pinned as a
# fixed base-level outlet). This pass does the opposite: it actively reworks the
# shoreline with wave energy. Headlands that "see" lots of open water (long
# fetch) erode and retreat into cliffs; the removed sediment is redeposited as
# beaches in sheltered bays — the classic headland-and-bay equilibrium. Pure
# numpy + the scipy-optional helpers above. The algorithm is REIMPLEMENTED from
# the CEM (BSD) alongshore-transport idea + SCAPE-style shore-platform
# downwearing, so it carries no model dependency and stays GPL-clean.
#
# Intended ordering: run this BEFORE lem_erode so rivers then incise the
# reworked coast. Same `sea_mask = height_m <= sea_level` convention as
# run_erosion; output stays in metres (peak-rescale downstream as usual).

# Seaward look directions; diagonals carry a sqrt(2) step length.
_DIRS8 = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)


def _shift_fill0(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift ``a`` by (dy, dx) filling the exposed edge with 0 (no torus wrap, so
    off-grid reads as land and never inflates fetch at the borders)."""
    out = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
    if dy > 0:
        out[:dy, :] = 0
    elif dy < 0:
        out[dy:, :] = 0
    if dx > 0:
        out[:, :dx] = 0
    elif dx < 0:
        out[:, dx:] = 0
    return out


def _sea_fetch(sea: np.ndarray, dy: int, dx: int, max_px: int) -> np.ndarray:
    """Run-length (in cells, capped at ``max_px``) of contiguous sea looking in
    direction (dy, dx). Value at a sea cell ~ open water ahead; 0 on land.

    Parallel relaxation of ``F = sea * (1 + F[next])`` — the fixed point IS the
    true run length, and each iteration grows every cell by at most 1, so
    stopping after ``max_px`` iters caps the fetch at ``max_px`` (we don't care
    about open water past the wave-relevant distance)."""
    sea_f = sea.astype(np.float32)
    F = sea_f.copy()
    for _ in range(int(max(max_px, 1))):
        nxt = _shift_fill0(F, -dy, -dx)  # value of the cell at c+(dy,dx), read at c
        F = sea_f * (1.0 + nxt)
    return F


def wave_exposure(
    height_m: np.ndarray,
    cell_m: float,
    sea_level: float = 0.0,
    *,
    max_fetch_km: float = 25.0,
    swell_deg: Optional[float] = None,
    swell_focus: float = 0.6,
) -> np.ndarray:
    """Per-cell wave-exposure field, in **metres of effective open-water fetch**.

    For every cell we look seaward in 8 directions; exposure is the (optionally
    swell-weighted) sum of the open-water fetch reaching it. Coast cells facing
    long stretches of ocean (headlands) score high; the backs of bays score low.
    Computed for land AND sea cells: land uses it as the erosion driver, sea uses
    its inverse as the deposition "shelter" weight.

    ``swell_deg`` is the compass bearing (0=N, 90=E) the dominant swell comes
    FROM; ``swell_focus`` 0 -> isotropic, 1 -> strongly directional. None -> calm
    (all directions equal)."""
    sea = np.asarray(height_m, dtype=np.float32) <= float(sea_level)
    max_px = max(1, int(round(max_fetch_km * 1000.0 / max(float(cell_m), 1e-6))))
    max_px = min(max_px, max(height_m.shape))  # fetch can't exceed the grid

    if swell_deg is None:
        weights = [1.0] * len(_DIRS8)
    else:
        th = np.radians(float(swell_deg))
        src = np.array([-np.cos(th), np.sin(th)])  # (dy,dx) the waves come FROM
        weights = []
        for (dy, dx) in _DIRS8:
            v = np.array([dy, dx], dtype=np.float64)
            v /= np.hypot(*v)
            align = max(0.0, float(np.dot(v, src)))
            weights.append((1.0 - float(swell_focus)) + float(swell_focus) * align)
    wmean = max(sum(weights) / len(weights), 1e-9)

    expo = np.zeros(height_m.shape, dtype=np.float32)
    for (dy, dx), w in zip(_DIRS8, weights):
        step_m = float(cell_m) * (np.sqrt(2.0) if (dy and dx) else 1.0)
        F = _sea_fetch(sea, dy, dx, max_px)
        neigh = _shift_fill0(F, -dy, -dx)  # fetch of the seaward neighbour, read at c
        expo += (w / wmean) * neigh * step_m
    return expo


def coastal_erode(
    height_m: np.ndarray,
    cell_m: float,
    *,
    sea_level: float = 0.0,
    steps: int = 25,
    rate_m: float = 3.0,
    notch_m: float = 20.0,
    platform_depth_m: float = 4.0,
    max_fetch_km: float = 25.0,
    swell_deg: Optional[float] = None,
    swell_focus: float = 0.6,
    susceptibility: Optional[np.ndarray] = None,
    deposition: bool = True,
    deposit_radius_px: int = 6,
    talus_deg: float = 0.0,
    refetch_every: int = 3,
) -> Tuple[np.ndarray, Dict]:
    """Wave-driven coastal erosion + beach deposition.

    Each step: (1) compute the fetch/exposure field; (2) erode land within the
    wave-attack band (``notch_m`` above sea level), weighted by exposure and
    ``susceptibility``, but never below the wave-cut platform at
    ``sea_level - platform_depth_m``; (3) redeposit the removed volume as beaches
    in sheltered (low-exposure) shallow water near the coast, capped at sea level;
    (4) optional ``talus_deg`` collapse so undercut cliff faces retreat instead of
    standing vertical. Flooded land becomes sea on the next step, so the coastline
    physically retreats. Returns ``(z_metres, info)``.

    Tuning notes: ``rate_m`` is the max vertical lowering at the most-exposed
    waterline cell per step; ``notch_m`` sets how far up the cliff waves bite;
    ``susceptibility`` is an optional (H,W) erodibility field (~1 = default rock,
    >1 = soft, 0 = armoured). Fetch cost scales with ``max_fetch_km / cell``; on
    big grids raise ``refetch_every`` or lower ``max_fetch_km``. Mass is conserved
    up to what shallow shelter can hold — ``info['conserved_frac']`` reports the
    rest (lost to deep water offshore)."""
    z = np.asarray(height_m, dtype=np.float64).copy()
    cell_area = float(cell_m) ** 2
    floor = float(sea_level) - float(platform_depth_m)
    sus = (np.ones_like(z) if susceptibility is None
           else np.asarray(susceptibility, dtype=np.float64).reshape(z.shape))
    talus_drop = (np.tan(np.radians(talus_deg)) * float(cell_m)) if talus_deg > 0 else 0.0

    eroded_vol = 0.0
    deposited_vol = 0.0
    expo = None
    hot = 0.0
    t0 = time.perf_counter()
    for it in range(int(steps)):
        if expo is None or (it % max(int(refetch_every), 1) == 0):
            expo = wave_exposure(z, cell_m, sea_level, max_fetch_km=max_fetch_km,
                                 swell_deg=swell_deg, swell_focus=swell_focus)
            hot = float(np.percentile(expo[expo > 0], 99)) if np.any(expo > 0) else 0.0
        if hot <= 0.0:
            break
        e = np.clip(expo / hot, 0.0, 1.0)

        # Erode the wave-attack band; clamp the cut so it can't pass the platform.
        h = z - float(sea_level)
        band = np.clip(1.0 - h / max(float(notch_m), 1e-6), 0.0, 1.0) * (h > 0.0)
        erode = float(rate_m) * e * band * sus
        erode = np.minimum(erode, np.maximum(z - floor, 0.0))
        z = z - erode
        vol = float(erode.sum()) * cell_area
        eroded_vol += vol

        # Redeposit as beaches in sheltered shallow water adjacent to the coast.
        if deposition and vol > 0.0:
            land = z > float(sea_level)
            near = _maximum_filter(land.astype(np.uint8), size=2 * int(deposit_radius_px) + 1) > 0
            room = np.clip(float(sea_level) - z, 0.0, None)  # >0 only where submerged
            target = near & (room > 0.0)
            if target.any():
                shelter = np.clip(1.0 - e, 0.0, 1.0)
                w = shelter * np.sqrt(room) * target
                wsum = float(w.sum())
                if wsum > 1e-9:
                    add = (vol / cell_area) * (w / wsum)
                    add = np.where(target, np.minimum(add, room), 0.0)
                    z = z + add
                    deposited_vol += float(add.sum()) * cell_area

        # Light seaward collapse so undercut faces retreat (gated to the coast).
        if talus_drop > 0.0:
            coast = _maximum_filter((z <= float(sea_level)).astype(np.uint8),
                                    size=2 * int(deposit_radius_px) + 1) > 0
            zmin = _minimum_filter(z, size=3)
            over = ((z - zmin) > talus_drop) & coast & (z > float(sea_level))
            z = np.where(over, zmin + talus_drop, z)

    secs = time.perf_counter() - t0
    info = dict(
        secs=round(secs, 2), steps=int(steps),
        eroded_m3=round(eroded_vol, 1), deposited_m3=round(deposited_vol, 1),
        conserved_frac=(round(deposited_vol / eroded_vol, 3) if eroded_vol > 0 else float("nan")),
    )
    return z.astype(np.float32), info


# ---------------------------------------------------------------------------
# Metrics (score on theta / band_slope, NOT spectral beta)
# ---------------------------------------------------------------------------

def d8_slope(z: np.ndarray, cell: float) -> np.ndarray:
    """Steepest-descent (D8) downhill slope per cell from z alone. Returns (H,W) >= 0."""
    h, w = z.shape
    zp = np.pad(z, 1, mode="edge")
    diag = cell * np.sqrt(2.0)
    best = np.zeros_like(z)
    for dy, dx, dist in [(-1, 0, cell), (1, 0, cell), (0, -1, cell), (0, 1, cell),
                         (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)]:
        nb = zp[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        best = np.maximum(best, (z - nb) / dist)
    return np.maximum(best, 0.0)


def slope_area_concavity(z: np.ndarray, dr: np.ndarray, cell: float,
                         lo_pct: float = 45, hi_pct: float = 99, nbins: int = 16
                         ) -> Tuple[float, float, float]:
    """theta, R2, n_decades from a binned slope-area regression over the fluvial domain.
    Steady-state stream power gives ``S ~ A^-theta`` with theta=m/n (~0.5). REAL drainage -> clean
    negative power law (high R2); NOISE -> theta~0, low R2. This is the believability gate."""
    s = d8_slope(z, cell).ravel()
    a = dr.ravel().astype(float)
    m = (a > 0) & (s > 0)
    if m.sum() < 50:
        return float("nan"), float("nan"), 0.0
    la, ls = np.log10(a[m]), np.log10(s[m])
    lo, hi = np.percentile(la, lo_pct), np.percentile(la, hi_pct)
    bins = np.linspace(lo, hi, nbins + 1)
    idx = np.digitize(la, bins)
    xs, ys = [], []
    for b in range(1, len(bins)):
        sel = idx == b
        if sel.sum() > 20:
            xs.append(0.5 * (bins[b - 1] + bins[b]))
            ys.append(np.median(ls[sel]))
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 5:
        return float("nan"), float("nan"), 0.0
    coef = np.polyfit(xs, ys, 1)
    pred = np.polyval(coef, xs)
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return float(-coef[0]), float(r2), float(xs.max() - xs.min())


def band_slope(z: np.ndarray, dr: np.ndarray, cell: float,
               area_threshold_m2: float = 25e6, win: int = 9) -> float:
    """Median per-pixel gradient magnitude in the dilated channel band. LOWER = flatter/wider valley
    floors (flat-bottomed); HIGHER = sharp V-notch walls. The right floor-flatness proxy for the
    carver. Uses per-pixel differences (``np.gradient(z)``, no cell spacing) to match the research
    reference (run_wilbur_schedule.py); ``cell`` is accepted only for call-site parity."""
    gy, gx = np.gradient(z)
    gm = np.sqrt(gx ** 2 + gy ** 2)
    band = _maximum_filter((dr >= area_threshold_m2).astype(np.uint8), size=win) > 0
    return float(np.median(gm[band])) if band.any() else float("nan")


def flat_floor_score(z: np.ndarray, dr: np.ndarray, area_threshold_m2: float = 25e6,
                     win: int = 9, tol_frac: float = 0.05) -> float:
    """Fraction of valley-band cells within ``tol`` of their local min elevation (wide flat floors
    -> higher). Weaker than band_slope per §10.1; reported for parity."""
    chan = dr >= area_threshold_m2
    if chan.sum() == 0:
        return float("nan")
    zmin = _minimum_filter(z, size=win)
    zmax = _maximum_filter(z, size=win)
    tol = tol_frac * (zmax - zmin + 1e-9)
    nearfloor = (z - zmin) <= tol
    band = _maximum_filter(chan.astype(np.uint8), size=win) > 0
    return float(nearfloor[band].mean())


def drainage_density_fixed(dr: np.ndarray, area_threshold_m2: float = 25e6) -> float:
    """Channel-cell fraction at a FIXED drainage-area threshold (default 25 km^2 channel head)."""
    return float((dr >= area_threshold_m2).mean())


def score(z: np.ndarray, dr: np.ndarray, cell: float) -> Dict:
    """Bundle the trustworthy metrics for an in-UI quality readout."""
    theta, r2, ndec = slope_area_concavity(z, dr, cell)
    return dict(
        theta=round(theta, 3) if np.isfinite(theta) else float("nan"),
        r2=round(r2, 3) if np.isfinite(r2) else float("nan"),
        n_decades=round(ndec, 2),
        band_slope=round(band_slope(z, dr, cell), 4),
        drainage_density=round(drainage_density_fixed(dr), 4),
        relief_m=round(float(z.max() - z.min()), 1),
    )


# ---------------------------------------------------------------------------
# Peak rescaling (linear; NEVER CDF-remap -> re-amplifies noise)
# ---------------------------------------------------------------------------

def rescale_peak(z_m: np.ndarray, target_peak_m: float, base: float = 0.0) -> np.ndarray:
    """Linearly scale land above ``base`` so its max == ``target_peak_m`` (HANDOFF §7). Does NOT
    CDF-remap. If the current peak is at/below base, returns z unchanged."""
    z = z_m.astype(np.float32)
    cur_peak = float(z.max())
    if cur_peak <= base or target_peak_m <= base:
        return z
    scale = (float(target_peak_m) - base) / (cur_peak - base)
    return (base + (z - base) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Orchestrator (used by the operator and by validation)
# ---------------------------------------------------------------------------

def run_erosion(
    height_m: np.ndarray,
    cell_m: float,
    tile_km: float,
    res_px: float,
    *,
    noise_kind: str = "gauss",
    noise_amp: float = 0.55,
    noise_seed: int = 7,
    climate_kind: str = "uniform",
    climate_strength: float = 1.0,
    rainfall: Optional[np.ndarray] = None,
    k_sp: float = 3e-5,
    m_sp: float = 0.5,
    n_sp: float = 1.0,
    diffusivity: float = 0.5,
    uplift: float = 1e-3,
    dt: float = 1000.0,
    steps: int = 200,
    enable_overlay: bool = False,
    overlay_depth_m: float = 200.0,
    overlay_w_macro_km: float = 8.0,
    overlay_r: float = 0.4,
    target_peak_m: Optional[float] = None,
    base: float = 0.0,
    sea_level_m: float = 0.0,
    flow_metric: str = "D8",
    enable_coastal: bool = False,
    coastal_steps: int = 25,
    coastal_rate_m: float = 3.0,
    coastal_notch_m: float = 20.0,
    coastal_max_fetch_km: float = 25.0,
    coastal_swell_deg: Optional[float] = None,
    coastal_swell_focus: float = 0.0,
    coastal_talus_deg: float = 0.0,
    coastal_deposition: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Full per-section pipeline: (optional coastal wave reworking) -> condition seed -> rainfall
    -> LEM -> (optional light Wilbur overlay) -> linear peak rescale. Returns
    ``(eroded_metres, metrics_dict)``. Pure: callable from the Blender operator and from a
    headless validation harness alike.

    The coastal pass (when ``enable_coastal``) runs FIRST and reshapes the baseline surface, so
    everything downstream -- ``sea_mask``, seed conditioning, the ocean restore -- sees the
    reworked coast. The operator applies the same pass to its blend baseline (see erode_ops), so
    coastline changes survive the strength blend instead of being restored away.
    """
    height_m = np.asarray(height_m, dtype=np.float32)

    coastal_info = None
    if enable_coastal:
        height_m, coastal_info = coastal_erode(
            height_m, cell_m, sea_level=float(sea_level_m), steps=coastal_steps,
            rate_m=coastal_rate_m, notch_m=coastal_notch_m, max_fetch_km=coastal_max_fetch_km,
            swell_deg=coastal_swell_deg, swell_focus=coastal_swell_focus,
            talus_deg=coastal_talus_deg, deposition=coastal_deposition,
        )

    # Ocean = cells at/below sea level (from the ORIGINAL surface, before noise). They
    # become fixed base-level outlets in the LEM and are restored exactly afterwards,
    # so the sea stays flat and the coastline keeps its shape.
    sea_mask = height_m <= float(sea_level_m)
    has_sea = bool(sea_mask.any()) and not bool(sea_mask.all())

    seed = condition_seed(height_m, noise_kind, noise_amp, seed=noise_seed)
    if has_sea:
        seed = seed.copy()
        seed[sea_mask] = float(sea_level_m)  # don't condition (or erode) the ocean

    # Per-node runoff multiplier driving discharge (Q = R*A). Priority: an explicit
    # rainfall MAP (normalised to mean ~1 over land) > the synthetic climate pattern
    # > uniform (None). The map lets incision concentrate where the user paints rain.
    if rainfall is not None:
        rain_field = np.asarray(rainfall, dtype=np.float64)
        if rain_field.shape != height_m.shape:
            rain_field = rain_field.reshape(height_m.shape)
        land = ~sea_mask
        mr = float(rain_field[land].mean()) if land.any() else float(rain_field.mean())
        rain_field = (rain_field / mr) if mr > 1e-9 else rain_field
        rain_field = np.clip(rain_field, 0.0, 10.0).astype(np.float32)
    elif climate_kind != "uniform":
        rain_field = climate_field(climate_kind, height_m.shape, strength=climate_strength)
    else:
        rain_field = None

    z, dr, lem_info = lem_erode(
        seed, cell_m, rainfall=rain_field, k_sp=k_sp, m_sp=m_sp, n_sp=n_sp,
        diffusivity=diffusivity, uplift=uplift, dt=dt, steps=steps, flow_metric=flow_metric,
        sea_mask=sea_mask if has_sea else None, sea_level=float(sea_level_m),
    )

    overlay_secs = None
    if enable_overlay:
        z, dr, overlay_secs = wilbur_overlay(
            z, cell_m, tile_km, res_px, rainfall=rain_field,
            depth_macro_m=overlay_depth_m, w_macro_km=overlay_w_macro_km, r=overlay_r,
            skip_macro=True, base=base, flow_metric=flow_metric,
        )

    # Restore the ocean to its original (flat/black) values -- the LEM held it at sea
    # level, but the optional overlay is sea-unaware, so re-pin it here.
    if has_sea:
        z = z.copy()
        z[sea_mask] = height_m[sea_mask]

    metrics = score(z, dr, cell_m)

    if target_peak_m is not None:
        z = rescale_peak(z, float(target_peak_m), base=base)

    metrics.update(
        router=lem_info.get("router"),
        lem_secs=lem_info.get("secs"),
        overlay_secs=round(overlay_secs, 2) if overlay_secs is not None else None,
        coastal_secs=coastal_info.get("secs") if coastal_info else None,
        coastal_conserved_frac=coastal_info.get("conserved_frac") if coastal_info else None,
        cell_m=round(float(cell_m), 2),
    )
    return z.astype(np.float32), metrics


def make_synthetic_seed(size: int, noise_kind: str = "gauss", amp: float = 0.6, seed: int = 1
                        ) -> np.ndarray:
    """A smooth macro seed + conditioning noise, in metres — for validation/preview only."""
    rng = np.random.default_rng(seed)
    sig = _MACRO_SIGMA_FRAC * size
    base = _gaussian_blur(rng.random((size, size)), sig) * 900.0
    return condition_seed(base.astype(np.float32), noise_kind, amp, seed=seed + 6)
