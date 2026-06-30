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

# One "amplitude unit" of seed-conditioning noise, in metres (legacy absolute fallback,
# used only when a relief is not supplied).
UNIT_M = 30.0

# Seed-conditioning noise as a FRACTION of the terrain's relief per amplitude unit.
# On real heightmaps the relief is hundreds-to-thousands of metres, so a fixed 30 m
# perturbation is negligible and the Noise Amount appears to do nothing. Scaling to
# relief makes the amount an actual lever: amp 0.55 -> ~3% of relief, amp 2 -> ~12%.
REL_NOISE_FRAC = 0.06

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


def _box_mean(arr: np.ndarray, radius_px: float) -> np.ndarray:
    """Mean over a ``(2r+1)`` square window, O(N) REGARDLESS of window size (so a wide inland
    relief window stays cheap at 4K). scipy ``uniform_filter`` if present, else a summed-area
    (integral-image) table. Edges use a shrinking window (clamped), not zero-padding."""
    r = max(int(round(radius_px)), 1)
    a = np.asarray(arr, dtype=np.float64)
    try:
        from scipy.ndimage import uniform_filter
        return uniform_filter(a, size=2 * r + 1, mode="nearest")
    except Exception:
        H, W = a.shape
        ii = np.zeros((H + 1, W + 1), dtype=np.float64)
        ii[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
        y0 = np.clip(np.arange(H) - r, 0, H); y1 = np.clip(np.arange(H) + r + 1, 0, H)
        x0 = np.clip(np.arange(W) - r, 0, W); x1 = np.clip(np.arange(W) + r + 1, 0, W)
        s = (ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)] - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)])
        cnt = (y1 - y0)[:, None] * (x1 - x0)[None, :]
        return s / np.maximum(cnt, 1.0)


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


def condition_seed(height_m: np.ndarray, kind: str, amp: float, seed: int = 7,
                   relief_m: Optional[float] = None) -> np.ndarray:
    """Return ``height_m`` plus fine conditioning noise. ``amp`` scales the noise
    RELATIVE to ``relief_m`` (terrain relief) when given -- so it's a meaningful lever
    at any terrain height -- and falls back to absolute 30 m units otherwise."""
    if kind == "none" or amp == 0.0:
        return height_m.astype(np.float32, copy=True)
    tex = make_noise(kind, height_m.shape, seed=seed)
    if relief_m and relief_m > 0.0:
        noise_std = amp * REL_NOISE_FRAC * float(relief_m)
    else:
        noise_std = amp * UNIT_M
    return (height_m + noise_std * tex).astype(np.float32)


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

def _stretch01(a: np.ndarray, land: Optional[np.ndarray] = None) -> np.ndarray:
    """Robustly rescale an array to [0,1] using p2..p98 of its LAND values (so a
    spatial-driver crop spans the full knob range regardless of absolute brightness)."""
    a = np.asarray(a, dtype=np.float64)
    vals = a[land] if (land is not None and np.any(land)) else a.ravel()
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if hi - lo < 1e-9:
        return np.full_like(a, 0.5)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


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
    k_field: Optional[np.ndarray] = None,
    uplift_field: Optional[np.ndarray] = None,
    enable_deposition: bool = False,
    depo_v_s: float = 1.0,
    depo_k_sed: Optional[float] = None,
    depo_k_br: Optional[float] = None,
    depo_h_star_m: float = 1.0,
    depo_phi: float = 0.3,
    depo_f_f: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Evolve ``height_m`` under stream-power incision + linear hillslope diffusion.

    Returns ``(z_metres, drainage_area_m2, info)``. ``rainfall`` is an optional per-node runoff
    multiplier (mean ~1); when None a uniform field is used, making ``E = K A^m S^n``. With a
    rainfall field the discharge ``Q = R*A`` drives ``E = K Q^m S^n`` (climate-driven incision).
    ``sea_mask`` (cells at/below ``sea_level``) pins the ocean as a fixed base-level outlet so the
    land erodes toward the coast without the sea uplifting or the coastline being reworked.

    ``k_field`` (spatial erodibility) and ``uplift_field`` (spatial uplift) are optional per-cell
    arrays (same 2-D shape as ``height_m``); when given they REPLACE the scalar ``k_sp`` / ``uplift``
    so softer rock erodes faster and orogenic belts uplift more. ``None`` => the scalar (unchanged
    behaviour).
    """
    from landlab.components import FastscapeEroder, LinearDiffuser

    seed = height_m.astype(np.float64)
    if rainfall is None:
        runoff = np.ones(seed.size, dtype=np.float64)
    else:
        runoff = np.asarray(rainfall, dtype=np.float64).ravel()

    g, z = _new_grid(seed, cell_m, sea_mask=sea_mask, sea_level=sea_level)
    router, router_name = _make_router(g, runoff, flow_metric=flow_metric)

    # Spatial erodibility: FastscapeEroder accepts a per-node K array. Register it as a
    # grid field (the portable form across landlab versions) and pass by name.
    if k_field is not None:
        k_arr = np.asarray(k_field, dtype=np.float64).ravel()
        if k_arr.size != z.size:
            raise ValueError(f"k_field size {k_arr.size} != node count {z.size}")
        if "K_sp_field" in g.at_node:
            g.at_node["K_sp_field"][:] = k_arr
        else:
            g.add_field("K_sp_field", k_arr, at="node")
        k_arg = g.at_node["K_sp_field"]
    else:
        k_arg = k_sp
    ld = LinearDiffuser(g, linear_diffusivity=diffusivity)
    core = g.core_nodes

    # Spatial uplift: per-node uplift rate applied to core nodes each step.
    u_arr = None
    if uplift_field is not None:
        u_arr = np.asarray(uplift_field, dtype=np.float64).ravel()
        if u_arr.size != z.size:
            raise ValueError(f"uplift_field size {u_arr.size} != node count {z.size}")

    sed_mean = sed_max = None
    if enable_deposition:
        # Transport-limited fluvial erosion + DEPOSITION (SPACE): tracks a sediment layer over
        # bedrock and lays alluvium down where transport capacity drops -- valley floors,
        # lowlands, and approaching base level -- so the section grows flat depositional land
        # instead of incising canyons everywhere. ``depo_v_s`` (settling velocity) is the main
        # deposition lever; ``K_sed``/``K_br`` default to the stream-power ``k_sp``/``k_field``.
        from landlab.components import SpaceLargeScaleEroder
        if "soil__depth" not in g.at_node:
            g.add_zeros("soil__depth", at="node")
        if "bedrock__elevation" not in g.at_node:
            g.add_zeros("bedrock__elevation", at="node")
        soil = g.at_node["soil__depth"]
        bedrock = g.at_node["bedrock__elevation"]
        soil[:] = 0.0
        bedrock[:] = z - soil  # bedrock + soil == topographic, the invariant SPACE maintains
        space = SpaceLargeScaleEroder(
            g, K_sed=(k_arg if depo_k_sed is None else depo_k_sed),
            K_br=(k_arg if depo_k_br is None else depo_k_br),
            F_f=float(depo_f_f), phi=float(depo_phi), H_star=float(depo_h_star_m),
            v_s=float(depo_v_s), m_sp=m_sp, n_sp=n_sp,
            discharge_field="surface_water__discharge",
        )
        # SPACE is an EXPLICIT scheme: unlike the implicit FastscapeEroder it has a CFL limit, and
        # at Project-R's dt + large discharges the bedrock-incision term overshoots into spurious
        # deep pits and giant sediment mounds. Sub-step each year-step on a CFL estimate of the
        # incision-wave speed (K Q^m S^{n-1}); re-route between sub-steps so flow follows the
        # evolving surface. ``kmax`` bounds the rate when K is a spatial field.
        disch = g.at_node["surface_water__discharge"]
        slope = g.at_node["topographic__steepest_slope"]
        kmax = float(np.max(k_arg)) if isinstance(k_arg, np.ndarray) else float(k_arg)
        sub_max = 1
        t0 = time.perf_counter()
        for _ in range(int(steps)):
            bedrock[core] += (u_arr[core] if u_arr is not None else uplift) * dt
            z[:] = bedrock + soil
            router.run_one_step()
            rate = kmax * np.power(np.maximum(disch, 0.0), m_sp) \
                * np.power(np.maximum(slope, 1e-6), max(n_sp - 1.0, 0.0))
            rmax = float(rate.max()) if rate.size else 0.0
            nsub = int(np.clip(np.ceil(dt / (0.2 * cell_m / rmax)) if rmax > 0.0 else 1, 1, 64))
            sub_max = max(sub_max, nsub)
            sub_dt = dt / nsub
            for k in range(nsub):
                if k > 0:
                    router.run_one_step()
                space.run_one_step(sub_dt)
            ld.run_one_step(dt)
            # Hillslope creep changed the SURFACE; split it back into bedrock + soil for the next
            # SPACE step. Where creep cut below the (exhausted) regolith, lower BEDROCK too so
            # bedrock-cored ridges still round off; elsewhere the change lands in the sediment
            # layer. ``z`` stays the authoritative surface, so soil/bedrock/topo stay consistent.
            np.minimum(bedrock, z, out=bedrock)
            soil[:] = z - bedrock
        secs = time.perf_counter() - t0
        sed_mean = round(float(soil[core].mean()), 3)
        sed_max = round(float(soil[core].max()), 1)
    else:
        sp = FastscapeEroder(g, K_sp=k_arg, m_sp=m_sp, n_sp=n_sp,
                             discharge_field="surface_water__discharge")
        t0 = time.perf_counter()
        for _ in range(int(steps)):
            z[core] += (u_arr[core] if u_arr is not None else uplift) * dt
            router.run_one_step()
            sp.run_one_step(dt)
            ld.run_one_step(dt)
        secs = time.perf_counter() - t0

    info = {"router": router_name, "secs": round(secs, 2), "steps": int(steps),
            "spatial_k": k_field is not None, "spatial_uplift": uplift_field is not None,
            "deposition": bool(enable_deposition)}
    if enable_deposition:
        info["sed_mean_m"] = sed_mean
        info["sed_max_m"] = sed_max
        info["sub_steps_max"] = sub_max
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


# ---------------------------------------------------------------------------
# Glacial erosion (Hergarten stream-power law) -> U-troughs, over-deepening, fjords
# ---------------------------------------------------------------------------

def glacial_erode(
    height_m: np.ndarray,
    cell_m: float,
    *,
    ela_m: float,
    full_glac_m: Optional[float] = None,
    precip: Optional[np.ndarray] = None,
    ice_frac: float = 0.1,
    ablation: float = 4.0,
    k_g: float = 1.9e-5,
    m_g: float = 0.5,
    n_g: float = 1.0,
    alpha: float = 0.30,
    psi: float = 3.0,
    h_scale: float = 0.4,
    quarry_mult: float = 1.0,
    quarry_step_m: float = 30.0,
    diffuse: float = 0.3,
    max_incise_m: float = 30.0,
    flux_min: float = 0.0,
    dt: float = 2000.0,
    steps: int = 120,
    reroute_every: int = 4,
    snout_clamp: bool = True,
    flow_metric: str = "D8",
    sea_mask: Optional[np.ndarray] = None,
    sea_level: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Glacial stream-power erosion (Hergarten 2021 ESurf / 2023 GMD) -> over-deepened
    U-troughs and fjords. Pure (no bpy).

    Returns ``(z_bedrock, ice, info)``. The heightmap ``z_bedrock`` is the eroded BEDROCK
    only -- ice is never added to it, so it carries the erosion effects alone. ``ice`` is a
    dict of separate fields the caller can save/overlay independently (e.g. add the ice in
    Gaea2): ``thickness`` (metres of ice), ``surface`` (bedrock + ice = ice-surface
    elevation), ``mask`` (glaciated extent, bool), ``flux`` (the ice flux A_i).

    Law (Hergarten eq. 19, exponents identical to fluvial):

        E = K_g * A_i^m * S^n          m = 0.5, n = 1

    Three things make it glacial rather than fluvial, and each one is load-bearing:

    - **A_i is ICE flux, not water discharge.** Per-node ice production
      ``p_i = ice_frac * precip * clip((z - H_e)/(H_f - H_e), -ablation, 1)`` (H_e = ELA,
      H_f = full-glaciation altitude) is accumulated down-flow and clamped at 0 along the
      flow stack, so the glacier terminates at its snout. Flux peaks in the trunk near the
      coast -- which is exactly where fjords cut deepest.

    - **S is the slope of the ICE SURFACE, not the bed.** Ice thickness follows the
      paper-faithful closure ``h ~ A_i^((1-alpha)/psi) / S_bed`` (alpha=0.30, psi=3 ->
      exponent ~0.233). The smooth ice surface keeps sloping downflow even over a reverse-
      sloped bed, so erosion continues INTO closed over-deepenings and carves BELOW sea
      level -> fjords once flooded. (Pure bed-slope stream power is base-level limited and
      cannot do this, which is why this carves explicitly instead of via FastscapeEroder.)

    - **A quarrying term is added** (keyed to bed-riser height): abrasion alone under-cuts
      real fjords; quarrying of bedrock steps does the deepening.

    ``K_g`` (= 19 Ma^-1 in the GMD calibration) and ``h_scale`` are the two knobs to
    calibrate to your units/relief -- ``h_scale`` so trunk ice lands in the 100s of metres,
    ``K_g`` linearly to a target trough depth (``info['overdeepening_m']``). Concavity should
    land at the glacial ``theta_g ~= 0.47``, scorable with the existing slope-area metric.
    """
    z = np.asarray(height_m, dtype=np.float64).copy()
    H, W = z.shape
    H_f = float(full_glac_m) if full_glac_m is not None else float(ela_m) + 500.0
    denom = max(H_f - float(ela_m), 1.0)
    p = (np.ones_like(z) if precip is None
         else np.asarray(precip, dtype=np.float64).reshape(z.shape))
    sea = (np.zeros_like(z, dtype=bool) if sea_mask is None
           else np.asarray(sea_mask, dtype=bool).reshape(z.shape))
    cell_area = float(cell_m) ** 2
    s_min = 1e-4
    h_exp = (1.0 - float(alpha)) / float(psi)   # paper-faithful thickness exponent (~0.233)

    has_sea = bool(sea.any()) and not bool(sea.all())
    g, zg = _new_grid(z, cell_m, sea_mask=(sea if has_sea else None), sea_level=sea_level)
    router, rname = _make_router(g, np.ones(z.size, dtype=np.float64), flow_metric=flow_metric)
    sea_flat = sea.ravel()
    rec = stack = None

    def _route(bed):
        """Route ice over ``bed``; (re)populate the receiver + flow-stack node arrays."""
        nonlocal rec, stack
        zg[:] = bed.ravel()
        if has_sea:
            zg[sea_flat] = float(sea_level)     # ocean = flat base-level outlet for ice
        router.run_one_step()                   # fills pits FOR ROUTING ONLY; the bed keeps them
        rec = np.asarray(g.at_node["flow__receiver_node"])
        stack = np.asarray(g.at_node["flow__upstream_node_order"])

    def _ice_flux(bed):
        """ELA-gated ice production accumulated down-flow, clamped at 0 (glacier snout)."""
        prod = (ice_frac * p.ravel()
                * np.clip((bed.ravel() - float(ela_m)) / denom, -float(ablation), 1.0) * cell_area)
        if snout_clamp:
            for nd in stack[::-1]:              # upstream -> downstream (stack is d/s -> u/s)
                v = prod[nd]
                if v < 0.0:
                    v = prod[nd] = 0.0          # ablation can't yield negative ice
                r = rec[nd]
                if r != nd:
                    prod[r] += v
        else:
            np.maximum(prod, 0.0, out=prod)
            for nd in stack[::-1]:
                r = rec[nd]
                if r != nd:
                    prod[r] += prod[nd]
        return np.maximum(prod, 0.0).reshape(H, W)

    def _thickness(A_i, bed):
        """Ice thickness from the closure h ~ A_i^h_exp / S_bed. The slope is floored
        (~0.5deg) so the flat floors the glacier itself carves don't blow up h; the result is
        smoothed because the ice SURFACE is smooth (that smoothness is what carves fjords)."""
        S_bed = np.maximum(d8_slope(bed, cell_m), 0.01)
        h = np.where(A_i > float(flux_min), h_scale * np.power(A_i, h_exp) / S_bed, 0.0)
        return _gaussian_blur(h, sigma=2.0)

    t0 = time.perf_counter()
    for it in range(int(steps)):
        if it % max(int(reroute_every), 1) == 0:
            _route(z)
        A_i = _ice_flux(z)
        glac = A_i > float(flux_min)

        # ICE-SURFACE slope is the fjord-making term: the smooth ice surface keeps sloping
        # over a reverse-sloped bed, so erosion cuts BELOW base level into over-deepenings.
        h_ice = _thickness(A_i, z)
        S_srf = np.maximum(d8_slope(z + h_ice, cell_m), s_min)

        # Abrasion (Hergarten stream power) + quarrying of bed risers. EXPLICIT subtraction
        # (not FastscapeEroder) so it can excavate closed over-deepenings below base level.
        abrasion = k_g * np.power(A_i, m_g) * np.power(S_srf, n_g)
        riser = z - _minimum_filter(z, size=3)
        quarry = (k_g * quarry_mult * np.power(A_i, m_g)
                  * np.clip(riser / max(quarry_step_m, 1e-6), 0.0, None))
        dz = np.minimum((abrasion + quarry) * dt, max_incise_m)
        z = z - np.where(glac & ~sea, dz, 0.0)      # BEDROCK only -- ice is never added here

        # Lateral smoothing under thick ice -> U-shaped (not V-shaped) cross-section.
        if diffuse > 0.0 and glac.any():
            ref = float(np.percentile(h_ice[glac], 90)) + 1e-6
            wt = np.clip(diffuse * (h_ice / ref), 0.0, 0.7)
            z = np.where(glac & ~sea, (1.0 - wt) * z + wt * _gaussian_blur(z, sigma=1.0), z)

    # Final ice state CONSISTENT with the returned (eroded) bed, so the separate ice
    # outputs line up exactly with the bedrock heightmap.
    _route(z)
    A_i = _ice_flux(z)
    h_ice = _thickness(A_i, z)
    glac = A_i > float(flux_min)
    secs = time.perf_counter() - t0

    below = (z < float(sea_level)) & ~sea
    info = dict(
        router=rname, secs=round(secs, 2), steps=int(steps),
        glaciated_frac=round(float(glac.mean()), 3),
        overdeepening_m=round(float((float(sea_level) - z[below]).max()), 1) if below.any() else 0.0,
        ela_m=round(float(ela_m), 1), h_max_m=round(float(h_ice.max()), 1),
    )
    ice = dict(
        thickness=h_ice.astype(np.float32),         # metres of ice (the separate ice layer)
        surface=(z + h_ice).astype(np.float32),     # bedrock + ice = ice-surface elevation
        mask=glac,                                  # glaciated extent (bool)
        flux=A_i.astype(np.float32),                # ice flux A_i
    )
    return z.astype(np.float32), ice, info


# ---------------------------------------------------------------------------
# Sea floor / bathymetry -> continental shelf, slope, abyssal plain (+ keeps fjords)
# ---------------------------------------------------------------------------

def _coast_distance(sea_mask: np.ndarray, land_elev: np.ndarray, cell_m: float,
                    max_iter: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """For every cell, distance (m) to the nearest LAND cell and that land cell's elevation.

    scipy's exact EDT (with feature indices) when available; otherwise a capped iterative
    8-neighbour expansion (good enough for previews -- the bathymetry profile is smooth in
    distance anyway). ``land_elev`` supplies the elevation carried out from each coast."""
    try:
        from scipy.ndimage import distance_transform_edt
        dist_px, (iy, ix) = distance_transform_edt(
            sea_mask, return_distances=True, return_indices=True)
        lev_near = land_elev[iy, ix]
        return (dist_px * float(cell_m)).astype(np.float64), lev_near.astype(np.float64)
    except Exception:
        H, W = sea_mask.shape
        INF = np.float64(max_iter + 2)
        dist = np.where(sea_mask, INF, 0.0).astype(np.float64)      # in CELLS for the chamfer
        lev = np.where(sea_mask, 0.0, land_elev).astype(np.float64)
        d1, d2 = 1.0, np.sqrt(2.0)
        offs = [(-1, -1, d2), (-1, 0, d1), (-1, 1, d2), (0, -1, d1),
                (0, 1, d1), (1, -1, d2), (1, 0, d1), (1, 1, d2)]
        for _ in range(int(max_iter)):
            changed = False
            for dy, dx, w in offs:
                cand = np.roll(np.roll(dist, dy, axis=0), dx, axis=1) + w
                src = np.roll(np.roll(lev, dy, axis=0), dx, axis=1)
                better = cand < dist
                if better.any():
                    dist = np.where(better, cand, dist)
                    lev = np.where(better, src, lev)
                    changed = True
            if not changed:
                break
        dist = np.where(np.isfinite(dist), dist, INF)
        return dist * float(cell_m), lev


def seafloor_bathymetry(
    height_m: np.ndarray,
    cell_m: float,
    *,
    sea_level: float = 0.0,
    shelf_depth_m: float = 130.0,
    shelf_width_km: float = 60.0,
    shelf_relief_mod: float = 0.7,
    relief_window_km: float = 15.0,
    slope_width_km: float = 40.0,
    floor_depth_m: float = 6000.0,
    relief_ref_m: Optional[float] = None,
    input_depth: Optional[np.ndarray] = None,
    input_weight: float = 1.0,
    smooth: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    """Fill the ocean (cells <= ``sea_level``) with a realistic continental margin and return
    ``(z, info)``. Pure (no bpy). Land is untouched; only sea cells are rewritten.

    Profile vs distance from the coast (metres):

    - **Continental shelf** -- a gentle CONCAVE ``depth = shelf_depth * (d / shelf_w)^(2/3)``
      (Dean's equilibrium form) from 0 at the shoreline to ``shelf_depth_m`` at the shelf break.
    - **Continental slope** -- a steeper smootherstep drop from ``shelf_depth_m`` to
      ``floor_depth_m`` across ``slope_width_km``.
    - **Abyssal plain** -- the flat ``floor_depth_m`` beyond.

    The shelf WIDTH is narrowed where the bordering land is high/steep (active-margin look:
    mountains plunge into deep water) and stays broad off lowlands (passive margin), driven by
    the nearest coast's elevation relative to ``relief_ref_m`` (default = p98 of land relief).

    Crucially the result is unioned (deeper-of) with whatever was already carved below sea level
    -- so the **glacial fjords keep their over-deepening**, appearing as deep troughs incised into
    the shelf (with the shallow sill the glacial snout-taper already left at the mouth).

    ``floor_depth_m`` should equal the WORLD ocean-floor depth used for the export datum, so the
    deepest brightness (0) is consistent across sections. An optional ``input_depth`` (a [0..1]
    crop) supplies a hand-painted / real bathymetry, blended in by ``input_weight`` and still
    unioned with the carve."""
    z = np.asarray(height_m, dtype=np.float64).copy()
    H, W = z.shape
    sea = z <= float(sea_level)
    land = ~sea
    if not sea.any():
        return z.astype(np.float32), dict(ocean_frac=0.0, deepest_m=0.0, note="no ocean")

    # "Hinterland relief": the average LAND height in a window inland (ocean excluded), so the
    # shelf width responds to the MOUNTAINS BEHIND the coast -- not the waterline cell, which is
    # ~sea level by definition. Carried out to each ocean cell via its nearest coast (EDT).
    radius_px = max(float(relief_window_km) * 1000.0 / float(cell_m), 1.0)
    wsum = _box_mean(land.astype(np.float64), radius_px)        # fraction of the window that is land
    hsum = _box_mean(np.where(land, z, 0.0), radius_px)         # mean land-height contribution
    hinterland = np.where(wsum > 1e-6, hsum / np.maximum(wsum, 1e-9), float(sea_level))
    dist_m, lev_near = _coast_distance(sea, hinterland, cell_m)

    # Shelf width modulated by that hinterland height (mountainous coast -> narrow shelf).
    if relief_ref_m is not None:
        relief_ref = max(float(relief_ref_m), 1.0)
    else:
        relief_ref = max(float(np.percentile(hinterland[land], 95)) - float(sea_level), 1.0) if land.any() else 1.0
    relief_norm = np.clip((lev_near - float(sea_level)) / relief_ref, 0.0, 1.0)
    base_w = float(shelf_width_km) * 1000.0
    shelf_w = np.maximum(base_w * (1.0 - float(shelf_relief_mod) * relief_norm), base_w * 0.15)
    slope_w = max(float(slope_width_km) * 1000.0, float(cell_m))

    # Piecewise depth(distance): concave shelf -> smootherstep slope -> flat abyssal floor.
    frac = np.clip(dist_m / np.maximum(shelf_w, float(cell_m)), 0.0, 1.0)
    depth = float(shelf_depth_m) * np.power(frac, 2.0 / 3.0)
    in_slope = dist_m > shelf_w
    t = np.clip((dist_m - shelf_w) / slope_w, 0.0, 1.0)
    smoothstep = t * t * (3.0 - 2.0 * t)
    depth = np.where(in_slope, float(shelf_depth_m)
                     + (float(floor_depth_m) - float(shelf_depth_m)) * smoothstep, depth)

    # Optional direct bathymetry input (a [0..1] depth crop) blended over the procedural floor.
    if input_depth is not None:
        d_in = np.clip(np.asarray(input_depth, dtype=np.float64).reshape(z.shape), 0.0, 1.0) * float(floor_depth_m)
        w = float(np.clip(input_weight, 0.0, 1.0))
        depth = (1.0 - w) * depth + w * d_in

    z_bathy = float(sea_level) - depth
    z_union = np.minimum(z_bathy, z)            # keep the DEEPER of shelf vs existing carve (fjords)

    out = z.copy()
    out[sea] = z_union[sea]
    if smooth > 0.0:
        # Blur ocean toward sea_level at the edge (natural shoaling) without bleeding land heights in.
        tmp = np.where(sea, out, float(sea_level))
        out = np.where(sea, _gaussian_blur(tmp, sigma=float(smooth)), out)

    deepest = float(float(sea_level) - out[sea].min())
    info = dict(
        ocean_frac=round(float(sea.mean()), 3),
        deepest_m=round(deepest, 1),
        shelf_depth_m=round(float(shelf_depth_m), 1),
        floor_depth_m=round(float(floor_depth_m), 1),
        shelf_w_km=[round(float(shelf_w.min()) / 1000.0, 1), round(float(shelf_w.max()) / 1000.0, 1)],
    )
    return out.astype(np.float32), info


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
    erodibility_norm: Optional[np.ndarray] = None,
    erodibility_contrast: float = 1.0,
    uplift_norm: Optional[np.ndarray] = None,
    uplift_influence: float = 0.0,
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
    overlay_flow_metric: str = "Quinn",
    enable_glacial: bool = False,
    glacial_ela_m: Optional[float] = None,
    glacial_full_glac_m: Optional[float] = None,
    glacial_k_g: float = 1.9e-5,
    glacial_quarry_mult: float = 1.0,
    glacial_diffuse: float = 0.3,
    glacial_steps: int = 120,
    glacial_ice_out: Optional[Dict] = None,
    enable_coastal: bool = False,
    coastal_steps: int = 25,
    coastal_rate_m: float = 3.0,
    coastal_notch_m: float = 20.0,
    coastal_max_fetch_km: float = 25.0,
    coastal_swell_deg: Optional[float] = None,
    coastal_swell_focus: float = 0.0,
    coastal_talus_deg: float = 0.0,
    coastal_deposition: bool = True,
    enable_deposition: bool = False,
    depo_v_s: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    """Full per-section pipeline: (optional coastal wave reworking) -> condition seed -> rainfall
    -> LEM -> (optional light Wilbur overlay) -> linear peak rescale. Returns
    ``(eroded_metres, metrics_dict)``. Pure: callable from the Blender operator and from a
    headless validation harness alike.

    When ``enable_glacial``, a glacial carve runs EVEN BEFORE the coastal pass: it over-deepens
    U-troughs below sea level so the coast/coastal pass and LEM all see fjords. The coastal pass
    (when ``enable_coastal``) then reshapes the baseline, so everything downstream -- ``sea_mask``,
    seed conditioning, the ocean restore -- sees the reworked coast. The operator must apply the
    SAME glacial+coastal passes to its blend baseline (see erode_ops), or the ocean restore /
    strength blend silently undoes the structural coastline changes.
    """
    height_m = np.asarray(height_m, dtype=np.float32)

    glacial_info = None
    if enable_glacial:
        glac_sea = height_m <= float(sea_level_m)
        glac_land = ~glac_sea
        if glacial_ela_m is None:
            hi = float(np.percentile(height_m[glac_land], 98)) if glac_land.any() else float(height_m.max())
            ela = float(sea_level_m) + 0.35 * (hi - float(sea_level_m))   # default ELA: 35% up the relief
        else:
            ela = float(glacial_ela_m)
        height_m, _glac_ice, glacial_info = glacial_erode(
            height_m, cell_m, ela_m=ela, full_glac_m=glacial_full_glac_m,
            k_g=glacial_k_g, quarry_mult=glacial_quarry_mult, diffuse=glacial_diffuse,
            steps=glacial_steps, flow_metric=flow_metric,
            sea_mask=glac_sea if (glac_sea.any() and not glac_sea.all()) else None,
            sea_level=float(sea_level_m),
        )
        # The heightmap (height_m) carries on as eroded BEDROCK only; the ice is handed back
        # separately via glacial_ice_out so the operator can save it as its own layer.
        if glacial_ice_out is not None:
            glacial_ice_out.update(_glac_ice)

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

    # Relief over land drives the (now relative) conditioning-noise amplitude, so the
    # Noise Amount is a real lever regardless of the terrain's absolute height.
    land0 = ~sea_mask
    if land0.any():
        land_vals = height_m[land0]
        relief_m = float(np.percentile(land_vals, 99) - max(float(sea_level_m), float(np.percentile(land_vals, 1))))
    else:
        relief_m = float(height_m.max() - height_m.min())

    seed = condition_seed(height_m, noise_kind, noise_amp, seed=noise_seed, relief_m=relief_m)
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

    # --- Optional spatial drivers (normalized [0,1] crops) -> per-cell K / U fields ---
    # Both are stretched to [0,1] over LAND (robust p2..p98) so the knob ranges behave
    # regardless of the map's absolute brightness; ocean cells don't evolve so their
    # values are irrelevant.
    land_for_norm = ~sea_mask
    k_field = None
    if erodibility_norm is not None and float(erodibility_contrast) > 1.0001:
        en = _stretch01(np.asarray(erodibility_norm, dtype=np.float64).reshape(height_m.shape), land_for_norm)
        c = float(erodibility_contrast)
        # norm 0.5 -> Kx1 (neutral); 1 -> Kxc (softest, erodes faster); 0 -> K/c (hardest).
        k_field = (float(k_sp) * np.power(c, 2.0 * en - 1.0)).astype(np.float64)
    uplift_field = None
    if uplift_norm is not None and float(uplift_influence) > 1e-6:
        un = _stretch01(np.asarray(uplift_norm, dtype=np.float64).reshape(height_m.shape), land_for_norm)
        infl = float(np.clip(uplift_influence, 0.0, 1.0))
        uplift_field = (float(uplift) * ((1.0 - infl) + infl * un)).astype(np.float64)

    z, dr, lem_info = lem_erode(
        seed, cell_m, rainfall=rain_field, k_sp=k_sp, m_sp=m_sp, n_sp=n_sp,
        diffusivity=diffusivity, uplift=uplift, dt=dt, steps=steps, flow_metric=flow_metric,
        sea_mask=sea_mask if has_sea else None, sea_level=float(sea_level_m),
        k_field=k_field, uplift_field=uplift_field,
        enable_deposition=enable_deposition, depo_v_s=depo_v_s,
    )

    overlay_secs = None
    if enable_overlay:
        # The overlay carve uses discharge MAGNITUDE (not FastscapeEroder's single-flow
        # receivers), so it can use a MULTI-FLOW router (e.g. Quinn) that the D8 LEM
        # cannot -- this is what lets the engraved channels meander off the grid instead
        # of snapping to the 8 D8 directions.
        z, dr, overlay_secs = wilbur_overlay(
            z, cell_m, tile_km, res_px, rainfall=rain_field,
            depth_macro_m=overlay_depth_m, w_macro_km=overlay_w_macro_km, r=overlay_r,
            skip_macro=True, base=base, flow_metric=overlay_flow_metric,
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
        deposition=lem_info.get("deposition"),
        sed_mean_m=lem_info.get("sed_mean_m"),
        sed_max_m=lem_info.get("sed_max_m"),
        depo_sub_steps_max=lem_info.get("sub_steps_max"),
        overlay_secs=round(overlay_secs, 2) if overlay_secs is not None else None,
        coastal_secs=coastal_info.get("secs") if coastal_info else None,
        coastal_conserved_frac=coastal_info.get("conserved_frac") if coastal_info else None,
        glacial_secs=glacial_info.get("secs") if glacial_info else None,
        glacial_overdeepening_m=glacial_info.get("overdeepening_m") if glacial_info else None,
        glacial_glaciated_frac=glacial_info.get("glaciated_frac") if glacial_info else None,
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
