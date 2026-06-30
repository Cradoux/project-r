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


def _new_grid(seed_m: np.ndarray, cell_m: float):
    """Build a RasterModelGrid seeded with ``topographic__elevation`` (open edges)."""
    from landlab import RasterModelGrid
    size_y, size_x = seed_m.shape
    g = RasterModelGrid((int(size_y), int(size_x)), xy_spacing=float(cell_m))
    z = g.add_zeros("topographic__elevation", at="node")
    z[:] = seed_m.astype(np.float64).ravel()
    g.set_closed_boundaries_at_grid_edges(False, False, False, False)  # all edges open
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
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Evolve ``height_m`` under stream-power incision + linear hillslope diffusion.

    Returns ``(z_metres, drainage_area_m2, info)``. ``rainfall`` is an optional per-node runoff
    multiplier (mean ~1); when None a uniform field is used, making ``E = K A^m S^n``. With a
    rainfall field the discharge ``Q = R*A`` drives ``E = K Q^m S^n`` (climate-driven incision).
    """
    from landlab.components import FastscapeEroder, LinearDiffuser

    seed = height_m.astype(np.float64)
    if rainfall is None:
        runoff = np.ones(seed.size, dtype=np.float64)
    else:
        runoff = np.asarray(rainfall, dtype=np.float64).ravel()

    g, z = _new_grid(seed, cell_m)
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
    flow_metric: str = "D8",
) -> Tuple[np.ndarray, Dict]:
    """Full per-section pipeline: condition seed -> rainfall -> LEM -> (optional light Wilbur
    overlay) -> linear peak rescale. Returns ``(eroded_metres, metrics_dict)``. Pure: callable
    from the Blender operator and from a headless validation harness alike.
    """
    height_m = np.asarray(height_m, dtype=np.float32)

    seed = condition_seed(height_m, noise_kind, noise_amp, seed=noise_seed)
    rainfall = climate_field(climate_kind, height_m.shape, strength=climate_strength) \
        if climate_kind != "uniform" else None

    z, dr, lem_info = lem_erode(
        seed, cell_m, rainfall=rainfall, k_sp=k_sp, m_sp=m_sp, n_sp=n_sp,
        diffusivity=diffusivity, uplift=uplift, dt=dt, steps=steps, flow_metric=flow_metric,
    )

    overlay_secs = None
    if enable_overlay:
        z, dr, overlay_secs = wilbur_overlay(
            z, cell_m, tile_km, res_px, rainfall=rainfall,
            depth_macro_m=overlay_depth_m, w_macro_km=overlay_w_macro_km, r=overlay_r,
            skip_macro=True, base=base, flow_metric=flow_metric,
        )

    metrics = score(z, dr, cell_m)

    if target_peak_m is not None:
        z = rescale_peak(z, float(target_peak_m), base=base)

    metrics.update(
        router=lem_info.get("router"),
        lem_secs=lem_info.get("secs"),
        overlay_secs=round(overlay_secs, 2) if overlay_secs is not None else None,
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
