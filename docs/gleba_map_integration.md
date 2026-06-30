# Gleba Export Set — Project-R Integration Assessment

42 standard maps from the "Gleba" planet generator, all 4096×2048 equirectangular (exact 2:1), pixel-aligned. Confirmed by the analyzer on every entry.

## 1. Executive summary

- **The export is a complete, pixel-aligned 4096×2048 equirect stack.** Every map shares the same grid as the heightmap, so any field can be cropped/reprojected through the existing section pipeline with no re-registration. This is the single most important fact: spatial drivers are "free" to add once decoded.
- **Three maps drop in TODAY with zero or trivial decode:** `TileElevationGreyscaleBathymetry` (16-bit) → **Heightmap + Bathymetry in one file**; `TrueColor` (or `TileElevationColorSmooth*`) → **World Map**; any of the three viridis rainfall maps → **Rainfall** (needs a one-time viridis→scalar decode).
- **The heightmap question has a clear winner:** prefer `TileElevationGreyscaleBathymetry` (16-bit, 40k levels, carries sub-sea-level depth so it fills both heightmap and bathymetry). Use `TileElevationGreyscaleLand` only when you want a hard-zeroed ocean and a free land mask; `TileElevationGreyscale` is the land+flat-ocean middle option. All three are genuine 16-bit — **never** use the 8-bit colored `TileElevationColorSmooth*` variants for data (display only).
- **Rainfall: prefer `AverageRainfall`** (annual mean, clean viridis, high confidence) over `January`/`July` (seasonal, bias incision toward one hemisphere). `AverageHumidity` and `RainShadow` are weaker proxies — use only as fallbacks.
- **The highest-value NEW capability the rich maps unlock is a spatial UPLIFT field U:** `OrogenyStrength` (a near-ready viridis uplift map) plus `CombinedTectonics`/`TectonicPlates`/`Volcanism` let Project-R build persistent mountain belts at geologically correct locations instead of using a flat uplift constant. The second new capability is **spatial erodibility K_sp** from `RockType`/`GeologicProvince`. Both are extension slots not yet wired.
- **Land/sea masking is over-served:** at least 8 maps yield a clean binary coastline (`ClimateLandVsSea`, `CrustMap`, `DebugIsLand`, `Waterbodies`, `SinksAndSources`, etc.). One explicit mask slot would replace the current heightmap brightness-threshold guesswork.

## 2. Map catalog (grouped by Project-R role)

### World Map (display, sRGB load-as-is)
| Name | Enc (chan/bit) | Decode | Useful | Notes |
|---|---|---|---|---|
| **TrueColor** | RGB 8 | none (sRGB) | high | **RECOMMENDED default world map** — Blue-Marble style, most legible continents/deserts/ice. Also yields a land/sea mask. |
| TileElevationColorSmoothIceAge | RGB 8 | none | high | Best hypsometric variant; ice extents warn user off distorted polar sections. Pick if epoch = ice age. |
| TileElevationColorSmooth | RGB 8 | none | med | Hypsometric, no ice. Alternative to TrueColor for relief emphasis. |
| TileElevationColorSmoothNoIce | RGB 8 | none | med | Same, explicit no-ice epoch. |
| Koppen / Biome / GeographicRegion | RGB 8 cat | palette | low–med | Legible but unattractive as main texture; better as section-selection overlays (see below). |

**Recommendation:** `TrueColor` as default. Offer `TileElevationColorSmoothIceAge` as the "relief" alternative. **Use one epoch (ice vs no-ice) consistently across all slots.** Never invert the hypsometric color maps for elevation — use the grayscale sibling.

### Heightmap (16-bit grayscale → metres) — THE key input
| Name | Enc | Decode | Useful | Notes |
|---|---|---|---|---|
| **TileElevationGreyscaleBathymetry** | 1ch **16-bit** | v/65535 | high | **RECOMMENDED default.** 40k levels; carries land + ocean-floor depth → fills heightmap AND bathymetry. Sea level ≈ v 28672–32768 (calibrate). |
| TileElevationGreyscaleLand | 1ch 16-bit | v/65535 | high | Ocean hard-zeroed (77% zeros) → free land/sea mask via v>0. No bathymetry. Use when you want flat-zero ocean + clean shoreline. |
| TileElevationGreyscale | 1ch 16-bit | v/65535 | high | Land + ocean (ocean above 0, no deep bathymetry). Middle option. |

All three have thin ~1–2 px **black coastline-outline artifacts** that will seed spurious deep incision — median/dilate the shoreline before stream-power. **Physical elevation range is unlabeled** — user sets `max_elevation_m` (already exists, default 8849 m). Choose ONE as the canonical heightmap; the others are situational.

### Bathymetry (depth; white=deepest)
| Name | Enc | Decode | Useful | Notes |
|---|---|---|---|---|
| **TileElevationGreyscaleBathymetry** | 1ch 16-bit | (sea_level_v − v)/sea_level_v for v<sea_level | high | **Same file as heightmap** — extract sub-sea-level range, remap so white=deepest, feed sea-floor pass. One file, two slots. |

No dedicated standalone bathymetry export exists; this combined map is the source.

### Rainfall (single-channel runoff weight, optional)
| Name | Enc | Decode | Useful | Notes |
|---|---|---|---|---|
| **AverageRainfall** | RGB 8 viridis | LUT-invert → idx/255 | high | **RECOMMENDED default.** Annual mean, clean viridis (mean_err 0.036), direct fit. |
| JanuaryRainfall | RGB 8 viridis | LUT-invert | high | Seasonal; over-weights N-winter runoff. Use only for explicit seasonal modeling. |
| JulyRainfall | RGB 8 viridis | LUT-invert | high | Seasonal; over-weights ITCZ summer. Blend Jan+Jul (0.5/0.5) ≈ annual if no Average. |
| RainShadow | RGB 8 viridis | LUT-invert then **invert (1−s)** | med | Orographic dryness, not absolute rain; mask (30,30,30) ocean sentinel. Fallback/modulator only. |
| AverageHumidity | RGB 8 viridis* | LUT-invert | med | Humidity ≠ precip; custom-ish ramp (validate decode). Last-resort proxy. |

**Recommendation:** `AverageRainfall`. Decode once on load, cache as 16-bit single-channel in `source/`. Ocean pixels decode to nonzero but the heightmap sea-level mask suppresses erosion there.

### Erodibility K_sp (NEW extension slot)
| Name | Enc | Decode | Useful | Notes |
|---|---|---|---|---|
| **RockType** | RGB 8 cat (~15–20 true classes) | nearest-palette → class→K_sp LUT | med | **Best K_sp candidate.** Lithology blobs; drop fringe colors <0.5%. Ocean ≈ (2,3,33). Needs Gleba class legend for calibrated values. |
| GeologicProvince | RGB 8 cat (5 classes) | nearest-palette | med | Clean 5-class (craton/margin/shield/orogen/ocean); blocky but easy. Doubles as uplift + land mask. |
| SoilTexture | RGB 8 cat (~5 classes) | nearest-centroid | low | Surface erodibility not bedrock; weaker than RockType. |
| SoilDepth | RGB 8 viridis | LUT-invert | med | Deep soil → higher K_sp. Risk: correlates with relief (double-counts elevation). |
| Vegetation / SoilPermeability | RGB 8 ramp | HSV/LUT | low | 2nd-order modifiers (roots armor soil / infiltration reduces runoff). Indirect. |

**Recommendation:** `RockType` for spatial K_sp; `GeologicProvince` as the simpler 5-class alternative. **Nearest-neighbor resampling only** (never bilinear — it invents class colors).

### Uplift / Orogeny U (NEW extension slot)
| Name | Enc | Decode | Useful | Notes |
|---|---|---|---|---|
| **OrogenyStrength** | RGB 8 viridis (~105 levels) | LUT-invert; clamp [17,0,21] sentinel→0 | high | **RECOMMENDED default U.** Already a continuous uplift-intensity field; bright strands = active belts. Cmap ID slightly uncertain (try inferno/magma fallback). |
| CombinedTectonics | RGB 8 viridis + baked lines | LUT-invert; mask boundary lines + gray fill; 3×3 median inpaint | med | Tectonic intensity; needs line removal. Also a display layer. |
| TectonicPlates (16) | RGB 8 cat | palette → boundary distance-transform | med | Derive U from proximity-to-boundary. No boundary-type info. |
| TectonicMicroplates / OldTectonicPlates | RGB 8 cat (100s/90) | cluster → boundary DT | med | High boundary density; "Old" plates may not match current relief. |
| Volcanism | RGB 8 cat (10, mostly 2) | nearest-palette → 0/0.3/1.0 | med | Hotspot point markers; nearest-neighbor resample to preserve sparse dots. |

**Recommendation:** `OrogenyStrength` (ready-made continuous U). Use plate maps only for a boundary-proximity proxy if orogeny is unavailable.

### Land/Sea mask (NEW explicit slot — currently over-served)
| Name | Enc | Decode | Useful | Notes |
|---|---|---|---|---|
| **ClimateLandVsSea** | RGB 8 (=1ch) binary | R≥128 | med | **RECOMMENDED** — purpose-built binary, anti-aliased coast; threshold 0.5. |
| CrustMap | 1ch binary {0,255} | ==255 | med | Crust type (shelves may be white below sea level) — not exactly shoreline. |
| DebugIsLand | RGB 8 {30,80} | ≥55 | med | Clean binary; debug values. |
| Waterbodies / SinksAndSources / GeographicRegion / Biome | RGB 8 cat | class 0 = ocean | low–med | All yield ocean mask as byproduct. |
| TileElevationGreyscaleLand | 16-bit | v>0 | high | Free mask from chosen heightmap — **often makes a separate mask map unnecessary.** |

**Recommendation:** if heightmap is `...Land`, derive mask from it (v>0). Otherwise `ClimateLandVsSea`.

### Section-selection aids (overlay only)
GeographicRegion, Biome, Koppen, GeologicProvince (cyan orogens), TileSlope, InAndOutFlow (inflow basins), ErosionTimeWaterflow(Cutoff) (river skeleton), PopulationDensity, ClimateContinentality, SoilRichness, Volcanism (hotspots). All useful as "where to crop" overlays; none are erosion drivers.

### Informational-only / redundant for erosion
ClimateOceanCurrents, ClimateSurfaceWinds (wind, has baked direction arrows), GeographicArea (generator area-weight), ErosionTimeWaterflow/Cutoff (generator's own derived drainage — circular if fed back), InAndOutFlow, SinksAndSources (Project-R recomputes flow). **Do not feed any derived-flow map as rainfall** — it double-counts discharge.

## 3. Decoding guide

**(a) 16-bit grayscale → metres** (TileElevationGreyscale*, the authoritative data).
```
img = Image.open(path)            # PIL mode 'I;16' / 'I'
v   = np.asarray(img, np.float32) # 0..65535  — DO NOT convert to 8-bit first
elev_norm = v / 65535.0
elev_m    = elev_norm * max_elevation_m
```
Sea level: find the histogram valley (Bathymetry variant ≈ v 28672–32768; Land variant = exact 0). 16-bit gives 35k–40k levels — full precision. The 8-bit "top_colors / 99.89% white" readout is a PIL I→RGB scaling artifact; trust `n_unique_levels`.

**(b) Colormap ramp → scalar (LUT inversion).** Cmaps actually found: **viridis** (most ramps: Average/January/July Rainfall, AverageHumidity, OrogenyStrength, CombinedTectonics, SoilDepth, SoilPermeability, SoilRichness, RainShadow, TileSlope, ErosionTimeWaterflow, PopulationDensity), **cividis** (ClimateSurfaceWinds), and **custom hypsometric / jet-like** (TileElevationColor*, Vegetation — NOT invertible, display only).
```
lut = (matplotlib.colormaps['viridis'](np.linspace(0,1,256))[:, :3] * 255).astype(np.uint8)  # (256,3)
# mask the dark sentinel FIRST (viridis maps use (30,30,30) or (17,0,21), NOT viridis[0]=(68,1,84)):
ocean = np.all(rgb < 35, axis=-1)
d   = ((rgb[...,None,:].astype(int) - lut[None,None])**2).sum(-1)   # (...,256)
idx = d.argmin(-1)                                                  # 0..255
scalar = idx / 255.0                                               # [0,1] relative
scalar[ocean] = np.nan
```
Recovered range is **relative [0,1]** for every ramp — no legend gives physical units; rescale only if the user supplies a max. 8-bit → ≤256 levels (banding tolerable for soft fields, not for elevation). For viridis maps where the analyzer's auto-fit failed (Orogeny, SoilRichness, SoilPermeability), the sentinel dominates the fit — viridis is still correct; validate on a known-wet/known-peak pixel, fall back to inferno/magma for Orogeny only.

**(c) Categorical palette → class ids.**
```
# build palette: unique colors, drop those <0.5% of pixels (anti-alias fringe)
# assign each pixel to nearest palette entry by L2 in RGB; output integer class raster
```
Exact-match works where flat-filled (Biome, GeologicProvince 5-class, Volcanism 10-class). Cluster first (k-means/DBSCAN) where anti-aliasing inflates color count (TectonicMicroplates ~1711, SoilTexture ~29k, GeographicRegion ~7.7k). **Always resample categorical with nearest-neighbor.** Then class→scalar via a user-editable LUT (K_sp or U).

**(d) Truecolor → display only.** TrueColor, TileElevationColor*: load sRGB, use directly. Zero scalar recovery; do not invert.

## 4. Integration plan for Project-R (prioritized)

### Tier 1 — drop-in today
1. **Heightmap = `TileElevationGreyscaleBathymetry`** (16-bit). Wire into existing `heightmap_filename` slot, decode v/65535 × `max_elevation_m`. Set sea-level fraction from histogram valley (≈0.44). **Risk:** coastline-outline artifacts → add a shoreline median/dilate pre-pass before stream-power. If a flat-zero ocean is preferred, swap to `...Land`.
2. **Bathymetry = same file** (sub-sea-level range, remapped white=deepest) → existing `seafloor_bathy_filename` slot. **Risk:** sea-level threshold uncalibrated — cross-check against `...Land` sibling.
3. **World map = `TrueColor`** → preview sphere, sRGB, no decode.
4. **Rainfall = `AverageRainfall`**, viridis-decoded once on load to a 16-bit single-channel cached in `source/`, → existing `rainfall_filename` slot. **Decode work:** one LUT inversion. **Risk:** relative-only scale (fine for weighting).

These four require **no new pipeline** — only file selection plus a rainfall decode-on-load step.

### Tier 2 — high-value NEW spatial drivers
Data flow for both (mirrors heightmap): **global equirect PNG → decode-on-load to single-channel float cached in `source/` → cropped+Hammer-reprojected per section (same pass as heightmap, `interp_for_layer`) → normalized [0,1] field → modulates the LEM term per node.**

5. **Uplift U = `OrogenyStrength`.** Decode viridis→[0,1], clamp sentinel→0, mask ocean. In `erosion.py` set `U_eff(x,y) = U_base × orogeny(x,y)`. **Decode:** LUT inversion + sentinel handling. **Alignment:** native 4096×2048, crops with heightmap. **Risk:** cmap ID medium-confidence (105 levels); validate, fallback inferno/magma. New `uplift_filename` prop + UI control + erosion term.
6. **Erodibility K_sp = `RockType`** (or `GeologicProvince` for simplicity). Nearest-palette → class id → user-editable class→K_sp table → `K_sp(x,y) = K_base × factor[class]`. **Resample nearest-neighbor only.** Small Gaussian blur on the float K_sp field before the solver to avoid hard-edge numerical artifacts. **Risk:** no embedded legend → heuristic K_sp until Gleba class table obtained. New `erodibility_filename` prop + class-LUT UI.

### Tier 3 — nice-to-have / informational
7. **Explicit land/sea mask slot** (`ClimateLandVsSea`, or derive from `...Land` heightmap) → replaces brightness-threshold sea-level guesswork; feeds coastal/sea-floor passes. Low cost, high cleanliness.
8. **Section-selection overlays** (GeographicRegion / Biome / Koppen / TileSlope / Volcanism hotspots) on the preview sphere to guide where to crop. UX only, no data path.
9. Everything in "Informational-only" stays unwired (redundant or circular for erosion).

## 5. Optional "Map Loading" UI design

### Placement
Add **`PP_PT_inputs`** ("Map Inputs", `bl_parent_id="PP_PT_main"`, `bl_options={'DEFAULT_CLOSED'}`), sitting **above `PP_PT_section`** in the panel tree (`PP_PT_main > Map Inputs / Sphere / Section / Erosion / Reassembly`). **Consolidate:** move the existing ad-hoc `heightmap_filename` (Erosion panel), `rainfall_filename`, and `seafloor_bathy_filename` (Sea Floor sub-panel) pickers into this one panel so all map I/O lives in a single place. The current `max_elevation_m` / `ocean_floor_depth_m` scalars stay next to the heightmap slot.

### Slots (all optional — empty string = unused, mirroring `heightmap_filename`)
Each is a `StringProperty` holding a filename in `source/` (matching the existing convention, **not** a FILE-subtype absolute path), rendered as: label + filename + a file-browse operator + an "X" clear operator.

| Slot prop | Role | Decode-on-load | Color mgmt |
|---|---|---|---|
| `heightmap_filename` *(exists)* | Heightmap | none (16-bit linear) | linear |
| `bathymetry_filename` *(rename seafloor_bathy)* | Bathymetry | none (16-bit) | linear |
| `rainfall_filename` *(exists)* | Rainfall | **viridis→1ch** if RGB | linear |
| `world_map_filename` *(new)* | World/preview | none | **sRGB** |
| `landsea_mask_filename` *(new)* | Land/sea mask | binarize if needed | nearest categorical |
| `erodibility_filename` *(new)* | K_sp | **palette→class→1ch** | nearest categorical |
| `uplift_filename` *(new)* | Uplift U | **viridis→1ch** | linear |

### Auto-detection (point at a folder → pre-fill, every slot still overridable)
Extend the existing `layers.py` classifier (the single source of truth). Current keywords are too narrow — `HEIGHT_KEYWORDS=("height","elev","dem")` already catches `TileElevation*`; `RAINFALL_KEYWORDS=("rain","precip")` catches the rainfall trio. Add:

```python
# layers.py — extend the keyword table
BATHY_KEYWORDS     = ("bathy",)
WORLD_KEYWORDS     = ("truecolor", "color", "colour")
ERODIBILITY_KEYWORDS = ("rocktype", "geolog", "litholog", "soiltexture")
UPLIFT_KEYWORDS    = ("orogeny", "tectonic", "uplift")
# refine existing MASK to prefer the purpose-built one:
LANDSEA_KEYWORDS   = ("landvssea", "island", "crustmap", "landsea")
```
Add `is_bathy_name`, `is_world_name`, `is_erodibility_name`, `is_uplift_name`, `is_landsea_name` following the same pattern as `is_rainfall_name` (token-based, with precedence guards so `TileElevationGreyscaleBathymetry` resolves to **bathymetry+height** not just height, and `...Land` doesn't get grabbed as rainfall). Auto-fill picks, per slot, the **highest-priority filename match**, resolving the documented competitions:
- Heightmap: prefer `*Bathymetry` (16-bit, dual-purpose) > `*GreyscaleLand` > `*Greyscale`; **exclude `*Color*` from height** (they match "elev" but are 8-bit display — gate with `treat_as_color`).
- Rainfall: prefer `Average*` > `January`/`July`.
- World: prefer `TrueColor` > `TileElevationColorSmooth*` (and pick the ice/no-ice epoch matching a single addon `epoch` enum).

A "Detect from folder" operator scans `source/`, runs the classifiers, and fills empty slots only (never clobbers a user override).

### Decode-on-load vs used-as-is
- **Used as-is:** heightmap, bathymetry (16-bit linear); world map (sRGB).
- **Decode-on-load → cache single-channel 16-bit linear PNG in `source/`** (so the existing crop/reproject pipeline treats it like a heightmap, and `treat_as_color` returns False): rainfall (viridis), uplift (viridis), erodibility (palette→class index). Cache filename e.g. `_decoded_<role>.png` to keep the raw export untouched and make re-decode idempotent. `interp_for_layer` must return **nearest** for the erodibility/landsea cached layers (categorical) and **linear** for rainfall/uplift — extend `is_mask_name` coverage or add the new roles to the nearest-interp set so class IDs aren't interpolated at section edges.

### Faithfulness notes
- Keeps the manifest-layer model: decoded maps are written into `source/` and tracked like existing layers.
- Honors the single-channel-linear vs sRGB-color vs nearest-categorical contract already centralized in `layers.py` (`treat_as_color`, `interp_for_layer`).
- All slots optional, clearable, show their filename — same UX as `select_heightmap`/`select_rainfall`.

### Categorical → per-class B&W mask export (Biome / Koppen)  — Gaea downstream masks
**Motivation:** export each category of a palette map as a separate black/white mask so it can drive a
per-type mask in Gaea (e.g. apply a different surface/erosion treatment to each climate or biome).

**Validated prototype:** `split_categorical.py` (in this session's scratchpad) — proven on Biome and
Koppen against the real export:
- **Output style:** hard **8-bit, pure 0/255**, full-res 4096×2048, white = "this class". (Chosen over
  anti-aliased/16-bit.) Gaea-ready.
- **AA-snapping:** boundary speckle is assigned to the nearest dominant palette colour so each mask is a
  clean union with no fringe halo. Koppen: 28 raw colours → **22 real classes** at 99.87 % coverage; Biome:
  **13 classes** at 100 %.
- **Köppen auto-naming:** confirmed **Gleba uses the standard Köppen-Geiger palette** (class colours match
  the standard table at distance 0–1), so masks emit pre-named: `Koppen_mask_01_FF0000_BWh.png`,
  `..._ET.png`, `..._Dfc.png`, … No manual labelling needed. Generic categorical maps (Biome, RockType,
  plates…) name by index + hex; a name table can be layered on later.
- White/ocean background is detected and flagged (optionally skipped). A `*_palette.json` manifest records
  every class (rgb, hex, coverage %, Köppen guess).

**Scope:** Biome + Koppen for now (the script generalises to any categorical map listed in the catalog —
RockType, GeologicProvince, plates, Vegetation, Waterbodies — when we choose to extend it).

**Status:** IMPLEMENTED in the Map Inputs panel (`pp.export_class_masks`, GLOBAL + SECTION scope). The notes below
describe that design.

**UI / pipeline integration:** add an **"Export class masks"** action on the categorical layer slots
(Biome/Koppen) in the Map Inputs panel. Run it in **two modes**, reusing the same nearest-palette decode:
- **Global:** split the 4096×2048 equirect map → `source/<layer>_masks/*.png` + `*_palette.json`.
- **Per-section (the Gaea-relevant one):** when a section heightmap is exported, crop each categorical
  layer through the **same oblique Hammer projection + crop rect** as the heightmap (nearest-neighbour
  interp, per `interp_for_layer`) and split *that* into masks → `sections/<id>/masks/`. The masks then line
  up pixel-for-pixel with the section terrain you sculpt in Gaea.
- Port `split_categorical.py`'s logic into the addon (it's pure numpy/PIL, no bpy) so it reuses
  `layers.py`'s categorical handling rather than living as a standalone script.

## 6. Open questions / things to verify

1. **Physical scales unlabeled everywhere.** No legends/colorbars on any map. `max_elevation_m` (heightmap) and rainfall/uplift units are user-supplied or relative-only. Need the Gleba generator's elevation ceiling and any mm/yr rainfall scale to calibrate discharge.
2. **Sea-level value in `TileElevationGreyscaleBathymetry`** is inferred from the histogram valley (≈ v 28672–32768) — must be pinned by diffing against the `...Land` sibling (where ocean=0) before bathymetry extraction is trustworthy.
3. **Coastline-outline artifacts** (1–2 px black) in all grayscale elevation maps — confirm whether `...Land` lacks them; decide on the median/dilate shoreline pre-pass parameters.
4. **Ice-age vs no-ice epoch consistency.** `ClimateLandVsSea` and the `...IceAge` color map may reflect a different (lower) sea level than the default heightmap. Verify which epoch each data map belongs to and lock all slots to one epoch.
5. **Categorical legends are Gleba-internal and not embedded.** RockType (~15–20 classes), GeologicProvince (5), Biome (13), Koppen (28), Volcanism (10), SoilTexture (~5) — class→meaning (and thus K_sp/U values) needs the generator's source/enum. Until then K_sp/U mappings are heuristic.
6. **OrogenyStrength colormap identity** is medium-confidence (auto-fit returned nipy_spectral; viridis assumed). Validate decode on a known mountain belt; keep inferno/magma fallback.
7. **AverageHumidity custom cluster** — the (63,57,9) olive at 33% of pixels is anomalous for mid-viridis; the generator may use a non-standard ramp. Validate before trusting as rainfall.
8. **Alignment** is asserted as pixel-perfect 2:1 by the analyzer for all 42 maps but has not been visually cross-checked against the heightmap for any single map — spot-check one categorical (RockType) and one ramp (Orogeny) against the coastline before wiring spatial drivers.
9. **Redundancy pruning:** confirm we never feed a *derived* map (ErosionTimeWaterflow, InAndOutFlow, SinksAndSources, TileSlope) into an erosion driver slot — these are outputs of the generator's own model and would double-count or conflict with Project-R's stream-power computation.
