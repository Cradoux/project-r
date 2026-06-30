from __future__ import annotations

import bpy
from bpy.types import Panel

from . import deps


_CATEGORY = "Project-R"


class _PRPanel:
    """Shared panel placement (N-panel > Project-R tab)."""
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = _CATEGORY


# ---------------------------------------------------------------------------
# Root panel: dependency status + Project
# ---------------------------------------------------------------------------
class PP_PT_main(_PRPanel, Panel):
    bl_label = "Project-R"
    bl_idname = "PP_PT_main"

    def draw(self, context: bpy.types.Context) -> None:
        s = context.scene.projection_pasta
        layout = self.layout

        # Dependencies (cached checks; no heavy imports in draw).
        installing = deps.PP_OT_install_dependencies.is_running()
        missing = deps.missing_required()
        if missing or installing:
            box = layout.box()
            if installing:
                box.label(text="Installing dependencies... (see status bar)", icon="SORTTIME")
            else:
                box.alert = True
                box.label(text=f"Missing: {', '.join(missing)}", icon="ERROR")
            row = box.row()
            row.enabled = not installing
            row.scale_y = 1.2
            row.operator("pp.install_dependencies", text="Install Dependencies", icon="IMPORT")
            box.label(text="Installs in the background; no restart needed.")

        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(s, "project_root", text="Root")
        col.prop(s, "planet_radius_km", text="Planet Radius")
        col.prop(s, "max_elevation_m", text="Max Elevation")
        col.prop(s, "ocean_floor_depth_m", text="Ocean Floor Depth")

        mp = s.manifest_path()
        has_project = bool(mp and mp.exists())
        row = layout.row(align=True)
        if has_project:
            row.operator("pp.open_project", text="Reload Project", icon="FILE_REFRESH")
        else:
            row.operator("pp.init_project", text="Create Project", icon="NEWFOLDER")
        row.operator("pp.open_manifest", text="manifest.json", icon="TEXT")


# ---------------------------------------------------------------------------
# Map Inputs (optional, consolidated source maps + auto-detect + mask export)
# ---------------------------------------------------------------------------
class PP_PT_inputs(_PRPanel, Panel):
    bl_label = "Map Inputs"
    bl_idname = "PP_PT_inputs"
    bl_parent_id = "PP_PT_main"
    bl_order = 0
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context: bpy.types.Context) -> None:
        self.layout.label(text="", icon="IMAGE_DATA")

    def draw(self, context: bpy.types.Context) -> None:
        s = context.scene.projection_pasta
        es = context.scene.projection_pasta_erosion
        layout = self.layout

        layout.label(text="All optional. Maps load into source/.", icon="INFO")

        # Folder auto-detect for a consistent export set (e.g. Gleba).
        row = layout.row()
        row.scale_y = 1.2
        row.operator("pp.detect_source_maps", text="Detect Maps in source/", icon="VIEWZOOM")

        col = layout.column(align=True)

        # World map (display) -- uses the existing loader (builds the preview sphere).
        col.operator("pp.load_world_map", text="World Map (preview sphere)", icon="WORLD")

        # Heightmap slot (copy-aware picker + clear).
        hm = f"Heightmap: {s.heightmap_filename}" if s.heightmap_filename else "Heightmap (optional)"
        r = col.row(align=True)
        op = r.operator("pp.set_input_map", text=hm, icon="IMAGE_DATA")
        op.slot = "heightmap"; op.clear = False
        if s.heightmap_filename:
            cl = r.operator("pp.set_input_map", text="", icon="X")
            cl.slot = "heightmap"; cl.clear = True

        # Rainfall slot (decoded from a colormap map on load).
        rf = f"Rainfall: {es.rainfall_filename}" if es.rainfall_filename else "Rainfall (optional)"
        r = col.row(align=True)
        op = r.operator("pp.set_input_map", text=rf, icon="IMAGE_DATA")
        op.slot = "rainfall"; op.clear = False
        if es.rainfall_filename:
            cl = r.operator("pp.set_input_map", text="", icon="X")
            cl.slot = "rainfall"; cl.clear = True

        # Bathymetry slot (inverted Gleba depth map -> Sea Floor pass).
        bm = f"Bathymetry: {es.seafloor_bathy_filename}" if es.seafloor_bathy_filename else "Bathymetry (optional)"
        r = col.row(align=True)
        op = r.operator("pp.set_input_map", text=bm, icon="IMAGE_DATA")
        op.slot = "bathymetry"; op.clear = False
        if es.seafloor_bathy_filename:
            cl = r.operator("pp.set_input_map", text="", icon="X")
            cl.slot = "bathymetry"; cl.clear = True

        # Spatial erosion drivers (uplift / erodibility) -> modulate the LEM.
        layout.separator()
        drv = layout.column(align=True)
        drv.label(text="Spatial Drivers (erosion)", icon="FORCE_FORCE")

        um = f"Uplift: {es.uplift_filename}" if es.uplift_filename else "Uplift Map (optional)"
        r = drv.row(align=True)
        op = r.operator("pp.set_input_map", text=um, icon="IMAGE_DATA")
        op.slot = "uplift"; op.clear = False
        if es.uplift_filename:
            cl = r.operator("pp.set_input_map", text="", icon="X")
            cl.slot = "uplift"; cl.clear = True
        if es.uplift_filename:
            drv.prop(es, "uplift_influence", text="Influence")

        em = f"Erodibility: {es.erodibility_filename}" if es.erodibility_filename else "Erodibility Map (optional)"
        r = drv.row(align=True)
        op = r.operator("pp.set_input_map", text=em, icon="IMAGE_DATA")
        op.slot = "erodibility"; op.clear = False
        if es.erodibility_filename:
            cl = r.operator("pp.set_input_map", text="", icon="X")
            cl.slot = "erodibility"; cl.clear = True
        if es.erodibility_filename:
            drv.prop(es, "erodibility_contrast", text="Contrast")

        # Categorical -> per-class B&W masks (Gaea downstream).
        layout.separator()
        box = layout.box()
        box.label(text="Category Masks (for Gaea)", icon="MOD_MASK")
        box.label(text="Biome / Koppen -> one B&W mask per class")
        box.operator("pp.export_class_masks", text="Global Masks...", icon="EXPORT").scope = "GLOBAL"
        box.operator("pp.export_class_masks", text="Section Masks...", icon="EXPORT").scope = "SECTION"
        box.label(text="Section uses the Erosion panel's target", icon="INFO")


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------
class PP_PT_sphere(_PRPanel, Panel):
    bl_label = "Sphere"
    bl_idname = "PP_PT_sphere"
    bl_parent_id = "PP_PT_main"
    bl_order = 1

    def draw(self, context: bpy.types.Context) -> None:
        s = context.scene.projection_pasta
        layout = self.layout

        layout.operator("pp.load_world_map", text="Load World Map", icon="WORLD")
        hm_label = f"Heightmap: {s.heightmap_filename}" if s.heightmap_filename else "Select Heightmap (optional)"
        layout.operator("pp.select_heightmap", text=hm_label, icon="IMAGE_DATA")

        row = layout.row(align=True)
        row.operator("pp.expand_selection", text="Expand", icon="ADD")
        row.operator("pp.shrink_selection", text="Reduce", icon="REMOVE")

        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(s, "overlay_opacity")


# ---------------------------------------------------------------------------
# Section Export (+ Advanced sub-panel)
# ---------------------------------------------------------------------------
class PP_PT_section(_PRPanel, Panel):
    bl_label = "Section Export"
    bl_idname = "PP_PT_section"
    bl_parent_id = "PP_PT_main"
    bl_order = 2

    def draw(self, context: bpy.types.Context) -> None:
        s = context.scene.projection_pasta
        es = context.scene.projection_pasta_erosion
        layout = self.layout

        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(s, "new_section_name", text="Name")
        col.prop(s, "square_crop")
        col.prop(s, "feather_px", text="Feather")
        col.prop(s, "output_resolution")
        col.prop(es, "erode_on_create")

        row = layout.row()
        row.scale_y = 1.4
        row.operator("pp.create_section", text="Create Section from Selected Faces", icon="UV_FACESEL")


class PP_PT_section_advanced(_PRPanel, Panel):
    bl_label = "Advanced"
    bl_idname = "PP_PT_section_advanced"
    bl_parent_id = "PP_PT_section"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context: bpy.types.Context) -> None:
        # Disclosure header carries the override toggle; opening the panel to peek
        # no longer flips the override (the bug the old single-bool expander had).
        self.layout.prop(context.scene.projection_pasta, "override_projection_center", text="")

    def draw(self, context: bpy.types.Context) -> None:
        s = context.scene.projection_pasta
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        col = layout.column()
        col.active = s.override_projection_center
        col.prop(s, "override_center_lon", text="Center Longitude")
        col.prop(s, "override_center_lat", text="Center Latitude")


# ---------------------------------------------------------------------------
# Erosion (+ Advanced LEM and Channel Overlay sub-panels)
# ---------------------------------------------------------------------------
class PP_PT_erosion(_PRPanel, Panel):
    bl_label = "Erosion"
    bl_idname = "PP_PT_erosion"
    bl_parent_id = "PP_PT_main"
    bl_order = 3

    def draw_header(self, context: bpy.types.Context) -> None:
        self.layout.label(text="", icon="MOD_FLUIDSIM")

    def draw(self, context: bpy.types.Context) -> None:
        es = context.scene.projection_pasta_erosion
        s = context.scene.projection_pasta
        layout = self.layout

        if not deps.landlab_available():
            layout.label(text="Install dependencies to enable erosion", icon="INFO")
            return

        if deps.priorityflood_available():
            layout.label(text="Router: PriorityFlood (GPL, fast)", icon="CHECKMARK")
        else:
            layout.label(text="Router: MIT fallback (slower)", icon="INFO")

        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(es, "section")
        col.separator()
        col.prop(es, "lem_scale", text="Scale")
        if es.lem_scale != "CUSTOM":
            col.prop(es, "lem_intensity", text="Intensity")
        col.prop(es, "erosion_strength")
        col.separator()
        col.prop(es, "noise_kind", text="Seed Noise")
        col.prop(es, "noise_amp", text="Noise Amount")

        # Rainfall map (optional): file picker + clear. Drives where incision concentrates.
        rf_label = f"Rainfall: {es.rainfall_filename}" if es.rainfall_filename else "Rainfall Map (optional)"
        row = layout.row(align=True)
        row.operator("pp.select_rainfall", text=rf_label, icon="IMAGE_DATA")
        if es.rainfall_filename:
            row.operator("pp.select_rainfall", text="", icon="X").clear = True

        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        if es.lem_scale == "CUSTOM":
            col.prop(es, "steps")
            col.prop(es, "k_sp", text="Erodibility (K)")
        col.prop(s, "output_resolution", text="Detail / Output Res")
        col.prop(es, "target_peak_m", text="Target Peak")
        # The "Erode Section" button lives in PP_PT_erosion_run (a header-less child
        # panel ordered last), so it sits BELOW the advanced sub-panels below.


class PP_PT_erosion_lem(_PRPanel, Panel):
    bl_label = "Advanced LEM"
    bl_idname = "PP_PT_erosion_lem"
    bl_parent_id = "PP_PT_erosion"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return deps.landlab_available()

    def draw(self, context: bpy.types.Context) -> None:
        es = context.scene.projection_pasta_erosion
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column()
        col.prop(es, "sea_level_m", text="Sea Level")
        col.prop(es, "noise_seed")
        # Stream-power physics only apply when Scale = Custom; presets set them.
        if es.lem_scale == "CUSTOM":
            col.separator()
            col.prop(es, "m_sp")
            col.prop(es, "n_sp")
            col.prop(es, "diffusivity")
            col.prop(es, "uplift")
            col.prop(es, "dt")


class PP_PT_erosion_overlay(_PRPanel, Panel):
    bl_label = "Channel Overlay"
    bl_idname = "PP_PT_erosion_overlay"
    bl_parent_id = "PP_PT_erosion"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return deps.landlab_available()

    def draw_header(self, context: bpy.types.Context) -> None:
        self.layout.prop(context.scene.projection_pasta_erosion, "enable_overlay", text="")

    def draw(self, context: bpy.types.Context) -> None:
        es = context.scene.projection_pasta_erosion
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column()
        col.active = es.enable_overlay
        col.prop(es, "overlay_depth_m", text="Depth")
        col.prop(es, "overlay_w_macro_km", text="Widest Valley")
        col.prop(es, "overlay_r", text="Blur Ratio")


class PP_PT_erosion_glacial(_PRPanel, Panel):
    bl_label = "Glacial Erosion"
    bl_idname = "PP_PT_erosion_glacial"
    bl_parent_id = "PP_PT_erosion"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return deps.landlab_available()

    def draw_header(self, context: bpy.types.Context) -> None:
        self.layout.prop(context.scene.projection_pasta_erosion, "enable_glacial", text="")

    def draw(self, context: bpy.types.Context) -> None:
        es = context.scene.projection_pasta_erosion
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.label(text="Carves fjords before coast & rivers", icon="FREEZE")
        col = layout.column()
        col.active = es.enable_glacial
        col.prop(es, "glacial_ela_frac", text="Snowline")
        col.prop(es, "glacial_steps", text="Steps")
        col.separator()
        col.prop(es, "glacial_k_g", text="Carving Strength")
        col.prop(es, "glacial_quarry_mult", text="Quarrying")
        col.prop(es, "glacial_diffuse", text="U-Trough Smoothing")


class PP_PT_erosion_coastal(_PRPanel, Panel):
    bl_label = "Coastal Erosion"
    bl_idname = "PP_PT_erosion_coastal"
    bl_parent_id = "PP_PT_erosion"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return deps.landlab_available()

    def draw_header(self, context: bpy.types.Context) -> None:
        self.layout.prop(context.scene.projection_pasta_erosion, "enable_coastal", text="")

    def draw(self, context: bpy.types.Context) -> None:
        es = context.scene.projection_pasta_erosion
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.label(text="Reworks the coast before rivers carve", icon="MOD_OCEAN")
        col = layout.column()
        col.active = es.enable_coastal
        # Rate/Steps/Reach/Fetch follow the Scale x Intensity preset unless Scale = Custom.
        if es.lem_scale == "CUSTOM":
            col.prop(es, "coastal_rate_m", text="Erosion Rate")
            col.prop(es, "coastal_steps", text="Steps")
            col.prop(es, "coastal_notch_m", text="Wave Reach")
            col.prop(es, "coastal_max_fetch_km", text="Max Fetch")
        else:
            col.label(text="Rate / Steps / Reach / Fetch auto-sized by Scale", icon="INFO")
        col.separator()
        col.prop(es, "coastal_deposition", text="Build Beaches")
        col.prop(es, "coastal_talus_deg", text="Cliff Collapse")
        col.separator()
        col.prop(es, "coastal_swell_focus", text="Swell Focus")
        sub = col.column()
        sub.active = es.enable_coastal and es.coastal_swell_focus > 0.0
        sub.prop(es, "coastal_swell_deg", text="Swell Direction")


class PP_PT_erosion_seafloor(_PRPanel, Panel):
    bl_label = "Sea Floor / Gaea Export"
    bl_idname = "PP_PT_erosion_seafloor"
    bl_parent_id = "PP_PT_erosion"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return deps.landlab_available()

    def draw_header(self, context: bpy.types.Context) -> None:
        self.layout.prop(context.scene.projection_pasta_erosion, "enable_seafloor", text="")

    def draw(self, context: bpy.types.Context) -> None:
        es = context.scene.projection_pasta_erosion
        s = context.scene.projection_pasta
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.label(text="Fills the ocean + writes a Gaea-ready export", icon="MOD_OCEAN")
        col = layout.column()
        col.active = es.enable_seafloor
        col.prop(s, "ocean_floor_depth_m", text="Ocean Floor Depth")
        col.separator()
        col.prop(es, "seafloor_shelf_depth_m", text="Shelf Break Depth")
        col.prop(es, "seafloor_shelf_width_km", text="Shelf Width")
        col.prop(es, "seafloor_shelf_relief_mod", text="Shelf vs Relief")
        col.prop(es, "seafloor_slope_width_km", text="Slope Width")
        col.separator()
        col.prop(es, "seafloor_bathy_filename", text="Bathymetry Map")
        sub = col.column()
        sub.active = es.enable_seafloor and bool((es.seafloor_bathy_filename or "").strip())
        sub.prop(es, "seafloor_input_weight", text="Map Weight")

        # Gaea hand-off numbers from the last run -> the exact values to type into Gaea.
        if es.last_gaea_height_m > 0.0:
            box = layout.box()
            box.label(text="Last Gaea export -- set in Gaea:", icon="EXPORT")
            box.label(text=f"Sea level: {es.last_gaea_sea:.4f}")
            box.label(text=f"Height: {es.last_gaea_height_m:.0f} m")
            if es.last_gaea_width_scale < 0.999:
                box.label(text=f"Terrain width x {es.last_gaea_width_scale:.3f} (keep H:W ratio)", icon="ERROR")
            else:
                box.label(text="Terrain width: unchanged (1:1)")


class PP_PT_erosion_run(_PRPanel, Panel):
    # The Erode button + last-run readout. A header-less child panel with the highest
    # bl_order so it renders AFTER the Advanced LEM / Channel Overlay / Coastal
    # sub-panels -- i.e. the button sits at the very bottom of the Erosion section.
    bl_label = "Run Erosion"
    bl_idname = "PP_PT_erosion_run"
    bl_parent_id = "PP_PT_erosion"
    bl_order = 100
    bl_options = {"HIDE_HEADER"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return deps.landlab_available()

    def draw(self, context: bpy.types.Context) -> None:
        es = context.scene.projection_pasta_erosion
        layout = self.layout

        row = layout.row()
        row.scale_y = 1.4
        row.operator("pp.erode_section", text="Erode Section", icon="MOD_FLUIDSIM")

        if es.last_report:
            box = layout.box()
            ok = (0.4 <= es.last_theta <= 0.55) and (es.last_r2 > 0.9)
            box.label(
                text=f"theta={es.last_theta:.2f}  R2={es.last_r2:.2f}  band={es.last_band_slope:.3f}",
                icon="CHECKMARK" if ok else "INFO",
            )
            box.label(text=f"{es.last_router}   {es.last_secs:.1f}s")


# ---------------------------------------------------------------------------
# Reassembly
# ---------------------------------------------------------------------------
class PP_PT_reassembly(_PRPanel, Panel):
    bl_label = "Reassembly"
    bl_idname = "PP_PT_reassembly"
    bl_parent_id = "PP_PT_main"
    bl_order = 4
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context: bpy.types.Context) -> None:
        s = context.scene.projection_pasta
        layout = self.layout

        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(s, "reassembly_resolution")
        col.prop(s, "extend_edge_colors")
        col.prop(s, "normalize_heightmaps", text="Normalize Heights")

        layout.operator("pp.validate_processed", text="Validate", icon="CHECKMARK")
        row = layout.row()
        row.scale_y = 1.4
        row.operator("pp.reassemble", text="Reassemble", icon="MOD_BUILD")


_CLASSES = (
    PP_PT_main,
    PP_PT_inputs,
    PP_PT_sphere,
    PP_PT_section,
    PP_PT_section_advanced,
    PP_PT_erosion,
    PP_PT_erosion_lem,
    PP_PT_erosion_overlay,
    PP_PT_erosion_glacial,
    PP_PT_erosion_coastal,
    PP_PT_erosion_seafloor,
    PP_PT_erosion_run,
    PP_PT_reassembly,
)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
