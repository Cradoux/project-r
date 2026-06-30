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

        mp = s.manifest_path()
        has_project = bool(mp and mp.exists())
        row = layout.row(align=True)
        if has_project:
            row.operator("pp.open_project", text="Reload Project", icon="FILE_REFRESH")
        else:
            row.operator("pp.init_project", text="Create Project", icon="NEWFOLDER")
        row.operator("pp.open_manifest", text="manifest.json", icon="TEXT")


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
        col.prop(es, "noise_kind", text="Seed Noise")
        col.prop(es, "noise_amp", text="Noise Amount")
        col.separator()
        col.prop(es, "climate_kind", text="Climate")
        col.prop(es, "climate_strength", text="Strength")
        col.separator()
        col.prop(es, "steps")
        col.prop(es, "k_sp", text="Erodibility (K)")
        col.prop(es, "max_work_px", text="Max Work Res")
        col.prop(es, "target_peak_m", text="Target Peak")

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
        col.prop(es, "m_sp")
        col.prop(es, "n_sp")
        col.prop(es, "diffusivity")
        col.prop(es, "uplift")
        col.prop(es, "dt")
        col.prop(es, "noise_seed")


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
        col.prop(s, "extend_edge_colors")
        col.prop(s, "normalize_heightmaps", text="Normalize Heights")

        layout.operator("pp.validate_processed", text="Validate", icon="CHECKMARK")
        row = layout.row()
        row.scale_y = 1.4
        row.operator("pp.reassemble", text="Reassemble", icon="MOD_BUILD")


_CLASSES = (
    PP_PT_main,
    PP_PT_sphere,
    PP_PT_section,
    PP_PT_section_advanced,
    PP_PT_erosion,
    PP_PT_erosion_lem,
    PP_PT_erosion_overlay,
    PP_PT_reassembly,
)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
