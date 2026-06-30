from __future__ import annotations

import bpy
from bpy.types import Panel

from . import is_scipy_available, is_pillow_available, is_landlab_available, is_priorityflood_available


class PP_PT_main(Panel):
    bl_label = "Project-R"
    bl_idname = "PP_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Project-R"

    def draw(self, context: bpy.types.Context) -> None:
        s = context.scene.projection_pasta
        layout = self.layout

        # Dependencies check
        missing_deps = []
        if not is_pillow_available():
            missing_deps.append("Pillow")
        if not is_scipy_available():
            missing_deps.append("scipy")

        if missing_deps:
            box = layout.box()
            box.alert = True
            box.label(text=f"Missing: {', '.join(missing_deps)}", icon="ERROR")
            box.operator("pp.install_dependencies", text="Install Dependencies")
            box.label(text="(Restart Blender after install)")

        box = layout.box()
        box.label(text="Project")
        box.prop(s, "project_root")
        box.prop(s, "planet_radius_km")
        box.prop(s, "max_elevation_m")
        row = box.row(align=True)
        row.operator("pp.init_project", text="Init Project")
        row.operator("pp.open_manifest", text="Open manifest.json")

        box = layout.box()
        box.label(text="Sphere")
        box.operator("pp.load_world_map", text="Load World Map")
        # Heightmap selector - show filename if set
        hm_label = f"Heightmap: {s.heightmap_filename}" if s.heightmap_filename else "Select Heightmap (optional)"
        box.operator("pp.select_heightmap", text=hm_label)
        row = box.row(align=True)
        row.operator("pp.expand_selection", text="Expand")
        row.operator("pp.shrink_selection", text="Reduce")
        box.prop(s, "overlay_opacity")

        box = layout.box()
        box.label(text="Section Export")
        box.prop(s, "new_section_name")
        box.prop(s, "square_crop")
        box.prop(s, "feather_px")
        
        # Advanced options (collapsed by default)
        adv_box = box.box()
        adv_row = adv_box.row()
        adv_row.prop(
            s, "override_projection_center",
            icon="TRIA_DOWN" if s.override_projection_center else "TRIA_RIGHT",
            text="Advanced Options",
            emboss=False
        )
        if s.override_projection_center:
            col = adv_box.column(align=True)
            col.prop(s, "override_center_lon")
            col.prop(s, "override_center_lat")
        
        box.operator("pp.create_section", text="Create Section from Selected Faces")

        # --- Erosion ---
        es = context.scene.projection_pasta_erosion
        box = layout.box()
        box.label(text="Erosion", icon="MOD_FLUIDSIM")

        if not is_landlab_available():
            warn = box.box()
            warn.alert = True
            warn.label(text="landlab not installed", icon="ERROR")
            warn.operator("pp.install_dependencies", text="Install Dependencies")
        else:
            if is_priorityflood_available():
                box.label(text="Router: PriorityFlood (GPL, fast)", icon="CHECKMARK")
            else:
                box.label(text="Router: MIT fallback (slower)", icon="INFO")

            box.prop(es, "section_id", text="Section")

            col = box.column(align=True)
            col.prop(es, "noise_kind")
            col.prop(es, "noise_amp")

            col = box.column(align=True)
            col.prop(es, "climate_kind")
            col.prop(es, "climate_strength")

            col = box.column(align=True)
            col.prop(es, "steps")
            col.prop(es, "k_sp")

            # Advanced LEM (collapsed)
            adv = box.box()
            adv.prop(
                es, "show_lem_advanced",
                icon="TRIA_DOWN" if es.show_lem_advanced else "TRIA_RIGHT",
                text="Advanced LEM Settings",
                emboss=False,
            )
            if es.show_lem_advanced:
                c = adv.column(align=True)
                c.prop(es, "m_sp")
                c.prop(es, "n_sp")
                c.prop(es, "diffusivity")
                c.prop(es, "uplift")
                c.prop(es, "dt")
                c.prop(es, "noise_seed")
                c.prop(es, "max_work_px")

            # Wilbur overlay
            ov = box.box()
            ov.prop(es, "enable_overlay")
            if es.enable_overlay:
                c = ov.column(align=True)
                c.prop(es, "overlay_depth_m")
                c.prop(es, "overlay_w_macro_km")
                c.prop(es, "overlay_r")

            box.prop(es, "target_peak_m")
            box.operator("pr.erode_section", text="Erode Section", icon="MOD_FLUIDSIM")

            if es.last_report:
                rb = box.box()
                ok = (0.4 <= es.last_theta <= 0.55) and (es.last_r2 > 0.9)
                rb.label(
                    text=f"theta={es.last_theta:.2f} R2={es.last_r2:.2f} band={es.last_band_slope:.3f}",
                    icon="CHECKMARK" if ok else "INFO",
                )
                rb.label(text=f"{es.last_router}  {es.last_secs:.1f}s")

        box = layout.box()
        box.label(text="Reassembly")
        box.prop(s, "extend_edge_colors")
        box.prop(s, "normalize_heightmaps")
        row = box.row(align=True)
        row.operator("pp.validate_processed", text="Validate")
        row.operator("pp.reassemble", text="Reassemble")


def register() -> None:
    bpy.utils.register_class(PP_PT_main)


def unregister() -> None:
    bpy.utils.unregister_class(PP_PT_main)


