from __future__ import annotations

from pathlib import Path

import bpy
import bmesh
from bpy.types import Operator

from .. import manifest as manifest_lib


def _get_edit_bmesh(context: bpy.types.Context) -> bmesh.types.BMesh | None:
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        return None
    if context.mode != "EDIT_MESH":
        return None
    return bmesh.from_edit_mesh(obj.data)


def _get_sphere_obj(context: bpy.types.Context) -> bpy.types.Object | None:
    s = context.scene.projection_pasta
    obj = bpy.data.objects.get(s.sphere_object_name)
    if obj is None or obj.type != "MESH":
        return None
    return obj


def _sphere_material_name(obj: bpy.types.Object) -> str:
    return f"PP_SphereMat_{obj.name}"


def _set_single_material_slot(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    """Ensure the mesh has exactly 1 material slot pointing at `mat`."""
    mats = obj.data.materials
    mats.clear()
    mats.append(mat)


def _ensure_sphere_material(
    *,
    obj: bpy.types.Object,
    world_image: bpy.types.Image,
    overlay_path: Path | None,
) -> None:
    """
    Build (or rebuild) the sphere preview material.

    Structure:
        World Map Texture  -->  Base Color
        Overlay Texture    -->  Emission Color
                                Emission Strength = 10 * overlay_opacity
        Principled BSDF    -->  Material Output (Surface)
    """
    mat_name = _sphere_material_name(obj)
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nt = mat.node_tree

    # Clear all nodes
    nt.nodes.clear()

    # Material Output
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)

    # Principled BSDF
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.name = "PP_BSDF"
    bsdf.location = (100, 0)

    # World Map Texture
    tex_world = nt.nodes.new("ShaderNodeTexImage")
    tex_world.name = "PP_WorldTex"
    tex_world.label = "World Map"
    tex_world.image = world_image
    tex_world.location = (-300, 0)

    # Connect World Map -> Base Color
    nt.links.new(tex_world.outputs["Color"], bsdf.inputs["Base Color"])

    # Connect BSDF -> Material Output
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Overlay (if exists)
    opacity = bpy.context.scene.projection_pasta.overlay_opacity
    if overlay_path is not None and overlay_path.exists():
        overlay_img = bpy.data.images.load(str(overlay_path), check_existing=True)
        overlay_img.reload()  # Ensure we have the latest pixels
        overlay_img.colorspace_settings.name = "sRGB"

        tex_overlay = nt.nodes.new("ShaderNodeTexImage")
        tex_overlay.name = "PP_OverlayTex"
        tex_overlay.label = "Extracted Overlay"
        tex_overlay.image = overlay_img
        tex_overlay.location = (-300, -300)

        # Connect Overlay -> Emission Color
        nt.links.new(tex_overlay.outputs["Color"], bsdf.inputs["Emission Color"])

        # Set Emission Strength = 10 * opacity
        bsdf.inputs["Emission Strength"].default_value = 10.0 * opacity
    else:
        # No overlay: emission off
        bsdf.inputs["Emission Strength"].default_value = 0.0

    _set_single_material_slot(obj, mat)


def update_overlay_opacity(opacity: float) -> None:
    """Update Emission Strength on the sphere's Principled BSDF."""
    obj = _get_sphere_obj(bpy.context)
    if obj is None or not obj.data.materials:
        return
    mat = obj.data.materials[0]
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return
    bsdf = mat.node_tree.nodes.get("PP_BSDF")
    if bsdf is None:
        return
    bsdf.inputs["Emission Strength"].default_value = 10.0 * opacity


def ensure_overlay_connected(context: bpy.types.Context) -> None:
    """
    Ensure the sphere material includes the extracted overlay connected to Emission.
    Safe to call repeatedly.
    """
    s = context.scene.projection_pasta
    root = s.project_root_path()
    mp = s.manifest_path()
    if root is None or mp is None or not mp.exists():
        return

    obj = _get_sphere_obj(context)
    if obj is None:
        return

    manifest = manifest_lib.read_manifest(mp)
    world_map = manifest.get("global", {}).get("world_map", {}) or {}
    world_path = world_map.get("path")
    if not world_path:
        return

    world_img = bpy.data.images.load(world_path, check_existing=True)

    overlay_path = None
    overlay = manifest.get("global", {}).get("overlay")
    if overlay and overlay.get("path"):
        overlay_path = (root / overlay["path"]).resolve()

    _ensure_sphere_material(obj=obj, world_image=world_img, overlay_path=overlay_path)


def ensure_overlay_connected_with_paths(
    context: bpy.types.Context,
    *,
    overlay_path: Path,
) -> None:
    """
    Like ensure_overlay_connected(), but uses a known overlay path (e.g. freshly written)
    so we don't depend on manifest.json being updated yet.
    """
    s = context.scene.projection_pasta
    obj = _get_sphere_obj(context)
    if obj is None:
        return

    # Try to get world image from existing material
    world_img = None
    if obj.data.materials:
        mat = obj.data.materials[0]
        if mat and mat.use_nodes and mat.node_tree:
            node = mat.node_tree.nodes.get("PP_WorldTex")
            if node and getattr(node, "image", None) is not None:
                world_img = node.image

    # Fallback: load from manifest
    if world_img is None:
        root = s.project_root_path()
        mp = s.manifest_path()
        if root is None or mp is None or not mp.exists():
            return
        manifest = manifest_lib.read_manifest(mp)
        world_map = manifest.get("global", {}).get("world_map", {}) or {}
        world_path = world_map.get("path")
        if not world_path:
            return
        world_img = bpy.data.images.load(world_path, check_existing=True)

    _ensure_sphere_material(obj=obj, world_image=world_img, overlay_path=overlay_path)


class PP_OT_create_sphere(Operator):
    bl_idname = "pp.create_sphere"
    bl_label = "Create Projection Sphere"
    bl_description = "Create a UV sphere suitable for equirectangular preview and face selection"

    def execute(self, context: bpy.types.Context):
        s = context.scene.projection_pasta

        if context.mode != "OBJECT":
            if context.active_object is not None and bpy.ops.object.mode_set.poll():
                bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=64,
            ring_count=32,
            radius=1.0,
        )
        obj = context.active_object
        if obj is None:
            self.report({"ERROR"}, "Failed to create sphere")
            return {"CANCELLED"}
        obj.name = s.sphere_object_name

        if obj.data.uv_layers.active is None:
            obj.data.uv_layers.new(name="UVMap")

        self.report({"INFO"}, f"Created sphere: {obj.name}")
        return {"FINISHED"}


class PP_OT_assign_preview_texture(Operator):
    bl_idname = "pp.assign_preview_texture"
    bl_label = "Assign Preview Texture"
    bl_description = "Assign an Image Texture from source/ to the sphere for visual selection"

    filepath: bpy.props.StringProperty(  # type: ignore[valid-type]
        name="Image",
        subtype="FILE_PATH",
        default="",
    )

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: bpy.types.Context):
        s = context.scene.projection_pasta
        obj = bpy.data.objects.get(s.sphere_object_name)
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, f"Sphere object not found: {s.sphere_object_name}")
            return {"CANCELLED"}
        if not self.filepath:
            self.report({"ERROR"}, "No image selected")
            return {"CANCELLED"}

        img = bpy.data.images.load(self.filepath, check_existing=True)
        _ensure_sphere_material(obj=obj, world_image=img, overlay_path=None)

        self.report({"INFO"}, "Assigned preview texture to sphere")
        return {"FINISHED"}


class PP_OT_load_world_map(Operator):
    bl_idname = "pp.load_world_map"
    bl_label = "Load World Map"
    bl_description = "Load an equirectangular (2:1) world map, infer global size, create the sphere, and assign it as preview material"

    filepath: bpy.props.StringProperty(  # type: ignore[valid-type]
        name="World Map",
        subtype="FILE_PATH",
        default="",
    )

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: bpy.types.Context):
        s = context.scene.projection_pasta
        root = s.project_root_path()
        mp = s.manifest_path()
        if root is None or mp is None or not mp.exists():
            self.report({"ERROR"}, "manifest.json not found (set Project Root and run Init Project)")
            return {"CANCELLED"}
        if not self.filepath:
            self.report({"ERROR"}, "No world map selected")
            return {"CANCELLED"}

        img = bpy.data.images.load(self.filepath, check_existing=True)
        w, h = img.size
        if h == 0:
            self.report({"ERROR"}, "Invalid image size")
            return {"CANCELLED"}

        ratio = w / h
        tol = 0.05
        if abs(ratio - 2.0) > tol:
            self.report({"WARNING"}, f"World map aspect ratio is {ratio:.3f} (expected ~2.0)")

        if bpy.data.objects.get(s.sphere_object_name) is None:
            bpy.ops.pp.create_sphere()

        obj = bpy.data.objects.get(s.sphere_object_name)
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Sphere creation failed")
            return {"CANCELLED"}

        manifest = manifest_lib.read_manifest(mp)
        manifest.setdefault("global", {}).setdefault("projection", "Equirectangular")
        manifest["global"]["size"] = [int(w), int(h)]
        manifest.setdefault("global", {})["world_map"] = {
            "path": self.filepath,
            "size": [int(w), int(h)],
            "aspect_ratio": ratio,
            "aspect_ratio_ok": abs(ratio - 2.0) <= tol,
            "tolerance": tol,
        }
        layers = manifest["global"].setdefault("layers", [])
        already = any(l.get("id") == "color_map" for l in layers)
        if not already:
            ext = Path(self.filepath).suffix.lower()
            fmt = "PNG" if ext == ".png" else ("JPEG" if ext in (".jpg", ".jpeg") else "PNG")
            layers.append(
                {
                    "id": "color_map",
                    "path": self.filepath,
                    "datatype": "continuous",
                    "format": fmt,
                    "interp": "linear",
                }
            )
        manifest_lib.write_manifest(mp, manifest)

        overlay_path = None
        overlay = manifest.get("global", {}).get("overlay")
        if overlay and overlay.get("path"):
            overlay_path = (root / overlay["path"]).resolve()

        _ensure_sphere_material(obj=obj, world_image=img, overlay_path=overlay_path)

        self.report({"INFO"}, f"Loaded world map ({w}x{h}) and assigned to sphere")
        return {"FINISHED"}


class PP_OT_expand_selection(Operator):
    bl_idname = "pp.expand_selection"
    bl_label = "Expand Face Selection"
    bl_description = "Expand the current face selection by N rings (Edit Mode only)"

    def execute(self, context: bpy.types.Context):
        s = context.scene.projection_pasta
        bm = _get_edit_bmesh(context)
        if bm is None:
            self.report({"ERROR"}, "Must be in Edit Mode with a mesh active")
            return {"CANCELLED"}

        for _ in range(int(s.expand_selection_rings)):
            boundary = [f for f in bm.faces if f.select]
            for f in boundary:
                for e in f.edges:
                    for f2 in e.link_faces:
                        f2.select = True

        bmesh.update_edit_mesh(context.active_object.data)
        return {"FINISHED"}


class PP_OT_set_overlay_opacity(Operator):
    bl_idname = "pp.set_overlay_opacity"
    bl_label = "Set Overlay Opacity"
    bl_description = "Update the extracted overlay opacity on the sphere material"

    def execute(self, context: bpy.types.Context):
        update_overlay_opacity(context.scene.projection_pasta.overlay_opacity)
        return {"FINISHED"}


_CLASSES = (
    PP_OT_create_sphere,
    PP_OT_assign_preview_texture,
    PP_OT_load_world_map,
    PP_OT_expand_selection,
    PP_OT_set_overlay_opacity,
)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
