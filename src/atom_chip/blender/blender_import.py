bl_info = {
    "name": "Atom Chip Importer",
    "author": "Naoki Shibuya",
    "version": (1, 0),
    "blender": (4, 0, 0),  # depending on your Blender version
    "location": "File > Import > Atom Chip Export (JSON)",
    "description": "Import atom chip layout from JSON format",
    "category": "Import-Export",
}

# ruff: noqa: E402

# Atom Chip Layout Importer for Blender
# ==========================================================
# This script imports the layout of an atom chip via a JSON file.
#
# [Addon Registration]
# 1. Edit -> Preferences
# 2. Add-ons -> Install (dropdown menu @ top-right corner)
# 3. Select this script
#
# [Addon Uninstallation]
# You can enable/disable/delete it from the Add-ons tab
#
# [How to use]
# To import the atom chip layout from a JSON file into Blender:
# File > Import > Atom Chip Layout (.json)

# Note: bpy and mathutils are part of the Blender Python API
# This script is intended to be run inside Blender's scripting environment
# and is not meant to be run as a standalone Python script.
import json
import bpy
import mathutils
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper


# Material colors (RGBA format)
# fmt: off
MATERIAL_COLORS = {
    "copper": [
        [1.0 , 0.4 , 0.2, 1.0],  # For positive current
        [0.2 , 0.4 , 1.0, 1.0],  # For negative current
        [0.78, 0.46, 0.1, 0.5],  # For zero current
    ],
    "gold": [
        [1.0 , 0.84, 0.0, 1.0],  # For positive current
        [0.0 , 0.84, 1.0, 1.0],  # For negative current
        [0.78, 0.76, 0.0, 0.5],  # For zero current
    ],
}
# fmt: on


def get_material(name: str, current: float) -> mathutils.Color:
    # Get the color based on the current direction
    index = 0 if current > 0 else 1 if current < 0 else 2
    key = f"{name}:{index}"
    if key in bpy.data.materials:
        material = bpy.data.materials[key]  # Already cached
    else:
        # Create a new material with the specified color
        material = bpy.data.materials.new(name=key)
        material.diffuse_color = MATERIAL_COLORS[name][index]
    return material


class AtomChipImporter(Operator, ImportHelper):
    """Import Atom Chip Layout (.json)"""

    bl_idname = "import_scene.atom_chip_json"
    bl_label = "Import Atom Chip Layout (.json)"
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(  # type: ignore[reportInvalidTypeForm]
        default="*.json",
        options={"HIDDEN"},
    )

    def execute(self, context):
        with open(self.filepath, "r") as f:
            data = json.load(f)

        # Clear existing objects
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

        for component in data:
            component_id = component["component_id"]
            segment_id = component["segment_id"]
            start = mathutils.Vector(component["start"]) * 1e-3  # from mm to m
            end = mathutils.Vector(component["end"]) * 1e-3  # from mm to m
            width = component["width"] * 1e-3  # from mm to m
            height = component["height"] * 1e-3  # from mm to m
            current = component["current"]
            material = component["material"]

            center = (start + end) / 2
            direction = (end - start).normalized()
            length = (end - start).length

            # Sanity check: skip any broken geometry
            if not (width > 0 and height > 0 and length > 0):
                print(f"Skipping component {component_id} segment_id {segment_id} with non-positive size")
                print(f"[{component_id}, {segment_id}] Start: {start}, End: {end}, Center: {center}, Length: {length}")
                continue

            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
            obj = bpy.context.active_object
            obj.location = center
            obj.scale = (length, width, height)  # Scale the cube in the x, y, z directions
            obj.rotation_mode = "QUATERNION"
            # Rotate the object at the origin to align with the direction (start to end)
            obj.rotation_quaternion = mathutils.Vector((1, 0, 0)).rotation_difference(direction)
            obj.data.materials.append(get_material(material, current))

            # Set the other properties for the object
            obj.component_id = component_id
            obj.segment_id = segment_id
            obj.material = material
            obj.current = current

        return {"FINISHED"}


class AtomChipPropertiesPanel(bpy.types.Panel):
    bl_label = "Atom Chip Properties"
    bl_idname = "OBJECT_PT_atom_chip_props"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    def draw(self, context):
        layout = self.layout
        obj = context.object
        if obj:
            layout.label(text=f"Component ID: {obj.component_id}")
            layout.label(text=f"Segment ID: {obj.segment_id}")
            layout.prop(obj, "material")
            layout.prop(obj, "current")


def menu_func_import(self, context):
    self.layout.operator(AtomChipImporter.bl_idname, text="Atom Chip Layout (.json)")


def update_material_color(self, context):
    current = getattr(self, "current", 0.0)
    material = getattr(self, "material", "copper")
    mat = get_material(material, current)

    if self.type == "MESH":
        self.data.materials.clear()
        self.data.materials.append(mat)


def register():
    # Register custom property with update callback
    bpy.types.Object.component_id = bpy.props.IntProperty(name="Component ID")
    bpy.types.Object.segment_id = bpy.props.IntProperty(name="Segment ID")
    bpy.types.Object.material = bpy.props.EnumProperty(
        name="Material",
        items=[
            ("copper", "Copper", ""),
            ("gold", "Gold", ""),
        ],
        default="copper",
        update=update_material_color,
    )
    bpy.types.Object.current = bpy.props.FloatProperty(name="Current", default=0.0, update=update_material_color)
    bpy.utils.register_class(AtomChipImporter)
    bpy.utils.register_class(AtomChipPropertiesPanel)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    # Clean up property
    del bpy.types.Object.component_id
    del bpy.types.Object.segment_id
    del bpy.types.Object.material
    del bpy.types.Object.current

    bpy.utils.unregister_class(AtomChipPropertiesPanel)
    bpy.utils.unregister_class(AtomChipImporter)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
