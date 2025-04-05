"""
This script imports the layout (JSON) of an atom chip and visualizes it in Blender.
"""

# Note: bpy and mathutils are Blender's built-in modules (no need to install them).

import json
import bpy
import mathutils
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper


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
            material = component["material"]
            current = component["current"]

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
            obj.component_id = component_id
            obj.segment_id = segment_id
            obj.location = center
            obj.scale = (length, width, height)  # Scale the cube in the x, y, z directions
            obj.rotation_mode = "QUATERNION"
            # Rotate the object at the origin to align with the direction (start to end)
            obj.rotation_quaternion = mathutils.Vector((1, 0, 0)).rotation_difference(direction)
            obj.material = material
            obj.current = current

        return {"FINISHED"}


# === Menu Integration ===


def menu_func_import(self, context):
    self.layout.operator(AtomChipImporter.bl_idname, text="Atom Chip Layout (.json)")


# === Registration Management ===


def register():
    bpy.utils.register_class(AtomChipImporter)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(AtomChipImporter)
