"""
This script imports the layout (JSON) of an atom chip and visualizes it in Blender.
"""

# Note: bpy and mathutils are Blender's built-in modules (no need to install them).

import json
import bpy
import mathutils
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper
from .properties import add_rectangular_conductor


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
        clear_atom_chip_objects()

        wires = data["wires"]
        for component in wires:
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

            add_rectangular_conductor(
                component_id,
                segment_id,
                material,
                current,
                center,
                (length, width, height),
                direction,
            )

        if "bias_fields" in data:
            scene = bpy.context.scene
            bias_fields = data["bias_fields"]
            scene.bias_coil_factors_x = bias_fields["coil_factors"][0]
            scene.bias_coil_factors_y = bias_fields["coil_factors"][1]
            scene.bias_coil_factors_z = bias_fields["coil_factors"][2]
            scene.bias_currents_x = bias_fields["currents"][0]
            scene.bias_currents_y = bias_fields["currents"][1]
            scene.bias_currents_z = bias_fields["currents"][2]
            scene.bias_stray_fields_x = bias_fields["stray_fields"][0]
            scene.bias_stray_fields_y = bias_fields["stray_fields"][1]
            scene.bias_stray_fields_z = bias_fields["stray_fields"][2]

        # Deselect all objects
        bpy.ops.object.select_all(action="DESELECT")

        return {"FINISHED"}


def clear_atom_chip_objects():
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)


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
