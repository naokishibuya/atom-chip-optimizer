bl_info = {
    "name": "Atom Chip Exporter",
    "author": "Naoki Shibuya",
    "version": (1, 0),
    "blender": (4, 0, 0),  # depending on your Blender version
    "location": "File > Export > Atom Chip Export (JSON)",
    "description": "Export atom chip layout to JSON format",
    "category": "Import-Export",
}

# ruff: noqa: E402

# Atom Chip Layout Exporter for Blender
# This script exports the layout of an atom chip as a JSON file.
#
# [Registration]
# You can use the Blender's Scripting tab to enable this script:
# 1. Open this Python script in the Scripting.
# 2. Click Run Script.
# 3. File > Export > Atom Chip Layout (.json)
#
# Optionally, you can make it permanent:
# 1. Edit -> Preferences
# 2. Add-ons -> Install (dropdown menu @ top-right corner)
# 3. Select this script
# 4. You can enable/disable/delete it from the Add-ons tab
#
import bpy
import mathutils
import json
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper


class AtomChipExporter(Operator, ExportHelper):
    """Export Atom Chip Layout as JSON"""

    bl_idname = "export_scene.atom_chip_json"
    bl_label = "Export Atom Chip Layout (.json)"
    bl_options = {"PRESET"}

    filename_ext = ".json"

    def execute(self, context):
        data = []
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue

            # Compute start and end along local X axis
            start = obj.matrix_world @ mathutils.Vector((-0.5, 0, 0))
            end = obj.matrix_world @ mathutils.Vector((0.5, 0, 0))

            width, height = list(obj.scale[1:3])

            item = {
                "component_id": obj.get("component_id", -1),
                "segment_id": obj.get("segment_id", -1),
                "material": obj.get("material", "unknown"),
                "current": round_nz(obj.get("current", 0.0)),
                "start": [round_nz(v, 3) for v in start],
                "end": [round_nz(v, 3) for v in end],
                "width": round_nz(width, 3),
                "height": round_nz(height, 3),
            }
            data.append(item)

        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

        self.report({"INFO"}, f"Exported {len(data)} components to {self.filepath}")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


# helper to avoid -0.0 (nz for handling negative zero)
def round_nz(value, precision=3):
    rounded = round(value, precision)
    return 0.0 if rounded == -0.0 else rounded


# === Menu Integration ===
# This function adds the export option to the File > Export menu in Blender.
def menu_func_export(self, context):
    self.layout.operator(AtomChipExporter.bl_idname, text="Atom Chip (.json)")


def register():
    bpy.utils.register_class(AtomChipExporter)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.utils.unregister_class(AtomChipExporter)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)


if __name__ == "__main__":
    register()
