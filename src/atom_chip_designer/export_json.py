"""
This script exports the layout of an atom chip as a JSON file.
"""

# Note: bpy and mathutils are Blender's built-in modules (no need to install them).

import json
import bpy
import mathutils
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper


def export_atom_chip_layout() -> list[dict]:
    layout = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.parent:
            continue  # Skip child objects (like markers)

        # Compute start and end along local X axis
        start = obj.matrix_world @ mathutils.Vector((-0.5, 0, 0)) * 1e3  # m to mm
        end = obj.matrix_world @ mathutils.Vector((0.5, 0, 0)) * 1e3  # m to mm
        width = obj.scale[1] * 1e3  # m to mm
        height = obj.scale[2] * 1e3  # m to mm

        item = {
            "component_id": obj.component_id,
            "segment_id": obj.segment_id,
            "material": obj.material,
            "current": round_nz(obj.current),
            "start": [round_nz(v, 3) for v in start],
            "end": [round_nz(v, 3) for v in end],
            "width": round_nz(width, 3),
            "height": round_nz(height, 3),
        }
        layout.append(item)
    return layout


class AtomChipExporter(Operator, ExportHelper):
    """Export Atom Chip Layout as JSON"""

    bl_idname = "export_scene.atom_chip_json"
    bl_label = "Export Atom Chip Layout (.json)"
    bl_options = {"PRESET"}

    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(  # type: ignore[reportInvalidTypeForm]
        default="*.json",
        options={"HIDDEN"},
    )
    filename: bpy.props.StringProperty(  # type: ignore[reportInvalidTypeForm]
        default="atom_chip_layout.json",
    )

    def execute(self, context):
        layout = export_atom_chip_layout()
        with open(self.filepath, "w") as f:
            json.dump(layout, f, indent=2)

        self.report({"INFO"}, f"Exported {len(layout)} components to {self.filepath}")
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


# === Registration Management ===


def register():
    bpy.utils.register_class(AtomChipExporter)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.utils.unregister_class(AtomChipExporter)
