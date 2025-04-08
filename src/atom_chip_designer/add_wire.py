"""
This script adds a wire component in Blender.
"""

# Note: bpy and mathutils are Blender's built-in modules (no need to install them).

import bpy
from bpy.types import Operator, Panel
from bpy.props import FloatProperty, EnumProperty
from mathutils import Vector
from .properties import MATERIAL_ENUM_ITEMS, DEFAULT_MATERIAL, add_rectangular_conductor


# fmt: off
DEFAULT_CURRENT = 1.0
DEFAULT_LENGTH = 0.01  # 10 mm
DEFAULT_WIDTH  = 0.001  # 1 mm
DEFAULT_HEIGHT = 0.001  # 1 mm
STEP = 0.001  # 1 mm
# fmt: on


class AtomChipWireAdder(Operator):
    """Add a new atom chip wire component"""

    bl_idname = "object.add_atom_chip_wire"
    bl_label = "Add Atom Chip Wire"
    bl_options = {"REGISTER", "UNDO"}

    # === Properties to appear in the pop-up dialog ===
    # fmt: off
    material: EnumProperty (name="Material", items=MATERIAL_ENUM_ITEMS) # type: ignore[reportInvalidTypeForm]
    current : FloatProperty(name="Current [A]")                         # type: ignore[reportInvalidTypeForm]
    center_x: FloatProperty(name="Center X", unit='LENGTH', step=STEP)  # type: ignore[reportInvalidTypeForm]
    center_y: FloatProperty(name="Y"       , unit='LENGTH', step=STEP)  # type: ignore[reportInvalidTypeForm]
    center_z: FloatProperty(name="Z"       , unit='LENGTH', step=STEP)  # type: ignore[reportInvalidTypeForm]
    length  : FloatProperty(name="Length"  , unit='LENGTH', step=STEP)  # type: ignore[reportInvalidTypeForm]
    width   : FloatProperty(name="Width"   , unit='LENGTH', step=STEP)  # type: ignore[reportInvalidTypeForm]
    height  : FloatProperty(name="Height"  , unit='LENGTH', step=STEP)  # type: ignore[reportInvalidTypeForm]
    # fmt: on

    def execute(self, context):
        selected = context.active_object
        if selected and selected.get("component_id") is not None:
            new_component_id = selected["component_id"]
            ids = [obj.get("segment_id", -1) for obj in bpy.data.objects if obj.get("component_id") == new_component_id]
            new_segment_id = max(ids + [-1]) + 1
        else:
            # Generate a new component ID
            existing_ids = [obj.get("component_id", -1) for obj in bpy.data.objects]
            new_component_id = max(existing_ids + [-1]) + 1
            new_segment_id = 0

        # add a unit cube and scale it
        add_rectangular_conductor(
            new_component_id,
            new_segment_id,
            self.material,
            self.current,
            (self.center_x, self.center_y, self.center_z),
            (self.length, self.width, self.height),
        )

        return {"FINISHED"}

    def invoke(self, context, event):
        selected = context.active_object
        if selected and selected.get("component_id") is not None:
            # Get the local Z direction in world space
            z_axis_world = selected.matrix_world.to_3x3() @ Vector((0, 0, 1))
            local_offset = z_axis_world.normalized() * selected.scale[2]
            new_location = selected.location + local_offset

            # Get the existing wire location
            self.center_x = new_location[0]
            self.center_y = new_location[1]
            self.center_z = new_location[2]

            # Get the existing wire scale
            self.length = selected.scale[0]
            self.width = selected.scale[1]
            self.height = selected.scale[2]

            # Get the existing wire material and current
            self.material = getattr(selected, "material")  # as string
            self.current = getattr(selected, "current")  # as float
        else:
            # Default wire location
            self.center_x = 0.0
            self.center_y = 0.0
            self.center_z = 0.0

            # Default wire parameters
            self.length = DEFAULT_LENGTH
            self.width = DEFAULT_WIDTH
            self.height = DEFAULT_HEIGHT

            # Default wire material and current
            self.material = DEFAULT_MATERIAL
            self.current = DEFAULT_CURRENT

        return context.window_manager.invoke_props_dialog(self)


class AtomChipToolsPanel(Panel):
    bl_label = "Atom Chip Tools"
    bl_idname = "OBJECT_PT_atom_chip_tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AtomChip"

    def draw(self, context):
        layout = self.layout
        layout.operator("object.add_atom_chip_wire", icon="MESH_CUBE")
        layout.prop(context.scene, "show_atom_chip_markers")  # toggle visibility of markers


# === Registration Management ===

classes = [AtomChipWireAdder, AtomChipToolsPanel]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
