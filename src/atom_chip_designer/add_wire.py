"""
This script adds a wire component in Blender.
"""

# Note: bpy and mathutils are Blender's built-in modules (no need to install them).

import bpy
import mathutils
from bpy.types import Operator, Panel
from .properties import update_material_color, MATERIAL_ENUM_ITEMS, DEFAULT_MATERIAL


# fmt: off
DEFAULT_LENGTH = 0.01  # 10 mm
DEFAULT_WIDTH  = 0.0001  # 0.1 mm
DEFAULT_HEIGHT = 0.0001  # 0.1 mm
# fmt: on


class AtomChipWireAdder(Operator):
    """Add a new atom chip wire component"""

    bl_idname = "object.add_atom_chip_wire"
    bl_label = "Add Atom Chip Wire"
    bl_options = {"REGISTER", "UNDO"}

    # === Properties to appear in the pop-up dialog ===
    # fmt: off
    material: bpy.props.EnumProperty (name="Material", default=DEFAULT_MATERIAL, items=MATERIAL_ENUM_ITEMS)   # type: ignore[reportInvalidTypeForm]
    current : bpy.props.FloatProperty(name="Current",  default=1.0)                                           # type: ignore[reportInvalidTypeForm]
    length  : bpy.props.FloatProperty(name="Length" ,  default=DEFAULT_LENGTH  , min=0.001  , unit='LENGTH')  # type: ignore[reportInvalidTypeForm]
    width   : bpy.props.FloatProperty(name="Width"  ,  default=DEFAULT_WIDTH   , min=0.00001, unit='LENGTH')  # type: ignore[reportInvalidTypeForm]
    height  : bpy.props.FloatProperty(name="Height" ,  default=DEFAULT_HEIGHT  , min=0.00001, unit='LENGTH')  # type: ignore[reportInvalidTypeForm]
    # fmt: on

    def execute(self, context):
        selected = context.active_object
        if selected and selected.get("component_id") is not None:
            new_component_id = selected["component_id"]
            ids = [obj.get("segment_id", -1) for obj in bpy.data.objects if obj.get("component_id") == new_component_id]
            new_segment_id = max(ids + [-1]) + 1

            # Offset along local Z direction
            # Get the local Z direction in world space
            z_axis_world = selected.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1))
            local_offset = z_axis_world.normalized() * 0.001  # 1 mm offset
            new_location = selected.location + local_offset
        else:
            # Generate a new component ID
            existing_ids = [obj.get("component_id", -1) for obj in bpy.data.objects]
            new_component_id = max(existing_ids + [-1]) + 1
            new_segment_id = 0
            new_location = (0, 0, 0)

        # add a unit cube and scale it
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = context.active_object
        obj.location = new_location
        obj.scale = (self.length, self.width, self.height)  # scale the cube in the x, y, z directions
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = mathutils.Quaternion((1, 0, 0, 0))  # no rotation

        # Set custom properties
        obj.component_id = new_component_id
        obj.segment_id = new_segment_id
        obj.material = self.material
        obj.current = self.current

        # Apply material
        update_material_color(obj)
        return {"FINISHED"}

    def invoke(self, context, event):
        selected = context.active_object
        if selected and selected.get("component_id") is not None:
            # Get the existing wire parameters
            self.length = selected.scale[0]
            self.width = selected.scale[1]
            self.height = selected.scale[2]

            # Get the existing wire material and current
            self.material = getattr(selected, "material")  # as string
            self.current = getattr(selected, "current")  # as float
        else:
            # Default wire parameters
            self.length = DEFAULT_LENGTH
            self.width = DEFAULT_WIDTH
            self.height = DEFAULT_HEIGHT

            # Default wire material and current
            self.material = DEFAULT_MATERIAL
            self.current = 1.0

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


# === Registration Management ===

classes = [AtomChipWireAdder, AtomChipToolsPanel]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
