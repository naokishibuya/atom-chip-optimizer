"""
This script defines properties and functions for managing them.
"""

# Note: bpy and mathutils are Blender's built-in modules (no need to install them).

import bpy
import mathutils


# Shared enum items for material
MATERIAL_ENUM_ITEMS = [
    ("copper", "Copper", ""),
    ("gold", "Gold", ""),
]
DEFAULT_MATERIAL = "copper"


# Material colors (RGBA format)
# fmt: off
MATERIAL_COLORS = {
    "copper": [
        [1.0, 0.40, 0.2, 1.0],  # For positive current
        [0.2, 0.40, 1.0, 1.0],  # For negative current
        [1.0, 0.40, 0.2, 0.1],  # For zero current
    ],
    "gold": [
        [1.0, 0.84, 0.0, 1.0],  # For positive current
        [0.2, 0.84, 1.0, 1.0],  # For negative current
        [1.0, 0.84, 0.0, 0.1],  # For zero current
    ],
}
# fmt: on


class AtomChipPropertiesPanel(bpy.types.Panel):
    bl_label = "Atom Chip Properties"
    bl_idname = "OBJECT_PT_atom_chip_props"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    def draw(self, context):
        obj = context.object
        if not obj:
            return
        layout = self.layout
        if "component_id" in obj:  # only true if property is local
            layout.label(text=f"Component ID: {obj.component_id}")
            layout.label(text=f"Segment ID: {obj.segment_id}")
            layout.prop(obj, "material")
            layout.prop(obj, "current")
        elif obj.parent and "component_id" in obj.parent:  # only true if property is inherited
            layout.label(text=f"Component ID: {obj.parent.component_id}")
            layout.label(text=f"Segment ID: {obj.parent.segment_id}")
            layout.label(text="Remarks: This indicates the current flow direction.")


def add_rectangular_conductor(
    component_id: int,
    segment_id: int,
    material: str,
    current: float,
    location: mathutils.Vector,
    scale: mathutils.Vector,  # (length, width, height)
    direction: mathutils.Vector = mathutils.Vector((1, 0, 0)),
):
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    obj.name = f"Wire.{component_id:03d}_{segment_id:03d}"
    obj.location = location
    obj.scale = scale  # Scale the cube in the x, y, z directions
    obj.rotation_mode = "QUATERNION"
    # Rotate the object at the origin to align with the direction (start to end)
    obj.rotation_quaternion = mathutils.Vector((1, 0, 0)).rotation_difference(direction)

    # Add direction arrow (Do this before material and current are set to the object)
    add_current_markers(obj)

    obj.component_id = component_id
    obj.segment_id = segment_id
    obj.material = material
    obj.current = current
    return obj


def add_current_markers(obj):
    # Create marker1 (cube)
    bpy.ops.mesh.primitive_cube_add(size=1)
    marker1 = bpy.context.active_object
    marker1.name = f"{obj.name}_current_marker_1"
    marker1.location = (-0.475, 0, 0)  # Move it to the beginning of the conductor
    marker1.scale = (0.05, 1.0, 1.01)
    marker1.parent = obj
    marker1.matrix_parent_inverse.identity()  # Don't inherit scale from parent
    marker1.hide_select = True  # Hide from selection

    # Create marker2 (cube)
    bpy.ops.mesh.primitive_cube_add(size=1)
    marker2 = bpy.context.active_object
    marker2.name = f"{obj.name}_current_marker_2"
    marker2.location = (0.475, 0, 0)  # Move it to the end of the conductor
    marker2.scale = (0.05, 1.0, 1.01)
    marker2.parent = obj
    marker2.matrix_parent_inverse.identity()  # Don't inherit scale from parent
    marker2.hide_select = True  # Hide from selection


def update_material_color(obj):
    if obj.type != "MESH":
        return

    current = getattr(obj, "current", 0.0)
    material = getattr(obj, "material", "copper")

    # Get the material based on the current direction
    mat = get_material(material, current)
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # Update the color of the current markers
    if current > 0:
        # Positive current
        marker1_mat = get_current_source_material(material, current, "+")
        marker2_mat = get_current_source_material(material, current, "-")
    elif current < 0:
        # Negative current
        marker1_mat = get_current_source_material(material, current, "-")
        marker2_mat = get_current_source_material(material, current, "+")
    else:
        # no current
        marker1_mat = mat
        marker2_mat = mat

    # Get the current scene
    scene = bpy.context.scene
    show = scene.show_atom_chip_markers

    # Update the color of the current markers
    marker1_key = f"{obj.name}_current_marker_1"
    if marker1_key in bpy.data.objects:
        marker1 = bpy.data.objects[marker1_key]
        marker1.data.materials.clear()
        marker1.data.materials.append(marker1_mat)
        marker1.hide_set(not show)

    marker2_key = f"{obj.name}_current_marker_2"
    if marker2_key in bpy.data.objects:
        marker2 = bpy.data.objects[marker2_key]
        marker2.data.materials.clear()
        marker2.data.materials.append(marker2_mat)
        marker2.hide_set(not show)


def get_current_source_material(material, current, sign):
    # Get the color based on the current direction
    index = 0 if current > 0 else 1 if current < 0 else 2
    key = f"CURRENT_SOURCE:{material}:{index}:{sign}"
    if key in bpy.data.materials:
        return bpy.data.materials[key]  # Already cached

    r, g, b, a = MATERIAL_COLORS[material][index]
    if sign == "+":
        # make it darker for current source
        diff = 0.2
        color = [max(0.0, r - diff), max(0.0, g - diff), max(0.0, b - diff), a]
    elif sign == "-":
        # make it lighter for current sink
        diff = 0.2
        color = [min(1.0, r + diff), min(1.0, g + diff), min(1.0, b + diff), a]

    # Create a new material with the specified color
    mat = bpy.data.materials.new(name=key)
    mat.diffuse_color = color
    return mat


def get_or_create_material(name, color):
    mat = bpy.data.materials.get(name)
    if not mat:
        mat = bpy.data.materials.new(name=name)
        mat.diffuse_color = color
    return mat


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


def update_current(self, context):
    component_id = self.get("component_id")
    if component_id is None:
        return

    update_material_color(self)
    for obj in bpy.data.objects:
        if obj == self:
            continue
        if obj.get("component_id") == component_id:
            obj["current"] = self["current"]
            update_material_color(obj)


def update_material(self, context):
    component_id = self.get("component_id")
    if component_id is None:
        return

    update_material_color(self)
    for obj in bpy.data.objects:
        if obj == self:
            continue
        if obj.get("component_id") == component_id:
            obj["material"] = self["material"]
            update_material_color(obj)


def update_current_marker_visibility(scene, context):
    show = scene.show_atom_chip_markers
    for obj in bpy.data.objects:
        if "_current_marker_" in obj.name:
            obj.hide_set(not show)


def register():
    bpy.types.Object.component_id = bpy.props.IntProperty(name="Component ID")
    bpy.types.Object.segment_id = bpy.props.IntProperty(name="Segment ID")
    bpy.types.Object.material = bpy.props.EnumProperty(
        name="Material",
        items=MATERIAL_ENUM_ITEMS,
        default=DEFAULT_MATERIAL,
        update=update_material,
    )
    bpy.types.Object.current = bpy.props.FloatProperty(
        name="Current",
        default=0.0,
        update=update_current,
    )
    bpy.types.Scene.show_atom_chip_markers = bpy.props.BoolProperty(
        name="Show Current Markers",
        default=False,
        description="Toggle visibility of current direction markers",
        update=update_current_marker_visibility,
    )
    bpy.utils.register_class(AtomChipPropertiesPanel)


def unregister():
    del bpy.types.Object.component_id
    del bpy.types.Object.segment_id
    del bpy.types.Object.material
    del bpy.types.Object.current
    del bpy.types.Scene.show_atom_chip_markers
    bpy.utils.unregister_class(AtomChipPropertiesPanel)
