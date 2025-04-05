"""
This script defines properties and functions for managing them.
"""

# Note: bpy and mathutils are Blender's built-in modules (no need to install them).

import bpy


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


def update_material_color(obj):
    current = getattr(obj, "current", 0.0)
    material = getattr(obj, "material", "copper")

    if obj.type == "MESH":
        # Define material key
        index = 0 if current > 0 else 1 if current < 0 else 2
        key = f"{material}:{index}"

        # Create or get existing material
        mat = bpy.data.materials.get(key)
        if not mat:
            mat = bpy.data.materials.new(name=key)
            mat.diffuse_color = MATERIAL_COLORS[material][index]

        # Apply to object
        obj.data.materials.clear()
        obj.data.materials.append(mat)


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
    bpy.utils.register_class(AtomChipPropertiesPanel)


def unregister():
    bpy.utils.unregister_class(AtomChipPropertiesPanel)
    del bpy.types.Object.component_id
    del bpy.types.Object.segment_id
    del bpy.types.Object.material
    del bpy.types.Object.current
