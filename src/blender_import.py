# bpy and mathutils are part of the Blender Python API
# This script is intended to be run inside Blender's scripting environment
# and is not meant to be run as a standalone Python script.
import sys
import json
import bpy
import mathutils


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


def import_atom_chip_layout(components: str):
    """
    Import the atom chip layout from a JSON file and visualize it in Blender.
    Args:
        components (str): JSON string containing the components data.
    """
    # Clear default cube
    bpy.ops.object.select_all(action="SELECT")  # Selects all objects in the current scene.
    bpy.ops.object.delete(use_global=False)  # Deletes the selected objects.

    for component in components:
        component_id = component["component_id"]
        segment_id = component["segment_id"]
        start = mathutils.Vector(component["start"])
        end = mathutils.Vector(component["end"])
        width = component["width"]
        height = component["height"]
        current = component["current"]
        material = component["material"]

        center = (start + end) / 2
        length = (end - start).length

        # Sanity check: skip any broken geometry
        if not (width > 0 and height > 0 and length > 0):
            print(f"Skipping component {component_id} segment_id {segment_id} with non-positive size")
            print(f"[{component_id}, {segment_id}] Start: {start}, End: {end}, Center: {center}, Length: {length}")
            continue

        # Rotate the cube to align with the start and end points
        x_axis = mathutils.Vector((1, 0, 0))  # x_axis vector
        direction = (end - start).normalized()
        rotation = x_axis.rotation_difference(direction)

        # Add a unit cube at origin, rotate and scale
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.location = center
        obj.scale = (length, width, height)  # Scale the cube in the x, y, and z directions
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = rotation
        obj.data.materials.append(get_material(material, current))

        # Set the material for the object
        obj["component_id"] = component_id
        obj["segment_id"] = segment_id
        obj["material"] = material
        obj["current"] = current


if __name__ == "__main__":
    # json file path
    args = sys.argv
    if "--" not in args:
        print("Please provide the path to the JSON file after '--'")
        sys.exit(1)
    path = args[args.index("--") + 1]

    # Load the JSON file
    try:
        with open(path, "r") as f:
            components = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        if not path.endswith(".json"):
            print("Please ensure the file is a valid JSON file.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    # Call the function to show the visualization
    import_atom_chip_layout(components)
