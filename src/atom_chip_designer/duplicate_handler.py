import bpy
from .properties import add_current_markers, update_material_color


# This is a workaround to handle the case where an object is duplicated
# and the component_id is not updated correctly.
class ComponentTracker:
    _last_object_ids = {}
    _enabled = True

    @classmethod
    def pause(cls):
        cls._enabled = False

    @classmethod
    def resume(cls):
        cls._enabled = True
        cls.snapshot()

    @classmethod
    def snapshot(cls):
        cls._last_object_ids = {
            obj.name: obj["component_id"] for obj in bpy.data.objects if "component_id" in obj and obj.parent is None
        }

    @classmethod
    def on_update(cls, scene, depsgraph):
        if not cls._enabled:
            return

        # Delete orphaned current markers
        for obj in bpy.data.objects:
            if obj.name.endswith("_current_marker_1") or obj.name.endswith("_current_marker_2"):
                if obj.parent is None or obj.parent.name not in bpy.data.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)

        # Update component_id and segment_id for new objects
        for update in depsgraph.updates:
            if not isinstance(update.id, bpy.types.Object):
                continue
            obj = bpy.data.objects.get(update.id.name)
            if cls.is_new_object(obj):
                next_id = cls.get_next_component_id()
                segment_id = 0
                old_name = obj.name
                obj["component_id"] = next_id
                obj["segment_id"] = segment_id
                obj.name = f"Wire.{next_id:03d}_{obj.get('segment_id', segment_id):03d}"

                # Promote to actual registered properties
                obj.component_id = next_id
                obj.segment_id = segment_id
                obj.material = getattr(obj, "material", "copper")
                obj.current = getattr(obj, "current", 0.0)

                cls._last_object_ids[obj.name] = obj["component_id"]  # track the new name
                cls._last_object_ids.pop(old_name, None)  # remove old name from tracking

                # Add current markers
                add_current_markers(obj)
                update_material_color(obj)

    @classmethod
    def is_new_object(cls, obj):
        return obj and "component_id" in obj and obj.parent is None and obj.name not in cls._last_object_ids

    @classmethod
    def get_next_component_id(cls):
        """Return the next available component_id."""
        ids = [obj.get("component_id") for obj in bpy.data.objects if "component_id" in obj]
        return max(ids) + 1 if ids else 1


def register():
    if ComponentTracker.on_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(ComponentTracker.on_update)


def unregister():
    if ComponentTracker.on_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(ComponentTracker.on_update)
