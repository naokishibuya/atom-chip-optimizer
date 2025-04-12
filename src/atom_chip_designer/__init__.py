bl_info = {
    "name": "Atom Chip Designer",
    "author": "Naoki Shibuya",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "location": "File > Import/Export, Sidebar > AtomChip",
    "description": "Import/export and create atom chip wire layouts",
    "category": "Import-Export",
}

# ruff: noqa: E402

from . import (
    properties,
    duplicate_handler,
    import_json,
    export_json,
    control_panel,
)


modules = (
    properties,
    duplicate_handler,
    import_json,
    export_json,
    control_panel,
)


def register():
    for module in modules:
        module.register()


def unregister():
    for module in reversed(modules):
        module.unregister()
