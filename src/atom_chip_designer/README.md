# Atom Chip Designer (Integration for Blender)

This Blender add-on allows you to import, visualize, edit, and export atom chip wire layouts using JSON files compatible with the atom chip simulation framework.

## Installation

1. Zip the plugin folder

    Make sure you zip the entire folder (not just its contents):

    ```bash
    zip -r atom_chip_designer.zip atom_chip_designer/
    ```

2. Install the add-on in Blender

- Go to `Edit` → `Preferences`
- Select the `Add-ons` tab
- Click the `Install from Disk...` (top-right dropdown)
- Select the `atom_chip_designer.zip` you created above
- Enable the add-on by checking the box (if not already enabled)

## Uninstallation

1. Go to `Edit` → `Preferences` → `Add-ons`
2. Find **Atom Chip Designer**
3. Uncheck the box to disable
4. Click the `Uninstall` button to remove it completely

## Usage

### Importing Atom Chip Layout

- Go to File → Import → Atom Chip Layout (.json)
- Select your layout file
- The geometry will be imported into the 3D scene
  (materials/colors reflect the current direction)

### Exporting Atom Chip Layout

- Go to File → Export → Atom Chip Layout (.json)
- Choose a location and file name, then click Export
- The current layout will be saved as JSON

### Adding Wires

If the add-on includes wire creation tools:

- Hit `N` to open the N-panel (right side of the 3D Viewport) if not already visible
- Select the `AtomChip` tab to show `Atom Chip Tools` panel
- Click Add Atom Chip Wire to insert a new wire

### Tips

- All distances are in millimeters (mm). It may look tiny in Blender, but it is correct.
- Use the Atom Chip Properties panel in the Object Properties tab to view and edit wire metadata (current, material, etc.).
- Modifying current or material will update the wire's color in real-time.
- Segments belonging to the same component will be automatically updated when current or material is changed.

