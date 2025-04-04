#!/bin/sh

# obtain the path to the JSON design file
DESIGN_FILE=$1
if [ -z "$DESIGN_FILE" ]; then
    echo "Usage: $0 <design_file.json>"
    exit 1
fi
if [ ! -f "$DESIGN_FILE" ]; then
    echo "File not found: $DESIGN_FILE"
    exit 1
fi

CURRENT_DIR=$(dirname "$0")
DESIGN_DIR=$(realpath "$CURRENT_DIR/../designs")

blender --python "$DESIGN_DIR/blender.py" -- "$DESIGN_FILE"
