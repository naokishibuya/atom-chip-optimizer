import argparse
import yaml
from typing import Any
import matplotlib
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
from ..atom_chip import AtomChip
from .layout_3d import plot_layout_3d
from .potential_1d import plot_potential_1d
from .potential_2d import plot_potential_2d
from .potential_3d import plot_potential_3d


matplotlib.use("Qt5Agg")


__all__ = [
    "show",
    "plot_layout_3d",
    "plot_potential_1d",
    "plot_potential_2d",
    "plot_potential_3d",
]


# Function to show the layout of the atom chip using a YAML configuration file
def show(atom_chip: AtomChip, yaml_path: str):
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="Do not show the visualization")
    args = parser.parse_args()
    if args.no_show:
        return

    # Load the YAML file
    with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    data = _convert_scientific_notation(data)

    # Load plot function and call it
    figs = []
    for plot_name in data["plots"]:
        plot_config = data[plot_name]
        function = eval(plot_config["function"])
        params = plot_config.get("params", {})
        fig = function(atom_chip, **params)
        figs.append(fig)

    # Get the screen size
    screen_width, screen_height = QApplication.desktop().screenGeometry().getRect()[2:]

    # Initialize the top-left position
    top = data.get("top", 0)
    left = data.get("left", 0)
    top_left = (top, left)

    # Set the geometry of each figure
    max_height = 0
    for fig in figs:
        window = fig.canvas.manager.window
        width, height = window.geometry().getRect()[2:]
        if left + width > screen_width:
            left = top_left[1]
            top += max_height + 50
            max_height = 0
            if top + height > screen_height:
                top = top_left[0]
        window.setGeometry(left, top, width, height)
        fig.show()
        left += width + 10
        max_height = max(max_height, height)

    plt.show()


def _convert_scientific_notation(data) -> Any:
    """
    Convert scientific notation strings to floats.

    Args:
        data: Data to be converted.

    Returns:
        Data with scientific notation strings converted to floats.
    """

    if isinstance(data, dict):
        return {k: _convert_scientific_notation(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_scientific_notation(i) for i in data]
    elif isinstance(data, str):
        try:
            return float(data)
        except ValueError:
            return data
    else:
        return data
