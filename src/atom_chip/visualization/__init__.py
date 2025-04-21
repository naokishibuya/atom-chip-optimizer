import yaml
import sys
import logging
from typing import Any
import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
from ..atom_chip import AtomChip
from .layout_3d import plot_layout_3d
from .potential_1d import plot_potential_1d
from .potential_2d import plot_potential_2d
from .potential_3d import plot_potential_3d
from .summary_text import show_summary


__all__ = [
    "show",
    "plot_layout_3d",
    "plot_potential_1d",
    "plot_potential_2d",
    "plot_potential_3d",
    "show_summary",
    "Visualizer",
]


def show(atom_chip: AtomChip, yaml_path: str):
    # Parse command-line arguments
    if "--no-show" in sys.argv:
        return

    # Visualize atom chip layout and analytics
    visualizer = Visualizer(yaml_path)
    visualizer.update(atom_chip)

    input("\nPress Enter to close the figures...\n\n")


class Visualizer:
    def __init__(self, config_path: str):
        self._plot_windows = {}
        self._config = _load_config(config_path)

        # Initialize the top-left position
        self._global_top = self._config.get("top", 0)
        self._global_left = self._config.get("left", 0)
        self._top = self._global_top
        self._left = self._global_left
        self._bottom = 0

    def update(self, atom_chip: AtomChip):
        plt.ion()
        for plot_name in self._config["plots"]:
            plot_config = self._config[plot_name]
            try:
                self._update_plot(plot_name, atom_chip, plot_config)
            except Exception as e:
                logging.error(f"Error updating plot '{plot_name}': {e}")
        plt.draw()
        plt.pause(0.5)

    def _update_plot(self, name: str, atom_chip: AtomChip, plot_config: dict):
        function = eval(plot_config["function"])
        params = plot_config.get("params", {})

        if name in self._plot_windows:
            # Update the existing figure
            fig = self._plot_windows[name]
            function(atom_chip, fig=fig, **params)
        else:
            # Create a new figure if it doesn't exist and register the close event
            fig = function(atom_chip, **params)
            self._initialize_position(fig)
            self._plot_windows[name] = fig
            fig.canvas.mpl_connect("close_event", self._close_handler)

    def _initialize_position(self, fig):
        # Get the screen size
        screen_width, screen_height = QApplication.desktop().screenGeometry().getRect()[2:]
        window = fig.canvas.manager.window
        width, height = window.geometry().getRect()[2:]
        top, left = self._top, self._left
        if left + width > screen_width:
            top = self._bottom + 40
            left = self._global_left
            if top + height > screen_height:
                top = self._global_top
        window.setGeometry(left, top, width, height)
        self._bottom = max(self._bottom, top + height)
        self._top = top
        self._left = left + width + 10

    def _close_handler(self, event):
        title = event.canvas.manager.get_window_title()
        logging.info(f"Closed: '{title}'")
        for key, fig in list(self._plot_windows.items()):
            if fig.canvas.manager.get_window_title() == title:
                del self._plot_windows[key]
                break
        if not self._plot_windows:
            logging.info("All figures closed.")
            sys.exit(0)


def _load_config(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config = _convert_scientific_notation(config)
    return config


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
