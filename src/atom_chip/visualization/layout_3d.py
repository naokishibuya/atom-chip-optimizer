from typing import List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ..atom_chip import AtomChip
from ..components import RectangularConductor


def plot_layout_3d(
    atom_chip: AtomChip,
    title: str = "Atom Chip Layout",
    size: Tuple[int, int] = (7, 6),
    azim: float = 0.0,
    elev: float = 0.0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    zlim: Optional[Tuple[float, float]] = None,
    tick: Optional[Union[float, List[float]]] = None,
):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")

    # plot the atom chip components
    for component in atom_chip.components:
        if isinstance(component, RectangularConductor):
            _plot_rectangular_conductor(ax, component)
        else:
            raise ValueError(f"Unsupported component type: {type(component)}")

    # set title, label, and limits
    ax.set_title(title + f" ({atom_chip.name})")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    # set limits if not provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

    # Add dashed reference lines along axes
    ax.plot(ax.get_xlim(), [0, 0], [0, 0], color="gray", linestyle="--", linewidth=1, alpha=0.3)
    ax.plot([0, 0], ax.get_ylim(), [0, 0], color="gray", linestyle="--", linewidth=1, alpha=0.3)
    ax.plot([0, 0], [0, 0], ax.get_zlim(), color="gray", linestyle="--", linewidth=1, alpha=0.3)

    # Set ticks
    if tick is not None:
        if isinstance(tick, (int, float)):
            tick = [tick] * 3
        for axis, tick in zip([ax.xaxis, ax.yaxis, ax.zaxis], tick):
            axis.set_major_locator(plt.MultipleLocator(tick))

    # set view angle and aspect ratio
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((np.ptp(ax.get_xlim3d()), np.ptp(ax.get_ylim3d()), np.ptp(ax.get_zlim3d())))

    return fig


def _plot_rectangular_conductor(ax: plt.Axes, conductor: RectangularConductor):
    """
    Plot a rectangular conductor in 3D.

    Args:
        ax (plt.Axes): Matplotlib axis object.
        conductor (RectangularConductor): Rectangular conductor object.
    """
    # Get material properties
    current = conductor.current
    for corners in conductor.get_vertices():
        # Define the 6 faces of the cuboid
        faces = [
            [corners[j] for j in [0, 1, 2, 3]],
            [corners[j] for j in [4, 5, 6, 7]],
            [corners[j] for j in [0, 1, 5, 4]],
            [corners[j] for j in [2, 3, 7, 6]],
            [corners[j] for j in [0, 3, 7, 4]],
            [corners[j] for j in [1, 2, 6, 5]],
        ]

        # Get the color of the conductor
        color, alpha = _get_material_color(conductor.material, current)

        # Add the cuboid to the plot
        ax.add_collection3d(
            Poly3DCollection(
                faces,
                facecolors=color,
                edgecolor="k",
                alpha=alpha,
                linewidth=0.1,
                axlim_clip=True,
            )
        )


def _get_material_color(material: str, current: float) -> List[int]:
    """
    Get the color of a conductor based on its material and current direction.

    Args:
        material (str): Material of the conductor.
        current (float): Current in the conductor.

    Returns:
        list: RGBA color.
    """
    # Default colors for conductors (RGBA format)
    colors = {
        "copper": [
            [[255, 112, 66], 0.6],  # For positive current
            [[66, 112, 255], 0.6],  # For negative current
            [[198, 117, 26], 0.1],  # For zero current
        ],
        "gold": [
            [[255, 215, 0], 0.6],  # For positive current
            [[0, 215, 255], 0.6],  # For negative current
            [[198, 117, 26], 0.1],  # For zero current
        ],
    }[material]

    # Get the color based on the current direction
    index = 0 if current > 0 else 1 if current < 0 else 2
    color, alpha = colors[index]
    color = np.array(color) / 255  # Convert RGB to [0, 1] range
    return color, alpha
