from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from ..atom_chip import AtomChip
from ..potential import constants


def plot_potential_3d(
    atom_chip: AtomChip,
    size: Tuple[int, int],
    x_range: Tuple[float, float, int],
    y_range: Tuple[float, float, int],
    z_range: Optional[Tuple[float, float]] = None,
    z: Optional[float] = None,
    zlim: Optional[Tuple[float, float]] = None,
    fig: Optional[plt.Figure] = None,
) -> plt.Figure:
    if fig is None:
        fig = plt.figure(figsize=size)
    else:
        fig.clear()
    ax = fig.add_subplot(111, projection="3d")

    if not atom_chip.potential.minimum.found:
        fig.text(0.5, 0.5, "Potential Minimum not found.", ha="center", va="center", fontsize=12)
        return fig

    x_vals = np.linspace(*x_range)
    y_vals = np.linspace(*y_range)
    X, Y = np.meshgrid(x_vals, y_vals)

    if z_range:
        z_vals = np.linspace(z_range[0], z_range[1], z_range[2] + 1)
        initial_z = z_vals[len(z_vals) // 2]
    elif z is not None:
        initial_z = z
        z_vals = [z]
    else:
        initial_z = atom_chip.potential.minimum.position[2]
        z_vals = [initial_z]

    # Initial plot
    surf = _plot_3d_trapping_potential(atom_chip, ax, X, Y, initial_z, zlim)

    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label="Energy [μK]", pad=0.1)
    fig.tight_layout()

    # Slider
    if z_range:
        ax_slider = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # [left, bottom, width, height]
        slider = Slider(
            ax_slider,
            "z [mm]",
            z_vals[0],
            z_vals[-1],
            valinit=initial_z,
        )

        def update(val):
            ax.clear()
            _plot_3d_trapping_potential(atom_chip, ax, X, Y, val, zlim)
            fig.canvas.draw_idle()

        slider.on_changed(update)
        fig._slider = slider  # Prevent garbage collection
    return fig


def _plot_3d_trapping_potential(
    atom_chip: AtomChip,
    ax: plt.Axes,
    X: np.ndarray,  # Meshgrid x-coordinates
    Y: np.ndarray,  # Meshgrid y-coordinates
    z: float,  # z-coordinate of the minimum potential energy
    zlim: Tuple[float, float],  # z-axis limits for the plot
) -> Poly3DCollection:
    # Get the energy at a given z-coordinate or the minimum energy point
    E_min = atom_chip.potential.minimum
    z = z if z is not None else E_min.position[2]
    point = np.array([E_min.position[0], E_min.position[1], z])
    V_at_z = atom_chip.get_potentials(point)[0][0]
    points = np.array([[x, y, z] for x, y in zip(X.flatten(), Y.flatten())])
    E, _, B = atom_chip.get_potentials(points)
    V = E.reshape(X.shape)

    T = constants.joule_to_microKelvin(V)
    T_at_z = constants.joule_to_microKelvin(V_at_z)

    surf = ax.plot_surface(X, Y, T, cmap="jet", edgecolor="none", vmin=zlim[0], vmax=zlim[1])
    levels = np.linspace(zlim[0], zlim[1], 20)
    ax.contour(X, Y, T, levels=levels, cmap="jet", offset=zlim[0])

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("Energy [μK]")
    ax.set_title(f"3D Trapping Potential @ z = {z:.4g} mm ({T_at_z:.1f} μK)")
    ax.set_zlim(zlim)
    return surf
