from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
):
    if not atom_chip.trap.minimum.found:
        print("Minimum not found. Cannot plot potential.")
        return

    x_vals = np.linspace(*x_range)
    y_vals = np.linspace(*y_range)
    X, Y = np.meshgrid(x_vals, y_vals)

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")

    if z_range is not None:
        start, stop, num_intervals = z_range
        num_points = num_intervals + 1
        z_vals = np.linspace(start, stop, num_points)

        def update(frame):
            ax.clear()
            # Compute potentials at each point
            z = z_vals[frame]
            surf = _plot_3d_trapping_potential(atom_chip, ax, X, Y, z, zlim)
            return (surf,)

        # keep the animation object alive
        fig.anim = FuncAnimation(fig, update, frames=len(z_vals), interval=1000, blit=False)
        surf = update(0)[0]
    else:
        # Plot 3D trapping potential at a fixed z value
        surf = _plot_3d_trapping_potential(atom_chip, ax, X, Y, z, zlim)

    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label="Energy [μK]", pad=0.1)
    fig.tight_layout()
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
    E_min = atom_chip.trap.minimum
    z = z if z is not None else E_min.point[2]
    point = np.array([E_min.point[0], E_min.point[1], z])
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
