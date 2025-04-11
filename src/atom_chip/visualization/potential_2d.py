from typing import Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from ..atom_chip import AtomChip
from ..potential import constants


def plot_potential_2d(
    atom_chip: AtomChip,
    size: Tuple[int, int],
    x_range: Tuple[float, float, int],
    y_range: Tuple[float, float, int],
    z: Optional[float] = None,
    zlim: Optional[Tuple[float, float]] = None,
    isosurface_level: Optional[float] = None,
    quiver: Optional[bool] = False,
    cmap: Optional[str] = "jet",
    fig: Optional[plt.Figure] = None,
):
    if not atom_chip.potential.minimum.found:
        print("Minimum not found. Cannot plot potential.")
        return

    x_vals = np.linspace(*x_range)
    y_vals = np.linspace(*y_range)
    X, Y = np.meshgrid(x_vals, y_vals)

    E_min = atom_chip.potential.minimum
    if z is None:
        z = E_min.position[2]
    points = np.array([[x, y, z] for x, y in zip(X.flatten(), Y.flatten())])
    E, _, B = atom_chip.get_potentials(points)

    E = E.reshape(X.shape)
    T = constants.joule_to_microKelvin(E)

    # create figure
    if fig is None:
        fig = plt.figure(figsize=size)
    else:
        fig.clear()
    ax = fig.add_subplot(111)
    img = ax.imshow(
        T,
        extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
        origin="lower",
        cmap=cmap,
        aspect="auto",
        vmin=zlim[0] if zlim else None,
        vmax=zlim[1] if zlim else None,
    )
    fig.gca().set_facecolor([0.2, 0.2, 0.2])  # Dark background

    # Overlay contour line at the isosurface level
    relative_level = constants.joule_to_microKelvin(E_min.value) + isosurface_level
    ax.contour(X, Y, T, levels=[relative_level], colors="w", linestyles="dashed", linewidths=1)

    # Add Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    fig.colorbar(img, cax=cax, label="Energy [μK]")

    # Add Quiver Arrows (Magnetic Field Vectors)
    if quiver:
        # Reshape field components
        BX = B[:, 0].reshape(X.shape)
        BY = B[:, 1].reshape(X.shape)

        qp = 15  # Reduce vector density
        ax.quiver(
            X[::qp, ::qp],
            Y[::qp, ::qp],
            BX[::qp, ::qp],
            BY[::qp, ::qp],
            color="black",
            scale=1000,
            width=0.001,
            alpha=0.8,
            headwidth=10,
            headlength=10,
        )

    # Labels & Formatting
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Energy[μK] @ z = {z:.4g} mm (Isosurface @{isosurface_level} μK + Minimum)", y=1.05)
    ax.set_xlim([X.min(), X.max()])
    ax.set_ylim([Y.min(), Y.max()])
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig
