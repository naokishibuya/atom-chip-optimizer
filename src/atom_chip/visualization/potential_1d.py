from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from ..atom_chip import AtomChip
from ..potential import constants


def plot_potential_1d(
    atom_chip: AtomChip,
    size: Tuple[int, int],
    z_range: Tuple[float, float, int],
):
    if not atom_chip.trap.minimum.found:
        print("Minimum not found. Cannot plot potential.")
        return

    # Get the minimum energy point from the atom chip
    x, y = atom_chip.trap.minimum.point[:2]
    z_vals = np.linspace(*z_range)

    points = np.array([[x, y, z] for z in z_vals])
    E, B_mag, _ = atom_chip.get_potentials(points)

    fig, ax1 = plt.subplots(figsize=size)
    fig.gca().grid()

    # Magnetic field
    ax1.plot(z_vals, B_mag, "bo", markersize=1, label="|B| [G]")
    ax1.set_xlabel("z (mm)")
    ax1.set_ylabel("|B| (G)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # Energy overlay
    T = constants.joule_to_microKelvin(E)

    ax2 = ax1.twinx()
    ax2.plot(z_vals, T, "ro", markersize=1, label="Energy [μK]")
    ax2.set_ylabel("Energy (μK)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    # set title
    fig.gca().set_title(f"Magnetic Field & Energy at (x={x:.2g} mm, y={y:.2g} mm)", y=1.05)
    fig.tight_layout()
    return fig
