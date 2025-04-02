from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from ..atom_chip import AtomChip
from ..potential import constants, PotentialMinimum


def plot_potential_1d(
    atom_chip: AtomChip,
    E_min: PotentialMinimum,
    size: Tuple[int, int],
    z_range: Tuple[float, float, int],
):
    x, y = E_min.point[:2]
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
    E = E * 1e6 / constants.kB  # Convert to μK using Boltzmann constant (J/K)

    ax2 = ax1.twinx()
    ax2.plot(z_vals, E, "ro", markersize=1, label="Energy [μK]")
    ax2.set_ylabel("Energy (μK)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    # set title
    fig.gca().set_title(f"Magnetic Field & Energy at (x={x:.2g} mm, y={y:.2g} mm)", y=1.05)
    fig.tight_layout()
    return fig
