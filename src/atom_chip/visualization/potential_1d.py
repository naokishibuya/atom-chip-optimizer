from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from ..atom_chip import AtomChip
from ..potential import constants


def plot_potential_1d(
    atom_chip: AtomChip,
    size: Tuple[int, int],
    z_range: Tuple[float, float, int],
    fig: Optional[plt.Figure] = None,
) -> plt.Figure:
    if fig is None:
        fig = plt.figure(figsize=size)
    else:
        fig.clear()
    ax1 = fig.add_subplot(111)
    fig.gca().grid()

    if not atom_chip.potential.minimum.found:
        fig.text(0.5, 0.5, "Potential Minimum not found.", ha="center", va="center", fontsize=12)
        return fig

    # Get the minimum energy point from the atom chip
    x, y, z = atom_chip.potential.minimum.position
    z_vals = np.linspace(*z_range)

    points = np.array([[x, y, z] for z in z_vals])
    E, B_mag, _ = atom_chip.get_potentials(points)
    T = constants.joule_to_microKelvin(E)
    min_T = constants.joule_to_microKelvin(atom_chip.potential.minimum.value)

    # Check the energy around the minimum point
    check_index = np.argmin(np.abs(z_vals - z))
    check_start = max(0, check_index - 1)
    check_end = min(len(z_vals) - 1, check_index + 1)
    checK_z_vals = np.linspace(z_vals[check_start], z_vals[check_end], 100)
    check_points = np.array([[x, y, z] for z in checK_z_vals])
    check_E, _, _ = atom_chip.get_potentials(check_points)
    check_T = constants.joule_to_microKelvin(check_E)

    # Magnetic field
    ax1.plot(z_vals, B_mag, "bo", markersize=1, label="|B| [G]")
    ax1.set_xlabel("z (mm)")
    ax1.set_ylabel("|B| (G)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # Energy overlay
    ax2 = ax1.twinx()
    ax2.plot(z_vals, T, "ro", markersize=1, label="Energy [μK]")
    if np.any(check_T < min_T):  # make sure it's local minimum
        # plot text with a warning
        text = f"z={z:.04f} is not the minimum!"
        ax2.text(z + 0.1, min_T + 50, text, ha="left", va="center", fontsize=12, color="red")
        ax2.hlines(
            min_T,
            z_range[0],
            z_range[1],
            color="r",
            linestyle="-",
            linewidth=1,
            alpha=0.5,
        )
    ax2.set_ylabel("Energy (μK)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    # set title
    fig.gca().set_title(f"Magnetic Field & Energy at (x={x:.2g} mm, y={y:.2g} mm)", y=1.05)
    fig.tight_layout()
    return fig
