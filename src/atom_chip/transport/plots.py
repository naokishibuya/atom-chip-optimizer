from typing import List
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
import jax
import jax.numpy as jnp
import atom_chip as ac
from . import builder, planner


def show(
    positions: jnp.ndarray,
    desired_r: jnp.ndarray,
    I_schedule: jnp.ndarray,
    depths: jnp.ndarray,
    frequencies: jnp.ndarray,
    search_grid: jnp.ndarray,
):
    # Plot the results using the given current schedule
    figs = []
    figs.append(plot_trap_positions(positions, desired_r))
    figs.append(plot_current_schedule(I_schedule))
    figs.append(plot_trap_depths(depths))
    # figs.append(plot_trap_frequencies(frequencies))
    # figs.append(plot_potential_3d_along_trajectory(I_schedule, positions, search_grid))
    # figs.append(plot_trajectory_xy(I_schedule, positions, width=1e-2, resolution=40, every=5))
    adjust_plot_positions(figs)
    input("Press Enter to close the plots...")


# Plot the actual and desired trap positions over time
def plot_trap_positions(positions, desired):
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, label in enumerate(["x", "y", "z"]):
        axs[i].plot(positions[:, i], label="Actual", color="blue")
        axs[i].plot(desired[:, i], "--", label="Desired", color="green")
        axs[i].set_ylabel(label + " (mm)")
        axs[i].legend()
        axs[i].grid()
    axs[-1].set_xlabel("Step")
    plt.tight_layout()
    return fig


# Plot current schedules for shifting and guiding wires
def plot_current_schedule(I_schedule):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for k, ax in enumerate(axes):
        if k == 0:
            for i in range(planner.NUM_SHIFTING_WIRES):
                ax.plot(I_schedule[i], label=f"Shifting Wire {i}")
            ax.set_title("Shifting Currents")
        else:
            wire_range = [[1, 2, 6, 7], [3, 4, 5], [0, 8]][k - 1]
            for i in wire_range:
                ax.plot(I_schedule[planner.NUM_SHIFTING_WIRES + i], label=f"Guiding Wire {i}")
            ax.set_title(f"Guiding Currents {wire_range}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.set_xlabel("Step")
        ax.set_ylabel("Current (A)")
        ax.grid()

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # leave space for legend
    return fig


def plot_trap_depths(depths):
    # Plot U0 (trap depth)
    fig = plt.figure(figsize=(8, 4))
    plt.plot(depths)
    plt.xlabel("Step")
    plt.ylabel("Trap Minimum $U_0$")
    plt.title("Trap Depth Over Transport")
    plt.grid()
    plt.tight_layout()
    return fig


def plot_trap_frequencies(frequencies):
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = jnp.arange(planner.NUM_STEPS + 1)
    labels = ["$\\omega_x$", "$\\omega_y$", "$\\omega_z$"]
    for i in range(3):
        ax.plot(steps, frequencies[:, i], label=labels[i])
    ax.set_xlabel("Step")
    ax.set_ylabel("Trap Frequency (Hz)")
    ax.set_title("Trap Frequencies Over Transport")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    return fig


def plot_potential_3d_along_trajectory(I_schedule, positions, search_grid):
    all_U0, all_points = [], []
    for step in range(planner.NUM_STEPS + 1):
        # Build the atom chip for the current step
        currents = I_schedule[:, step]
        atom_chip = builder.build_atom_chip(
            planner.SHIFTING_WIRES,
            currents[: planner.NUM_SHIFTING_WIRES],
            planner.GUIDING_WIRES,
            currents[planner.NUM_SHIFTING_WIRES :],
            planner.BIAS_FIELDS,
        )

        # Evaluate the trap position and potential energy at the search grid
        center = positions[step]
        grid_points = center + search_grid
        Us = jax.vmap(atom_chip.potential_energy)(grid_points)
        all_U0.append(Us)
        all_points.append(grid_points)

    # Convert lists to arrays
    all_U0 = jnp.concatenate(all_U0)  # shape (N_total,)
    all_points = jnp.vstack(all_points)  # shape (N_total, 3)
    x, y, z = all_points.T

    # Sort the points by potential energy for better visualization
    sort_idx = jnp.argsort(all_U0)
    x, y, z, all_U0 = x[sort_idx], y[sort_idx], z[sort_idx], all_U0[sort_idx]

    all_U0 = jnp.vectorize(ac.constants.joule_to_microKelvin)(all_U0)

    # Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    # Plot the trap path
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "k-", linewidth=1.5, label="Trap path")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color="black", s=5)
    # Scatter plot of the potential energy along the path
    sc = ax.scatter(x, y, z, c=all_U0, cmap="jet", s=3)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Potential Energy ($\\mu$K)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.set_title("Magnetic Potential Along Transport Path")
    plt.tight_layout()
    return fig


def plot_trajectory_xy(I_schedule, positions, width=1e-2, resolution=40, every=5):
    """
    Plots a sequence of x-y contours along the trajectory at every `every` steps.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for step in range(0, planner.NUM_STEPS + 1, every):
        r0 = positions[step]
        currents = I_schedule[:, step]
        atom_chip = builder.build_atom_chip(
            planner.SHIFTING_WIRES,
            currents[: planner.NUM_SHIFTING_WIRES],
            planner.GUIDING_WIRES,
            currents[planner.NUM_SHIFTING_WIRES :],
            planner.BIAS_FIELDS,
        )

        lin = jnp.linspace(-width / 2, width / 2, resolution)
        dx, dy = jnp.meshgrid(lin, lin, indexing="ij")
        dz = jnp.zeros_like(dx)

        grid = jnp.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=-1)
        grid_points = grid + r0

        Us = jax.vmap(atom_chip.potential_energy)(grid_points)
        U_grid = Us.reshape(resolution, resolution)

        # Draw contours
        levels = jnp.linspace(jnp.min(U_grid), jnp.max(U_grid), 10)
        ax.contour((dx + r0[0]) * 1e3, (dy + r0[1]) * 1e3, U_grid, levels=levels, alpha=0.6, cmap="viridis")

        # Mark trap position
        ax.plot(r0[0] * 1e3, r0[1] * 1e3, "rx", markersize=4)

    ax.set_title("x–y Potential Contours Along Transport Path")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.axis("equal")
    ax.grid(True)
    plt.tight_layout()
    return fig


def adjust_plot_positions(figs: List[plt.Figure]):
    # Adjust the positions of multiple plot windows based on their geometry
    screen_width, screen_height = QApplication.desktop().screenGeometry().getRect()[2:]

    # Position settings
    global_top, global_left = 100, 100  # Initial global top, left position
    global_bottom = 0  # Track the bottom of the last window
    top, left = global_top, global_left

    for fig in figs:
        window = fig.canvas.manager.window
        width, height = window.geometry().getRect()[2:]
        # Set the position based on the geometry
        if left + width > screen_width:
            top = global_bottom + 40
            left = global_left
            if top + height > screen_height:
                top = global_top
                global_bottom = 0  # Reset global bottom if we wrap around
        window.setGeometry(left, top, width, height)
        # Update the global geometry
        global_bottom = max(global_bottom, top + height)
        top = top
        left = left + width + 10
    plt.show(block=False)
