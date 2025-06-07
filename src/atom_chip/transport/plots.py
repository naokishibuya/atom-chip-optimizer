from typing import List
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
import jax.numpy as jnp


def show(
    trap_trajectory: jnp.ndarray,  # desired trap trajectory
    I_schedule: jnp.ndarray,
    r0s: jnp.ndarray,
    U0s: jnp.ndarray,
    omegas: jnp.ndarray,
    loss_log: dict[str, list[float]],
):
    # Plot the results using the given current schedule
    figs = []
    figs.append(plot_trap_positions(trap_trajectory, r0s))
    figs.append(plot_trap_minimum_positions(r0s))
    figs.append(plot_trap_minimum_energies(U0s))
    figs.append(plot_trap_frequencies(omegas))
    figs.append(plot_current_schedule(I_schedule))
    figs.append(plot_loss_components(loss_log))
    adjust_plot_positions(figs)

    # save the figures to files
    for i, fig in enumerate(figs):
        fig.savefig(f"transport-optimizer_{i}.png", dpi=300, bbox_inches="tight")

    input("Press Enter to close the plots...")


# Plot the reconstructed and desired trap positions over time
def plot_trap_positions(target_trajectory: jnp.ndarray, r0s: jnp.ndarray):
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, label in enumerate(["x", "y", "z"]):
        axs[i].plot(r0s[:, i], label="Simulated", color="blue", marker=".", markersize=3)
        axs[i].plot(target_trajectory[:, i], "--", label="Desired", color="green")
        axs[i].set_ylabel(label + " (mm)")
        axs[i].legend()
        axs[i].grid()
    axs[-1].set_xlabel("Step")
    plt.tight_layout()
    return fig


def plot_trap_minimum_positions(r0s: jnp.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*r0s.T, marker=".", markersize=3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Optimized Trap Trajectory")
    return fig


def plot_trap_minimum_energies(U0s: jnp.ndarray):
    fig = plt.figure()
    plt.plot(U0s * 1e28, marker=".", markersize=3)  # Convert to x10^-30 J for readability
    plt.xlabel("Time Step")
    plt.ylabel("Trap Energy $U_0$ [$10^{-28}$ J]")
    plt.title("Trap Potential Energy Over Time")
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_trap_frequencies(frequencies):
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = jnp.arange(frequencies.shape[0])
    labels = ["$\\omega_x$", "$\\omega_y$", "$\\omega_z$"]
    for i in range(3):
        ax.plot(steps, frequencies[:, i], label=labels[i], marker=".", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Trap Frequency (Hz)")
    ax.set_title("Trap Frequencies Over Transport")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    return fig


def plot_current_schedule(I_schedule: jnp.ndarray):
    """
    Plot time-dependent wire currents for shifting and guiding wires.
    Args:
        I_schedule: shape (n_steps, n_wires)
    """
    I_schedule = jnp.asarray(I_schedule)  # ensure it’s not a tracer
    I_np = jnp.array(I_schedule).T  # shape: (n_wires, n_steps)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    # Panel 1: shifting wires (first 6)
    NUM_SHIFTING_WIRES = 6
    ax = axes[0]
    for i in range(NUM_SHIFTING_WIRES):
        ax.plot(I_np[i], label=f"Shift {i}", marker=".", markersize=3)
    ax.set_title("Shifting Wire Currents")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_ylabel("Current (A)")
    ax.grid()

    # Panels 2–4: guiding wire subsets
    guiding_indices = [[1, 2, 6, 7], [3, 4, 5], [0, 8]]
    for k, group in enumerate(guiding_indices):
        ax = axes[k + 1]
        for i in group:
            idx = NUM_SHIFTING_WIRES + i
            ax.plot(I_np[idx], label=f"Guide {i}", marker=".", markersize=3)
        ax.set_title(f"Guiding Wires: {group}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.set_ylabel("Current (A)")
        ax.grid()

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    return fig


def plot_loss_components(loss_log: dict[str, list[float]]):
    n = len(loss_log)
    ncols = 3
    nrows = n // ncols + (n % ncols > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 5), sharex=True)

    for i, (key, values) in enumerate(loss_log.items()):
        ax = axes[i // ncols, i % ncols]
        ax.plot(values, marker=".", markersize=3)
        ax.set_ylabel(key)
        ax.grid(True)

    fig.suptitle("Loss Components Over Optimization Steps", fontsize=12)
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
