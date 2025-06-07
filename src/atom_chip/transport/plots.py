from typing import List
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
import jax.numpy as jnp
import atom_chip as ac
from .metrics import simulate_trap_dynamics


def show(
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.field.BiasFieldConfig,
    trap_trajectory: jnp.ndarray,
    I_schedule: jnp.ndarray,
    loss_log: dict[str, list[float]],
):
    # This uses the optimized currents and trajectory
    U0s, omegas = simulate_trap_dynamics(
        atom=ac.rb87,
        wire_config=wire_config,
        bias_config=bias_config,
        trap_trajectory=trap_trajectory,
        I_schedule=I_schedule,
    )

    # Plot the results using the given current schedule
    figs = []
    # figs.append(plot_anchor_currents(anchor_currents))
    figs.append(plot_trajectory(trap_trajectory))
    # figs.append(plot_trap_positions(initial_trap_trajectory, trap_trajectory))
    figs.append(plot_trap_dynamics(U0s))
    figs.append(plot_trap_frequencies(omegas))
    figs.append(plot_current_schedule(I_schedule))
    figs.append(plot_loss_components(loss_log))
    adjust_plot_positions(figs)
    input("Press Enter to close the plots...")


def plot_trajectory(trap_trajectory: jnp.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*trap_trajectory.T, marker=".")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Optimized Trap Trajectory")
    return fig


# Plot the actual and desired trap positions over time
def plot_trap_positions(initial_trap_trajectory: jnp.ndarray, trap_trajectory: jnp.ndarray):
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, label in enumerate(["x", "y", "z"]):
        axs[i].plot(trap_trajectory[:, i], label="Actual", color="blue")
        axs[i].plot(initial_trap_trajectory[:, i], "--", label="Desired", color="green")
        axs[i].set_ylabel(label + " (mm)")
        axs[i].legend()
        axs[i].grid()
    axs[-1].set_xlabel("Step")
    plt.tight_layout()
    return fig


def plot_anchor_currents(anchor_currents: jnp.ndarray):
    fig = plt.figure()
    for i in range(anchor_currents.shape[1]):
        plt.plot(anchor_currents[:, i], label=f"Wire {i}")
    plt.title("Optimized Anchor Currents per Wire")
    plt.xlabel("Anchor Index")
    plt.ylabel("Current (A)")
    plt.legend(ncol=3, fontsize=6)
    return fig


def plot_trap_dynamics(U0s: jnp.ndarray):
    fig = plt.figure()
    plt.plot(U0s * 1e30)  # Convert to x10^-30 J for readability
    plt.xlabel("Time Step")
    plt.ylabel("Trap Energy $U_0$ [$10^{-30}$ J]")
    plt.title("Trap Potential Energy Over Time")
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_trap_frequencies(frequencies):
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = jnp.arange(frequencies.shape[0])
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
        ax.plot(I_np[i], label=f"Shift {i}")
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
            ax.plot(I_np[idx], label=f"Guide {i}")
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
    ncols = 4
    nrows = n // 4 + (n % 4 > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10), sharex=True)

    for i, (key, values) in enumerate(loss_log.items()):
        ax = axes[i // ncols, i % ncols]
        ax.plot(values)
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
