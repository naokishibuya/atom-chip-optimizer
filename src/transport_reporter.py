import argparse
import datetime
import json
import os
import uuid
import textwrap
from typing import List, Tuple
import jax.numpy as jnp
import matplotlib.pyplot as plt
import atom_chip as ac
import transport_initializer


SHIFTING_WIRES, _ = transport_initializer.setup_wire_layout()


# ----------------------------------------------------------------------------------------------------
# Utility class for dictionary key access
# ----------------------------------------------------------------------------------------------------
class attrdict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


# ----------------------------------------------------------------------------------------------------
# Report optimization results, save them to a directory, and plot the results.
# ----------------------------------------------------------------------------------------------------
def report_results(params: attrdict, results: attrdict, error_log: List):
    print(params)
    results_dir = create_results_dir()
    save_results(params, results, error_log, results_dir)
    plot_results(params, results, results_dir)


# ----------------------------------------------------------------------------------------------------
# Save and load results to/from a JSON file.
# ----------------------------------------------------------------------------------------------------
def create_results_dir(base_dir="results"):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")  # YYYYMMDD_HHMMSS_microseconds
    run_id_suffix = uuid.uuid4().hex[:6]  # Short unique ID

    folder_name = f"{timestamp}_{run_id_suffix}"
    full_path = os.path.join(base_dir, folder_name)

    os.makedirs(full_path, exist_ok=True)
    return full_path


def save_results(params: attrdict, results: attrdict, error_log: List, results_dir: str):
    """
    Save the results into a JSON file
    """
    with open(os.path.join(results_dir, "optimization_params.json"), "w") as f:
        json.dump(params, f, indent=4)

    results = {key: val.tolist() for key, val in results.items()}
    with open(os.path.join(results_dir, "optimization_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_dir}")

    # Save error log if there are any errors
    if error_log:
        with open(os.path.join(results_dir, "optimization_errors.json"), "w") as f:
            json.dump(error_log, f, indent=4)
        print(f"{len(error_log)} errors!!!")


def load_results(results_dir: str) -> Tuple[attrdict, attrdict]:
    """
    Load the results from a JSON file.
    """
    with open(os.path.join(results_dir, "optimization_params.json"), "r") as f:
        params = json.load(f)

    with open(os.path.join(results_dir, "optimization_results.json"), "r") as f:
        results = json.load(f)
    print(f"Results loaded from {results_dir}")

    # Convert lists back to jnp.array except for parameters
    for key in results:
        results[key] = jnp.array(results[key])
    return attrdict(params), attrdict(results)


# ----------------------------------------------------------------------------------------------------
# Plotting functions for the optimization results.
# ----------------------------------------------------------------------------------------------------
# fmt: off
def plot_results(params: attrdict, results: attrdict, results_dir: str = None):
    # Collect x positions of shifting wires for plotting
    shifting_wire_x = jnp.array([wire[0][0] for wire in SHIFTING_WIRES[14:21]])  # Collect x positions of shifting wires

    # Plotting the trap trajectory, currents, U0, and omega.
    suptitle = textwrap.fill(", ".join(
        f"{key}={value}" for key, value in params.items()
    ), width=80)

    fig1, axs1 = plt.subplots(2, 2, figsize=(8, 5))
    fig1.suptitle(f"Optimization Currents: {suptitle}", fontsize=8, y=0.99)
    current_log = results.current_log
    plot_wire_currents   (axs1[0, 0], current_log, wire_indices=[0, 1, 2, 3, 4, 5])
    plot_wire_currents   (axs1[0, 1], current_log, wire_indices=[6, 14])
    plot_wire_currents   (axs1[1, 0], current_log, wire_indices=[7, 8, 12, 13])
    plot_wire_currents   (axs1[1, 1], current_log, wire_indices=[9, 10, 11])
    fig1.tight_layout()

    fig2, axs2 = plt.subplots(2, 4, figsize=(18, 6))
    fig2.suptitle(f"Optimization Trajectory: {suptitle}", fontsize=8, y=0.99)
    trajectory, target_rs = results.trajectory, results.target_rs
    plot_x_over_time     (axs2[0, 0], trajectory, target_rs, shifting_wire_x)
    plot_r_i_over_time   (axs2[0, 1], trajectory, target_rs, axis_index=1)  # y positions
    plot_r_i_over_time   (axs2[0, 2], trajectory, target_rs, axis_index=2)  # z positions
    plot_axial_velocity  (axs2[1, 0], trajectory, axis_index=0)  # x velocity
    plot_axial_velocity  (axs2[1, 1], trajectory, axis_index=1)  # y velocity
    plot_axial_velocity  (axs2[1, 2], trajectory, axis_index=2)  # z velocity
    plot_xy_trajectory   (axs2[0, 3], trajectory, target_rs, shifting_wire_x)
    plot_xz_trajectory   (axs2[1, 3], trajectory, target_rs, shifting_wire_x)
    fig2.tight_layout()

    fig3, axs3 = plt.subplots(2, 3, figsize=(10, 5))
    fig3.suptitle(f"Optimization Geometry: {suptitle}", fontsize=8, y=0.99)
    plot_trap_potential  (axs3[0, 0], results.U0s)
    plot_mu_values       (axs3[0, 1], results.mu_vals)
    plot_adiabaticity    (axs3[0, 2], trajectory, results.omegas, params.transport_time)
    plot_trap_frequencies(axs3[1, 0], results.omegas)
    plot_trap_radii      (axs3[1, 1], results.BEC_radii, title="BEC Radii")
    plot_trap_radii      (axs3[1, 2], results.TF_radii, title=f"TF Radii ({params.n_atoms} atoms)")
    fig3.tight_layout()

    adjust_plot_window_geometry(fig1, None, offset_x=100, offset_y=100)
    adjust_plot_window_geometry(fig2, fig1, offset_y=40)
    adjust_plot_window_geometry(fig3, fig1, offset_x=10)

    save_plot(fig1, "currents", results_dir)
    save_plot(fig2, "trajectory", results_dir)
    save_plot(fig3, "geometry", results_dir)
    plt.show()
# fmt: on


def adjust_plot_window_geometry(fig: plt.Figure, ref_fig: plt.Figure = None, offset_x: int = 0, offset_y: int = 0):
    """
    Adjust the geometry of a plot window based on a reference figure.
    """
    win = fig.canvas.manager.window
    geo = win.geometry()
    width, height = geo.width(), geo.height()

    if ref_fig is None:
        # If no reference figure, just use the current window
        top, left = offset_x, offset_y
    else:
        ref_geo = ref_fig.canvas.manager.window.geometry()
        top = ref_geo.bottom() + offset_y if offset_y > 0 else ref_geo.top()
        left = ref_geo.right() + offset_x if offset_x > 0 else ref_geo.left()

    win.setGeometry(left, top, width, height)
    return win.geometry()


def save_plot(fig: plt.Figure, title: str, results_dir: str):
    """
    Save the plot to a file.
    """
    if results_dir is not None:
        fig.savefig(os.path.join(results_dir, f"optimization_{title}.png"), dpi=300)


def plot_x_over_time(ax: plt.Axes, trajectory: jnp.ndarray, trajectory_ref: jnp.ndarray, shifting_wire_x: jnp.ndarray):
    # Plot x over time.
    x = trajectory[:, 0]
    x_ref = trajectory_ref[:, 0]
    for wire_x in shifting_wire_x:
        ax.axhline(wire_x, linestyle="--", color="gray", linewidth=0.5)
    ax.plot(x_ref, label="Target x position", color="orange", marker=".", markersize=0.1)
    ax.plot(x, label="x position", markersize=0.1)
    ax.set_title("Trap x Position Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel("x (mm)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_r_i_over_time(ax: plt.Axes, trajectory: jnp.ndarray, trajectory_ref: jnp.ndarray, axis_index: int):
    # Plot y over time.
    r_i = trajectory[:, axis_index]
    r_i_ref = trajectory_ref[:, axis_index]
    axis = ["x", "y", "z"][axis_index]
    ax.plot(r_i, label=f"{axis} positions", markersize=0.1)
    ax.plot(r_i_ref, label=f"Target {axis} position", color="orange", marker=".", markersize=0.1)
    ax.set_title(f"Trap {axis} Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel(f"{axis} (mm)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_xy_trajectory(
    ax: plt.Axes, trajectory: jnp.ndarray, trajectory_ref: jnp.ndarray, shifting_wire_x: jnp.ndarray
):
    # Plot x-y trap trajectory.
    x, y = trajectory[:, 0], trajectory[:, 1]
    x_ref, y_ref = trajectory_ref[:, 0], trajectory_ref[:, 1]
    for wire_x in shifting_wire_x:
        ax.axvline(wire_x, linestyle="--", color="gray", linewidth=0.5)
    ax.plot(x_ref, y_ref, label="Target x-y position", color="orange", linewidth=1.0)
    ax.plot(x, y, label="x-y position", markersize=0.1)
    ax.set_title("Trap Trajectory (x-y)", fontsize=8, pad=0.0)
    ax.set_xlabel("x (mm)", fontsize=8, labelpad=0.0)
    ax.set_ylabel("y (mm)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_xz_trajectory(
    ax: plt.Axes, trajectory: jnp.ndarray, trajectory_ref: jnp.ndarray, shifting_wire_x: jnp.ndarray
):
    # Plot x-z trap trajectory.
    x, z = trajectory[:, 0], trajectory[:, 2]
    x_ref, z_ref = trajectory_ref[:, 0], trajectory_ref[:, 2]
    for wire_x in shifting_wire_x:
        ax.axvline(wire_x, linestyle="--", color="gray", linewidth=0.5)
    ax.plot(x_ref, z_ref, label="Target x-z position", color="orange", linewidth=1.0)
    ax.plot(x, z, label="x-z position", markersize=0.1)
    ax.set_title("Trap Trajectory (x-z)", fontsize=8, pad=0.0)
    ax.set_xlabel("x (mm)", fontsize=8, labelpad=0.0)
    ax.set_ylabel("z (mm)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_axial_velocity(ax: plt.Axes, trajectory: jnp.ndarray, axis_index: int = 0):
    # Plot axial velocity over time.
    # Calculate the velocity as the difference between consecutive positions
    velocities = jnp.diff(trajectory[:, axis_index])  # dx/dt, dy/dt, dz/dt
    axis = ["x", "y", "z"][axis_index]
    ax.plot(velocities, label="Axial Velocity", markersize=0.1)
    ax.set_title(f"Axial Velocity ($d{axis}/dt$) Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel("Velocity (mm/step)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_wire_currents(ax: plt.Axes, current_log: jnp.ndarray, wire_indices: List[int]):
    # Plot wire currents over time.
    for i in wire_indices:
        ax.plot(current_log[:, i], label=f"Wire {i}", markersize=0.1)
    ax.set_title("Wire Currents Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel("Current (A)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small", ncol=2)


def plot_trap_potential(ax: plt.Axes, U0_vals: List[float]):
    # Plot U0 (trap potential energy) over time.
    ax.plot(U0_vals, label="U0", markersize=0.1)
    ax.set_title("Trap Potential Energy (U0) Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel("U0 (J)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8, pad=0.3)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_trap_frequencies(ax: plt.Axes, omega_vals: jnp.ndarray):
    # Plot trap frequencies over time.
    for i, var in enumerate(["$\\omega_x$", "$\\omega_y$", "$\\omega_z$"]):
        ax.plot(omega_vals[:, i], label=var, markersize=0.1)
    ax.set_title("Trap Frequencies Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel("Frequency (Hz)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_trap_radii(ax: plt.Axes, radii_vals: jnp.ndarray, title: str):
    # Plot BEC radii over time.
    for i, var in enumerate(["$r_x$", "$r_y$", "$r_z$"]):
        ax.plot(radii_vals[:, i], label=f"{var} {i + 1}", markersize=0.1)
    ax.set_title(f"{title} Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel("Radius (m)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_mu_values(ax: plt.Axes, mu_vals: jnp.ndarray):
    # Plot chemical potential over time.
    ax.plot(mu_vals, label="Chemical Potential", markersize=0.1)
    ax.set_title("Chemical Potential Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel("Chemical Potential (J)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_adiabaticity(ax: plt.Axes, trajectory: jnp.ndarray, omega_vals: jnp.ndarray, dt: float):
    v_x = jnp.diff(trajectory[:, 0]) / dt  # x velocity
    omega_x = omega_vals[:-1, 0] * 2 * jnp.pi  # Convert to rad/s
    atom = transport_initializer.ATOM
    sigma_x = jnp.sqrt(ac.constants.hbar / (2 * atom.mass * omega_x))  # Position uncertainty
    eps = jnp.abs(v_x) / (omega_x * sigma_x)  # Adiabaticity parameter
    ax.plot(eps, lw=0.8, c="tab:purple")
    ax.set_title("Adiabatic parameter ε(t)", fontsize=8, pad=0.0)
    ax.set_ylabel("ε", fontsize=8)


# ----------------------------------------------------------------------------------------------------
# Main entry point for the script.
# ----------------------------------------------------------------------------------------------------
def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Transport Results Visualization")
    parser.add_argument("--results_dir",    type=str, default=None, help="Results directory")
    parser.add_argument("--transport_time", type=float, default=None, help="Transport time in seconds")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to the results directory")
    args = parser.parse_args()
    # fmt: on

    # Load results from a CSV file if a path is provided
    params, results = load_results(args.results_dir)
    if args.transport_time is not None:
        params.transport_time = args.transport_time  # Override transport time if provided
    results_dir = args.results_dir if args.save_plots else None
    plot_results(params, results, results_dir)


if __name__ == "__main__":
    main()
