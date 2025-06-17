import argparse
import datetime
import json
import os
import uuid
import textwrap
from typing import List, Tuple
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
def plot_results(
    params: attrdict, results: attrdict, save_dir: str = None, start_step: int = None, end_step: int = None):
    # Check step range
    if start_step is not None or end_step is not None:
        print(f"Plotting results from step {start_step} to {end_step}.")
        if start_step is None:
            start_step = 0
        if end_step is None:
            end_step = len(results.trajectory)
        steps = jnp.arange(start_step, end_step)
        results = attrdict({key: val[start_step:end_step] for key, val in results.items()})
    else:
        steps = jnp.arange(len(results.trajectory))

    # Collect x positions of shifting wires for plotting
    shifting_wire_x = jnp.array([wire[0][0] for wire in SHIFTING_WIRES[14:21]])  # Collect x positions of shifting wires

    # Plotting the trap trajectory, currents, U0, and omega.
    def suptitle(width: int = None, include_time: bool = False) -> str:
        title = ", ".join(f"{key}={value}" for key, value in params.items() if include_time or key != "transport_time")
        return textwrap.fill(title, width=width) if width else title

    fig1, axs1 = plt.subplots(2, 2, figsize=(7, 5))
    fig1.suptitle(f"Optimization Currents: {suptitle(80)}", fontsize=8)
    current_log = results.current_log
    plot_wire_currents   (axs1[0, 0], steps, current_log, wire_indices=[0, 1, 2, 3, 4, 5])
    plot_wire_currents   (axs1[0, 1], steps, current_log, wire_indices=[6, 14])
    plot_wire_currents   (axs1[1, 0], steps, current_log, wire_indices=[7, 8, 12, 13])
    plot_wire_currents   (axs1[1, 1], steps, current_log, wire_indices=[9, 10, 11])
    fig1.tight_layout()

    fig2, axs2 = plt.subplots(2, 4, figsize=(18, 6))
    fig2.suptitle(f"Optimization Trajectory: {suptitle()}", fontsize=8)
    trajectory, target_rs = results.trajectory, results.target_rs
    plot_x_over_time     (axs2[0, 0], steps, trajectory, target_rs, shifting_wire_x)
    plot_r_i_over_time   (axs2[0, 1], steps, trajectory, target_rs, axis_index=1)  # y positions
    plot_r_i_over_time   (axs2[0, 2], steps, trajectory, target_rs, axis_index=2)  # z positions
    plot_axial_velocity  (axs2[1, 0], steps, trajectory, axis_index=0)  # x velocity
    plot_axial_velocity  (axs2[1, 1], steps, trajectory, axis_index=1)  # y velocity
    plot_axial_velocity  (axs2[1, 2], steps, trajectory, axis_index=2)  # z velocity
    plot_xy_trajectory   (axs2[0, 3], steps, trajectory, target_rs, shifting_wire_x)
    plot_xz_trajectory   (axs2[1, 3], steps, trajectory, target_rs, shifting_wire_x)
    fig2.tight_layout()

    fig3, axs3 = plt.subplots(2, 3, figsize=(11, 5))
    fig3.suptitle(f"Optimization Geometry: {suptitle(150)}", fontsize=8)
    plot_trap_potential  (axs3[0, 0], steps, results.U0s)
    plot_mu_values       (axs3[0, 1], steps, results.mu_vals)
    plot_adiabaticity    (axs3[0, 2], steps, trajectory, results.omegas, results.TF_radii, params.transport_time)
    plot_trap_frequencies(axs3[1, 0], steps, results.omegas)
    plot_trap_radii      (axs3[1, 1], steps, results.BEC_radii, title="BEC Radii")
    plot_trap_radii      (axs3[1, 2], steps, results.TF_radii, title=f"TF Radii ({params.n_atoms} atoms)")
    fig3.tight_layout()

    adjust_plot_window_geometry(fig1, None, offset_x=100, offset_y=100)
    adjust_plot_window_geometry(fig2, fig1, offset_y=40)
    adjust_plot_window_geometry(fig3, fig1, offset_x=10)

    save_plot(fig1, "currents", save_dir)
    save_plot(fig2, "trajectory", save_dir)
    save_plot(fig3, "geometry", save_dir)
    plt.show(block=False)
    input("Press Enter to close the plots.")
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


def save_plot(fig: plt.Figure, title: str, save_dir: str):
    """
    Save the plot to a file.
    """
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, f"optimization_{title}.png"), dpi=300)


def plot_x_over_time(
    ax: plt.Axes, steps: jnp.ndarray, trajectory: jnp.ndarray, trajectory_ref: jnp.ndarray, shifting_wire_x: jnp.ndarray
):
    """
    Plot x over time.
    """
    x = trajectory[:, 0]
    x_ref = trajectory_ref[:, 0]
    for wire_x in shifting_wire_x:
        if wire_x in x:
            ax.axhline(wire_x, linestyle="--", color="gray", linewidth=0.5)
    ax.plot(steps, x_ref, label="Target x position", color="orange", linewidth=1.0)
    ax.plot(steps, x, label="x position", linewidth=1.0)
    ax.set_title("Trap x Position Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel("x (mm)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_r_i_over_time(
    ax: plt.Axes, steps: jnp.ndarray, trajectory: jnp.ndarray, trajectory_ref: jnp.ndarray, axis_index: int
):
    """
    Plot y over time.
    """
    r_i = trajectory[:, axis_index]
    r_i_ref = trajectory_ref[:, axis_index]
    axis = ["x", "y", "z"][axis_index]
    ax.plot(steps, r_i_ref, label=f"Target {axis} position", color="orange", linewidth=1.0)
    ax.plot(steps, r_i, label=f"{axis} positions", linewidth=1.0)
    ax.set_title(f"Trap {axis} Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel(f"{axis} (mm)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_xy_trajectory(
    ax: plt.Axes, steps: jnp.ndarray, trajectory: jnp.ndarray, trajectory_ref: jnp.ndarray, shifting_wire_x: jnp.ndarray
):
    """
    Plot x-y trap trajectory.
    """
    x, y = trajectory[:, 0], trajectory[:, 1]
    x_ref, y_ref = trajectory_ref[:, 0], trajectory_ref[:, 1]
    for wire_x in shifting_wire_x:
        if wire_x in x:
            ax.axvline(wire_x, linestyle="--", color="gray", linewidth=0.5)
    ax.plot(x_ref, y_ref, label="Target x-y position", color="orange", linewidth=1.0)
    ax.plot(x, y, label="x-y position", linewidth=1.0)
    ax.set_title("Trap Trajectory (x-y)", fontsize=8, pad=0.0)
    ax.set_xlabel("x (mm)", fontsize=8, labelpad=0.0)
    ax.set_ylabel("y (mm)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.xaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))


def plot_xz_trajectory(
    ax: plt.Axes, steps: jnp.ndarray, trajectory: jnp.ndarray, trajectory_ref: jnp.ndarray, shifting_wire_x: jnp.ndarray
):
    """
    Plot x-z trap trajectory.
    """
    x, z = trajectory[:, 0], trajectory[:, 2]
    x_ref, z_ref = trajectory_ref[:, 0], trajectory_ref[:, 2]
    for wire_x in shifting_wire_x:
        if wire_x in x:  # Only plot vertical lines for wires that are in the x range
            ax.axvline(wire_x, linestyle="--", color="gray", linewidth=0.5)
    ax.plot(x_ref, z_ref, label="Target x-z position", color="orange", linewidth=1.0)
    ax.plot(x, z, label="x-z position", linewidth=1.0)
    ax.set_title("Trap Trajectory (x-z)", fontsize=8, pad=0.0)
    ax.set_xlabel("x (mm)", fontsize=8, labelpad=0.0)
    ax.set_ylabel("z (mm)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.xaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))


def plot_axial_velocity(ax: plt.Axes, steps: jnp.ndarray, trajectory: jnp.ndarray, axis_index: int = 0):
    """
    Plot axial velocity over time.
    """
    # Calculate the velocity as the difference between consecutive positions
    velocities = jnp.diff(trajectory[:, axis_index])  # dx/dt, dy/dt, dz/dt
    axis = ["x", "y", "z"][axis_index]
    ax.plot(steps[1:], velocities, label="Axial Velocity", linewidth=1.0)
    ax.set_title(f"Axial Velocity ($d{axis}/dt$) Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel("Velocity (mm/step)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_wire_currents(ax: plt.Axes, steps: jnp.ndarray, current_log: jnp.ndarray, wire_indices: List[int]):
    """
    Plot wire currents over time.
    """
    for i in wire_indices:
        ax.plot(steps, current_log[:, i], label=f"Wire {i}", linewidth=1.0)
    ax.set_title("Wire Currents Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel("Current (A)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small", ncol=2)


def plot_trap_potential(ax: plt.Axes, steps: jnp.ndarray, U0_vals: List[float]):
    """
    Plot U0 (trap potential energy) over time.
    """
    ax.plot(steps, U0_vals, label="U0", linewidth=1.0)
    ax.set_title("Trap Potential Energy (U0) Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel("U0 (J)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8, pad=0.3)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_trap_frequencies(ax: plt.Axes, steps: jnp.ndarray, omega_vals: jnp.ndarray):
    """
    Plot trap frequencies over time.
    """
    for i, var in enumerate(["$\\omega_x$", "$\\omega_y$", "$\\omega_z$"]):
        ax.plot(steps, omega_vals[:, i], label=var, linewidth=1.0)
    ax.set_title("Trap Frequencies Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel("Frequency (Hz)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_trap_radii(ax: plt.Axes, steps: jnp.ndarray, radii_vals: jnp.ndarray, title: str):
    """
    Plot trap radii over time.
    """
    for i, var in enumerate(["$r_x$", "$r_y$", "$r_z$"]):
        ax.plot(steps, radii_vals[:, i], label=f"{var} {i + 1}", linewidth=1.0)
    ax.set_title(f"{title} Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel("Radius (m)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_mu_values(ax: plt.Axes, steps: jnp.ndarray, mu_vals: jnp.ndarray):
    """
    Plot chemical potential over time.
    """
    ax.plot(steps, mu_vals, label="Chemical Potential", linewidth=1.0)
    ax.set_title("Chemical Potential Over Time", fontsize=8, pad=0.0)
    ax.set_ylabel("Chemical Potential (J)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


def plot_adiabaticity(
    ax: plt.Axes,
    steps: jnp.ndarray,
    trajectory: jnp.ndarray,
    omega_vals: jnp.ndarray,
    TF_radii: jnp.ndarray,
    transport_time: float,
):
    """
    ε(t) = |v_x| / (ω_x σ_x);  σ_x = TF_radius / sqrt(5)
           the RMS width (standard deviation) of the TF density profile.
           where Thomas-Fermi radius is max extent of the BEC in the x direction.
    Uses centred derivative for v_x.
    """
    n_steps = len(trajectory) - 1
    x = jnp.asarray(trajectory[:, 0]) * 1e-3  # Convert x positions from mm to m
    ωx = omega_vals[:, 0] * 2 * jnp.pi  # rad/s
    sigma_x = TF_radii[:, 0] / jnp.sqrt(5.0)  # The RMS width of the TF density profile

    ax.axhline(0.3, linestyle="--", linewidth=0.5, color="grey")  # Adiabaticity threshold line
    for s in range(-1, 2):
        duration = transport_time + s
        if duration <= 0:
            continue
        dt = duration / n_steps
        v_x = jnp.gradient(x, dt)  # m/s
        eps = jnp.abs(v_x) / (ωx * sigma_x)  # we may remove both ends like [1:-1] since we take the centred derivative
        ax.plot(steps, eps, linewidth=1.0, label=f"time = {duration:.2f} s")
    ax.set_title("Adiabatic parameter ε(t) = |v_x| / (ω_x σ_x)", fontsize=8)
    ax.set_ylabel("ε", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(fontsize="xx-small")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_position((-0.1, 0.0))  # Adjust offset text position


# ----------------------------------------------------------------------------------------------------
# Main entry point for the script.
# ----------------------------------------------------------------------------------------------------
def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Transport Results Visualization")
    parser.add_argument("--results_dir",    type=str, default=None, help="Results directory")
    parser.add_argument("--transport_time", type=float, default=None, help="Transport time in seconds")
    parser.add_argument("--start_step", type=int, default=None, help="Start step for plotting")
    parser.add_argument("--end_step",   type=int, default=None, help="End step for plotting")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save plots")
    args = parser.parse_args()
    # fmt: on

    # Load results from a CSV file if a path is provided
    params, results = load_results(args.results_dir)
    if args.transport_time is not None:
        params.transport_time = args.transport_time  # Override transport time if provided
    plot_results(params, results, args.save_dir, args.start_step, args.end_step)


if __name__ == "__main__":
    main()
