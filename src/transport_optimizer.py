import argparse
import datetime
import json
import os
import uuid
from typing import Callable, List, Tuple
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np  # formatting arrays only
from scipy.optimize import minimize
import atom_chip as ac
import transport_initializer


# Precompute the wire layout and segment counts
SHIFTING_WIRES, GUIDING_WIRES = transport_initializer.setup_wire_layout()
GUIDING_WIRE_SEGMENT_COUNTS = jnp.array([len(wire) for wire in GUIDING_WIRES], dtype=jnp.int32)


# ----------------------------------------------------------------------------------------------------
# Setup the wire configuration for the atom chip.
#
# This can be JIT-compiled but it's not necessary since it runs only once.
# ----------------------------------------------------------------------------------------------------
def setup_wire_config() -> ac.atom_chip.WireConfig:
    """
    Build the wire configuration for the atom chip.
    """
    starts = []
    ends = []
    widths = []
    heights = []

    # Collect segments for shifting and guiding wires
    for segment in SHIFTING_WIRES:
        s, e, w, h = segment
        starts.append(jnp.array(s, dtype=jnp.float64))
        ends.append(jnp.array(e, dtype=jnp.float64))
        widths.append(jnp.array(w, dtype=jnp.float64))
        heights.append(jnp.array(h, dtype=jnp.float64))

    for wire in GUIDING_WIRES:
        for segment in wire:
            s, e, w, h = segment
            starts.append(jnp.array(s, dtype=jnp.float64))
            ends.append(jnp.array(e, dtype=jnp.float64))
            widths.append(jnp.array(w, dtype=jnp.float64))
            heights.append(jnp.array(h, dtype=jnp.float64))

    # Convert to WireConfig
    wire_config = ac.atom_chip.WireConfig(
        starts=jnp.stack(starts),
        ends=jnp.stack(ends),
        widths=jnp.stack(widths),
        heights=jnp.stack(heights),
    )
    return wire_config


# ----------------------------------------------------------------------------------------------------
# Calculate the target position of the trap.
#
# This can be JIT-compiled but it's not necessary since it runs only once.
# ----------------------------------------------------------------------------------------------------
def calculate_destination(
    r0: jnp.ndarray,
    num_shifts: int,
    shifting_wire_distance: float,  # mm
) -> jnp.ndarray:
    """
    Calculate the target position of the trap after a number of shifts.
    """
    distance = num_shifts * shifting_wire_distance
    return r0 + distance * jnp.array([1.0, 0.0, 0.0])


# ----------------------------------------------------------------------------------------------------
# Find the local minimum of the magnetic potential energy.
#
# This is not a JIT function because it uses `minimize` from SciPy
# ----------------------------------------------------------------------------------------------------
def find_trap_minimum(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.field.BiasConfig,
    wire_currents: jnp.ndarray,
    guess: jnp.ndarray,
) -> jnp.ndarray:
    """
    Find the local minimum of the magnetic potential energy.
    """

    def objective(r):
        """Compute potential energy at position r."""
        U, _, _ = ac.atom_chip.trap_potential_energies(jnp.atleast_2d(r), atom, wire_config, wire_currents, bias_config)
        return U[0]

    # scipy minimize
    result = minimize(objective, **make_search_options(initial_guess=guess))
    if not result.success:
        print(f"Minimization failed: {result.message} result.x={result.x} guess={guess}")
    return result.x  # shape (3,)


def make_search_options(initial_guess) -> ac.potential.AnalysisOptions:
    """
    Create analysis options for the trap potential search.
    """
    x, y, _ = initial_guess
    bounds = [
        (x - 0.5, x + 0.5),  # x bounds
        (y - 0.5, y + 0.5),  # y bounds
        (0.0, 1.0),  # z bounds
    ]
    return dict(
        x0=initial_guess,  # Initial guess
        bounds=bounds,
        method="Nelder-Mead",
        options=dict(
            xatol=1e-10,
            fatol=1e-10,
            maxiter=int(1e5),
            maxfev=int(1e5),
            disp=False,
        ),
    )


# Utility class for dictionary key access
class attrdict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


# ----------------------------------------------------------------------------------------------------
# Optimize the transport of the atom chip trap.
# ----------------------------------------------------------------------------------------------------
def optimize_transport(params: attrdict):
    # 1. Simulation parameters
    # select which wires to optimize
    wire_ids = jnp.array(params.wire_ids, dtype=jnp.int32)
    mask = jnp.zeros((15,)).at[wire_ids].set(1.0)
    print(f"Transport optimization settings: {params}, mask={mask}")

    # 1st and 2nd derivatives are 0 at s = 0 and s = 1
    def smoothstep_quintic(s: float) -> jnp.ndarray:
        return 10 * s**3 - 15 * s**4 + 6 * s**5  # C² at both ends

    # Cosine schedule (range [0, 1] over finite time steps t over duration T)
    def cosine_schedule(s: float) -> jnp.ndarray:
        return 0.5 * (1 - jnp.cos(jnp.pi * s))

    schedule_func = {
        "smoothstep": smoothstep_quintic,
        "cosine": cosine_schedule,
    }[params.scheduler]

    # 2. Transport initial setup
    wire_config = setup_wire_config()
    bias_config = transport_initializer.setup_bias_config()

    I_shifting_wires = jnp.array(transport_initializer.SHIFTING_WIRE_CURRENTS, dtype=jnp.float64)
    I_guiding_wires = jnp.array(transport_initializer.GUIDING_WIRE_CURRENTS, dtype=jnp.float64)
    I_start = jnp.concatenate([I_shifting_wires, I_guiding_wires])
    I_limits = jnp.concatenate(
        [
            jnp.ones_like(I_shifting_wires) * params.I_max_shifting,  # Shifting wires limits
            jnp.ones_like(I_guiding_wires) * params.I_max_guiding,  # Guiding wires limits
        ]
    )  # shape: (15,)

    # 3. Reference values for the trap
    atom_chip = transport_initializer.build_atom_chip()
    analysis = transport_initializer.analyze_atom_chip(atom_chip)
    r0_ref = analysis.potential.minimum.position
    U0_ref = analysis.potential.minimum.value
    omega_ref = analysis.potential.trap.frequency
    BEC_radii_ref = analysis.potential.bec.radii  # non-interacting BEC radii
    TF_radii_ref = analysis.potential.tf.radii  # Thomas-Fermi radii
    mu_ref = analysis.potential.tf.mu  # Chemical potential
    destination_r = calculate_destination(r0_ref, params.num_shifts, shifting_wire_distance=0.4)  # mm

    print(f"Initial trap pos: {r0_ref}, U0: {U0_ref:.4g}")
    print(f"Trap frequency: {omega_ref} (Hz) radii: {BEC_radii_ref} (m) TF: {TF_radii_ref} (m) mu: {mu_ref:.4g} (J)")
    print(f"Desired final position: {destination_r}")

    # 4. Optimize the transport
    results, error_log = generate_schedule(
        atom=atom_chip.atom,
        wire_config=wire_config,
        bias_config=bias_config,
        I_start=I_start,
        I_limits=I_limits,
        mask=mask,
        r0_ref=r0_ref,
        U0_ref=U0_ref,
        omega_ref=omega_ref,
        BEC_radii_ref=BEC_radii_ref,
        TF_radii_ref=TF_radii_ref,
        mu_ref=mu_ref,
        destination_r=destination_r,
        schedule_func=schedule_func,
        T=params.T,
        reg=params.reg,
        n_atoms=params.n_atoms,
    )

    # 5. Save the results
    print(params)
    results_dir = create_results_dir()

    save_results(params, results, error_log, results_dir)

    # 6. Plot the results
    plot_results(params, results, results_dir)


# ----------------------------------------------------------------------------------------------------
# Generate the transport schedule.
# ----------------------------------------------------------------------------------------------------
# fmt: off
def generate_schedule(
    atom           : ac.Atom,
    wire_config    : ac.atom_chip.WireConfig,
    bias_config    : ac.field.BiasConfig,
    I_start        : jnp.ndarray,  # Initial currents for the wires (shape: (15,))
    I_limits       : jnp.ndarray,  # Optional limits for the currents (shape: (15,))
    mask           : jnp.ndarray,  # Mask to restrict current updates (shape: (15,))
    r0_ref         : jnp.ndarray,  # Reference position of the trap at t=0 (shape: (3,))
    U0_ref         : float,        # Reference potential energy of the trap
    omega_ref      : jnp.ndarray,  # Reference trap frequencies (shape: (3,))
    BEC_radii_ref  : jnp.ndarray,  # Reference BEC radii (shape: (3,))
    TF_radii_ref   : jnp.ndarray,  # Reference Thomas-Fermi radii (shape: (3,))
    mu_ref         : float,        # Reference chemical potential
    destination_r  : jnp.ndarray,  # Desired final position of the trap (shape: (3,))
    schedule_func  : Callable[[float], jnp.ndarray],  # Scheduler function
    T              : int,          # Number of time steps
    reg            : float,        # Regularization parameter
    n_atoms        : int,          # Number of atoms in the BEC (for chemical potential calculation)
) -> Tuple[attrdict, List]:
# fmt: on
    @jax.jit
    def trap_U(r: jnp.ndarray, I_wires: jnp.ndarray) -> float:
        wire_currents = distribute_currents_to_wires(I_wires)
        U, _, _ = ac.atom_chip.trap_potential_energies(
            jnp.atleast_2d(r), atom, wire_config, wire_currents, bias_config
        )
        return U[0]

    # Trap gradient
    # fmt: off
    grad_U_r  = jax.grad(trap_U, argnums=0)      # Gradient of potential energy w.r.t. position r
    hess_U_r  = jax.jacfwd(grad_U_r, argnums=0)  # Hessian of potential energy w.r.t. position r
    cross_jac = jax.jacfwd(grad_U_r, argnums=1)  # Cross Jacobian of potential energy w.r.t. wire currents I_wires
    # fmt: on

    # Implicit gradient dr/dI = -H^{-1} @ dgrad/dI
    @jax.jit
    def compute_dr_dI(r0, I_wires):
        H = hess_U_r(r0, I_wires)
        J = cross_jac(r0, I_wires)
        return -jnp.linalg.solve(H, J)

    @jax.jit
    def r_target(t: int, T: int) -> jnp.ndarray:
        s = t / T  # Normalize time step to [0, 1]
        return r0_ref + schedule_func(s) * (destination_r - r0_ref)

    @jax.jit
    def implicit_gradient(I_wires: jnp.ndarray, r_now: jnp.ndarray, r_next: jnp.ndarray) -> jnp.ndarray:
        delta_r = r_next - r_now
        J = compute_dr_dI(r_now, I_wires)
        alpha = reg * (1 + jnp.linalg.cond(J))
        delta_I = jnp.linalg.solve(J.T @ J + alpha * jnp.eye(J.shape[1]), J.T @ delta_r)
        return delta_I

    # fmt: off
    I_wires     = I_start
    trajectory  = [ r0_ref    ]
    target_rs   = [ r0_ref    ]
    current_log = [ I_wires   ]
    U0s         = [ U0_ref    ]
    omegas      = [ omega_ref ]
    BEC_radii   = [ BEC_radii_ref ]   # Non-interacting BEC radii
    TF_radii    = [ TF_radii_ref  ]  # Thomas-Fermi radii
    mu_vals     = [ mu_ref    ]  # Chemical potential
    error_log   = []
    # fmt: on

    # Control loop
    for t in range(T):
        # Compute the current target position
        r_now = trajectory[-1]
        r_next = r_target(t + 1, T)

        # Compute the implicit gradient and update currents
        delta_I = implicit_gradient(I_wires, r_now, r_next)
        I_wires = I_wires + delta_I * mask  # Apply mask to restrict current updates
        I_wires = jnp.clip(I_wires, -I_limits, I_limits)

        # Find the minimum trap position and evaluate the trap
        wire_currents = distribute_currents_to_wires(I_wires)
        r_min = find_trap_minimum(atom, wire_config, bias_config, wire_currents, r_now)

        # Evaluate the trap
        U0, omega, eigenvalues, bec_radii, tf_radii, mu = evaluate_trap(
            atom, wire_config, bias_config, wire_currents, r_min, n_atoms)

        # Check for NaN or negative eigenvalues in omega
        message = " ".join([
            f"Step {t + 1:4d}:",
            f"r_min={format_array(r_min)}",
            f"U0={U0:10.4g}",
            f"omega={format_array(omega)}",
            f"BEC-radii={format_array(bec_radii)}",
            f"TF-radii={format_array(tf_radii)}",
            f"mu={mu:10.4g}",
        ])
        print(message)

        if jnp.any(jnp.isnan(omega)) or jnp.any(eigenvalues < 0):
            error_log.append(f"{message}: NaN or negative eigenvalues!!!")
            continue

        # Log the results
        trajectory.append(r_min)
        target_rs.append(r_next)
        current_log.append(I_wires)
        U0s.append(U0)
        omegas.append(omega)
        BEC_radii.append(bec_radii)
        TF_radii.append(tf_radii)
        mu_vals.append(mu)

    if error_log:
        print(f"{len(error_log)} errors encountered during optimization!")
    else:
        print("Optimization completed successfully without errors.")

    # fmt: off
    results = attrdict(
        trajectory  = jnp.stack(trajectory),
        target_rs   = jnp.stack(target_rs),
        current_log = jnp.stack(current_log),
        U0s         = jnp.array(U0s),
        omegas      = jnp.stack(omegas),
        BEC_radii   = jnp.stack(BEC_radii),
        TF_radii    = jnp.stack(TF_radii),
        mu_vals     = jnp.array(mu_vals),
    )
    # fmt: on
    return results, error_log


def format_array(array: jnp.ndarray) -> str:
    return np.array2string(
        np.array(array),
        formatter={"float_kind": lambda x: f"{x: 10.4g}"},
        separator=" ",
    )


# ----------------------------------------------------------------------------------------------------
# Evaluate the potential energy and curvature at a fixed position.
#
# This function is JIT-compiled for performance.
# ----------------------------------------------------------------------------------------------------
@jax.jit
def evaluate_trap(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.field.BiasConfig,
    wire_currents: jnp.ndarray,
    trap_position: jnp.ndarray,
    n_atoms: int,
) -> Tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """Evaluate potential energy and curvature at a fixed position."""

    def objective(r):
        """Compute potential energy at position r."""
        U, _, _ = ac.atom_chip.trap_potential_energies(jnp.atleast_2d(r), atom, wire_config, wire_currents, bias_config)
        return U[0]

    U0 = objective(trap_position)
    # H = ac.potential.hessian_by_finite_difference(
    #     objective,
    #     jnp.ravel(trap_position),
    #     step=1e-3,
    # )
    # eigenvalues = H.eigenvalues * 1e6  # # J/mm^2 -> J/m^2
    # omega = jnp.sqrt(eigenvalues / atom.mass) / (2 * jnp.pi)
    H = jax.hessian(objective)(trap_position.ravel())  # .ravel() ensures it's 1D
    eigenvalues = jnp.linalg.eigvalsh(H) * 1e6  # Use eigvalsh for symmetric matrix, more stable
    omega = jnp.sqrt(eigenvalues / atom.mass) / (2 * jnp.pi)

    # Non-interacting BEC radii
    angular_freq = 2 * jnp.pi * omega
    bec_radii = jnp.sqrt(ac.constants.hbar / (atom.mass * angular_freq))

    # Thomas-Fermi radii
    w_ho = jnp.prod(angular_freq) ** (1 / 3)  # geometric mean
    a_ho = jnp.sqrt(ac.constants.hbar / (atom.mass * w_ho))
    mu = 0.5 * ac.constants.hbar * w_ho * (15 * atom.a_s * n_atoms / a_ho) ** (2 / 5)
    tf_radii = jnp.sqrt(2 * mu / atom.mass) / angular_freq

    return U0, omega, eigenvalues, bec_radii, tf_radii, mu


# ----------------------------------------------------------------------------------------------------
# Distribute logical wire currents to physical wire segments.
# ----------------------------------------------------------------------------------------------------
@jax.jit
def distribute_currents_to_wires(I_wires: jnp.ndarray) -> jnp.ndarray:
    """
    Distribute logical wire currents to physical wire segments.
    Input shape: (15,) = 6 shifting + 9 guiding
    Output shape: (73,) = expanded full wire layout
    """
    I_shifting_wires = I_wires[:6]  # shape: (6,)
    I_guiding_wires = I_wires[6:]  # shape: (9,)

    # Expand shifting wires
    total_shifting = jnp.concatenate(
        [I_shifting_wires, -I_shifting_wires, I_shifting_wires, -I_shifting_wires, I_shifting_wires]
    )  # (30,)

    # Expand guiding wires
    total_guiding = jnp.repeat(I_guiding_wires, GUIDING_WIRE_SEGMENT_COUNTS)  # (43,)

    return jnp.concatenate([total_shifting, total_guiding])  # (73,)


# ----------------------------------------------------------------------------------------------------
# Save and load results to/from a JSON file.
# ----------------------------------------------------------------------------------------------------
def create_results_dir(base_dir="results"):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f") # YYYYMMDD_HHMMSS_microseconds
    run_id_suffix = uuid.uuid4().hex[:6] # Short unique ID

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
    fig, axs = plt.subplots(4, 3, figsize=(14, 12))
    description = ", ".join(
        f"{key}={value}" for key, value in params.items()
    )
    fig.suptitle(f"{description}", fontsize=10)

    trajectory, target_rs = results.trajectory, results.target_rs
    plot_x_over_time     (axs[0, 0], trajectory, target_rs, shifting_wire_x)
    plot_xy_trajectory   (axs[0, 1], trajectory, target_rs, shifting_wire_x)
    plot_xz_trajectory   (axs[0, 2], trajectory, target_rs, shifting_wire_x)

    plot_trap_potential  (axs[2, 1], results.U0s)
    plot_mu_values       (axs[3, 1], results.mu_vals)
    plot_trap_frequencies(axs[1, 2], results.omegas)
    plot_trap_radii      (axs[2, 2], results.BEC_radii, title="BEC Radii")
    plot_trap_radii      (axs[3, 2], results.TF_radii, title=f"TF Radii ({params.n_atoms} atoms)")

    current_log = results.current_log
    plot_wire_currents   (axs[1, 1], current_log, wire_indices=[0, 1, 2, 3, 4, 5])
    plot_wire_currents   (axs[1, 0], current_log, wire_indices=[6, 14])
    plot_wire_currents   (axs[2, 0], current_log, wire_indices=[7, 8, 12, 13])
    plot_wire_currents   (axs[3, 0], current_log, wire_indices=[9, 10, 11])

    plt.tight_layout()
    if results_dir:
        plt.savefig(os.path.join(results_dir, "optimization_analysis.png"), dpi=300)
    plt.show()
# fmt: on


def plot_x_over_time(ax: plt.Axes, trajectory: jnp.ndarray, trajectory_ref: jnp.ndarray, shifting_wire_x: jnp.ndarray):
    # Plot x over time.
    x = trajectory[:, 0]
    x_ref = trajectory_ref[:, 0]
    for wire_x in shifting_wire_x:
        ax.axhline(wire_x, linestyle="--", color="gray", linewidth=0.5)
    ax.plot(x_ref, label="Target x position", color="orange", marker=".", markersize=0.1)
    ax.plot(x, label="x position", markersize=0.1)
    ax.set_title("Trap x Position Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("x (mm)")
    ax.legend()


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
    ax.set_title("Trap Trajectory (x-y)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend()


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
    ax.set_title("Trap Trajectory (x-z)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.legend()


def plot_wire_currents(ax: plt.Axes, current_log: jnp.ndarray, wire_indices: List[int]):
    # Plot wire currents over time.
    for i in wire_indices:
        ax.plot(current_log[:, i], label=f"Wire {i}", markersize=0.1)
    ax.set_title("Wire Currents Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Current (A)")
    ax.legend(fontsize="small", ncol=2)


def plot_trap_potential(ax: plt.Axes, U0_vals: List[float]):
    # Plot U0 (trap potential energy) over time.
    ax.plot(U0_vals, label="U0", markersize=0.1)
    ax.set_title("Trap Potential Energy (U0) Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("U0 (J)")
    ax.legend()


def plot_trap_frequencies(ax: plt.Axes, omega_vals: jnp.ndarray):
    # Plot trap frequencies over time.
    for i, label in enumerate(["$\\omega_x$", "$\\omega_y$", "$\\omega_z$"]):
        ax.plot(omega_vals[:, i], label=label, markersize=0.1)
    ax.set_title("Trap Frequencies Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend()


def plot_trap_radii(ax: plt.Axes, radii_vals: jnp.ndarray, title: str):
    # Plot BEC radii over time.
    for i, label in enumerate(["$r_x$", "$r_y$", "$r_z$"]):
        ax.plot(radii_vals[:, i], label=f"{label} {i + 1}", markersize=0.1)
    ax.set_title(f"{title} Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Radius (m)")
    ax.legend()


def plot_mu_values(ax: plt.Axes, mu_vals: jnp.ndarray):
    # Plot chemical potential over time.
    ax.plot(mu_vals, label="Chemical Potential", markersize=0.1)
    ax.set_title("Chemical Potential Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Chemical Potential (J)")
    ax.legend()


# ----------------------------------------------------------------------------------------------------
# Main entry point for the script.
# ----------------------------------------------------------------------------------------------------
def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Transport Optimizer for Atom Chip")
    parser.add_argument("--results_dir",    type=str,   default=None,     help="Results directory")
    parser.add_argument("--T",              type=int,   default=50000,    help="Number of time steps")
    parser.add_argument("--num_shifts",     type=int,   default=6,        help="Number of shifts to apply to the trap")
    parser.add_argument("--reg",            type=float, default=1e-2,     help="Regularization parameter")
    parser.add_argument("--n_atoms",        type=int,   default=int(1e5), help="Number of atoms in the BEC")
    parser.add_argument("--I_max_shifting", type=float, default=1.0,      help="Max current for shifting wires (A)")
    parser.add_argument("--I_max_guiding",  type=float, default=14.0,     help="Max current for guiding wires (A)")
    parser.add_argument("--wire_ids",       type=str, nargs='*', default=[0, 1, 2, 3, 4, 5, 6, 14],
                        help="List of wire IDs to optimize (default: all shifting wires and outermost guiding wires)")
    parser.add_argument("--scheduler",      type=str, default="smoothstep",
                        choices=["smoothstep", "cosine"],
                        help="Scheduler function to normalize time")
    args = parser.parse_args()
    # fmt: on

    if args.results_dir is None:
        # remove the result_path argument
        del args.results_dir

        # Run the transport optimization
        optimize_transport(attrdict(**vars(args)))
    else:
        # Load results from a CSV file if a path is provided
        params, results = load_results(args.results_dir)
        plot_results(params, results)


if __name__ == "__main__":
    main()
