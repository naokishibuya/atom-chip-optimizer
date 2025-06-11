from typing import List, Tuple
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    return U0, omega, eigenvalues


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
# Main function to run the transport optimization.
# ----------------------------------------------------------------------------------------------------
def main():
    # 1. Simulation parameters
    T = 1000
    num_shifts = 6
    λ = 1e-2  # λ ∈ [1e-4, 1e-1]

    I_max_shifting = 1.0  # A, max current for shifting wires
    I_max_guiding = 14.0  # A, max current for guiding wires

    # Mask to disable guiding wires during the optimization
    # wire_ids = jnp.arange(0, 15, dtype=jnp.int32)  # All wires
    wire_ids = jnp.arange(0, 6, dtype=jnp.int32)  # Only shifting wires
    # wire_ids = jnp.arange(0, 5, dtype=jnp.int32)  # shifting wires except zero current wire
    # wire_ids = jnp.concatenate([jnp.arange(0, 6), jnp.array([6, 14])], dtype=jnp.int32)
    # wire_ids = jnp.concatenate([jnp.arange(0, 6), jnp.arange(7, 14)], dtype=jnp.int32)
    # wire_ids = jnp.array([5, 6, 14], dtype=jnp.int32) # Only those with 0 current in the start
    mask = jnp.zeros((15,)).at[wire_ids].set(1.0)

    setting_info = f"T={T}, num_shifts={num_shifts}, λ={λ}"
    print(f"Transport optimization settings: {setting_info}, mask={mask}")

    # 2. Transport initial setup
    wire_config = setup_wire_config()
    bias_config = transport_initializer.setup_bias_config()

    I_shifting_wires = jnp.array(transport_initializer.SHIFTING_WIRE_CURRENTS, dtype=jnp.float64)
    I_guiding_wires = jnp.array(transport_initializer.GUIDING_WIRE_CURRENTS, dtype=jnp.float64)
    I_start = jnp.concatenate([I_shifting_wires, I_guiding_wires])
    I_limits = jnp.concatenate(
        [
            jnp.ones_like(I_shifting_wires) * I_max_shifting,  # Shifting wires limits
            jnp.ones_like(I_guiding_wires) * I_max_guiding,  # Guiding wires limits
        ]
    )  # shape: (15,)

    # 3. Reference values for the trap
    atom_chip = transport_initializer.build_atom_chip()
    analysis = transport_initializer.analyze_atom_chip(atom_chip)
    r0_ref = analysis.potential.minimum.position
    U0_ref = analysis.potential.minimum.value
    omega_ref = analysis.potential.trap.frequency
    radii_ref = analysis.potential.bec.radii  # non-interacting BEC radii
    destination_r = calculate_destination(r0_ref, num_shifts, shifting_wire_distance=0.4)  # mm

    print(f"Initial trap pos: {r0_ref}, U0: {U0_ref:.4g}, omega: {omega_ref} (Hz) radii: {radii_ref} (m)")
    print(f"Desired final position: {destination_r}")

    # 4. Optimize the transport
    trajectory, target_rs, current_log, U0_vals, omega_vals, radii_vals = optimize_transport(
        atom=atom_chip.atom,
        wire_config=wire_config,
        bias_config=bias_config,
        I_start=I_start,
        I_limits=I_limits,
        mask=mask,
        r0_ref=r0_ref,
        U0_ref=U0_ref,
        omega_ref=omega_ref,
        radii_ref=radii_ref,
        destination_r=destination_r,
        T=T,
        λ=λ,
    )

    # 5. Plot the results
    plot_results(setting_info, trajectory, target_rs, current_log, U0_vals, omega_vals, radii_vals)


# ----------------------------------------------------------------------------------------------------
# Optimize the transport of the atom chip trap.
# ----------------------------------------------------------------------------------------------------
# fmt: off
def optimize_transport(
    atom           : ac.Atom,
    wire_config    : ac.atom_chip.WireConfig,
    bias_config    : ac.field.BiasConfig,
    I_start        : jnp.ndarray,  # Initial currents for the wires (shape: (15,))
    I_limits       : jnp.ndarray,  # Optional limits for the currents (shape: (15,))
    mask           : jnp.ndarray,  # Mask to restrict current updates (shape: (15,))
    r0_ref         : jnp.ndarray,  # Reference position of the trap at t=0 (shape: (3,))
    U0_ref         : float,        # Reference potential energy of the trap
    omega_ref      : jnp.ndarray,  # Reference trap frequencies (shape: (3,))
    radii_ref      : jnp.ndarray,  # Reference BEC radii (shape: (3,))
    destination_r  : jnp.ndarray,  # Desired final position of the trap (shape: (3,))
    T              : int,          # Number of time steps
    λ              : float,        # Regularization parameter
):
# fmt: on
    # Cosine schedule (range [0, 1] over finite time steps t over duration T)
    def cosine_schedule(t: int, T: int) -> jnp.ndarray:
        return 0.5 * (1 - jnp.cos(jnp.pi * t / T))

    def r_target(t: int, T: int) -> jnp.ndarray:
        return r0_ref + cosine_schedule(t, T) * (destination_r - r0_ref)

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
    def compute_dr_dI(r0, I_wires):
        H = hess_U_r(r0, I_wires)
        J = cross_jac(r0, I_wires)
        return -jnp.linalg.solve(H, J)

    # fmt: off
    I_wires     = I_start
    trajectory  = [ r0_ref    ]
    target_rs   = [ r0_ref    ]
    current_log = [ I_wires   ]
    U0_vals     = [ U0_ref    ]
    omega_vals  = [ omega_ref ]
    radii_vals  = [ radii_ref ]
    error_log   = []
    # fmt: on

    # Initialize the trap radii based on the reference values
    def target_following_ratio(radii, k=10.0):
        scale = jnp.linalg.norm(radii / radii_ref - 1.0)
        return 1.0 - 1.0 / (1.0 + jnp.exp(k * (scale - 0.1)))  # ε ~ 0.1 or 0.2

    # Control loop
    for t in range(T):
        # Compute the current target position
        follow_ratio = target_following_ratio(radii_vals[-1])
        r_now = follow_ratio * r_target(t, T) + (1.0 - follow_ratio) * trajectory[-1]  # Average with last position
        r_next = r_target(t + 1, T)
        delta_r = r_next - r_now

        # Compute the implicit gradient and update currents
        J = compute_dr_dI(r_now, I_wires)
        cond_J = jnp.linalg.cond(J)
        adjusted_λ = λ * (1 + jnp.linalg.norm(cond_J))
        delta_I = jnp.linalg.solve(J.T @ J + adjusted_λ * jnp.eye(J.shape[1]), J.T @ delta_r)
        I_wires = I_wires + delta_I * mask  # Apply mask to restrict current updates
        I_wires = jnp.clip(I_wires, -I_limits, I_limits)

        # Find the minimum trap position and evaluate the trap
        wire_currents = distribute_currents_to_wires(I_wires)
        r_min = find_trap_minimum(atom, wire_config, bias_config, wire_currents, r_now)
        U0, omega, eigenvalues = evaluate_trap(atom, wire_config, bias_config, wire_currents, r_min)
        if jnp.any(jnp.isnan(omega)) or jnp.any(eigenvalues < 0):
            # Handle NaN or negative eigenvalues in omega
            message = "Encountered NaN or negative eigenvalues in trap evaluation."
            error_log.append((t, r_min, I_wires, U0, omega, message))
            continue

        radii = jnp.sqrt(ac.constants.hbar / (atom.mass *  2 * jnp.pi * omega))  # Calculate BEC radii from frequencies

        # Log the results
        trajectory.append(r_min)
        target_rs.append(r_next)
        current_log.append(I_wires)
        U0_vals.append(U0)
        omega_vals.append(omega)
        radii_vals.append(radii)

        print(f"Step {t + 1:4d}: r_min={r_min} U0={U0:.4g} omega={omega} radii={radii}")

    if error_log:
        print(f"Errors encountered during optimization: {len(error_log)} steps with NaN or negative eigenvalues.")
        for step, r_min, I_wires, U0, omega, message in error_log:
            print(f"Step {step}: r_min={r_min}, I_wires={I_wires}, U0={U0}, omega={omega}: {message}")
    else:
        print("Optimization completed successfully without errors.")

    return trajectory, target_rs, current_log, U0_vals, omega_vals, radii_vals


# fmt: off
def plot_results(
    setting_info: str,
    trajectory: List[jnp.ndarray],
    target_rs: List[jnp.ndarray],
    current_log: List[jnp.ndarray],
    U0_vals: List[float],
    omega_vals: List[jnp.ndarray],
    radii_vals: List[jnp.ndarray],
):
    trajectory  = jnp.stack(trajectory)
    target_rs   = jnp.stack(target_rs)
    current_log = jnp.stack(current_log)
    U0_vals     = jnp.array(U0_vals)
    omega_vals  = jnp.stack(omega_vals)
    radii_vals  = jnp.stack(radii_vals)

    # Plotting the trap trajectory, currents, U0, and omega.
    fig, axs = plt.subplots(4, 3, figsize=(14, 12))
    fig.suptitle(f"Optimization results: {setting_info}", fontsize=12)

    plot_x_over_time     (axs[0, 0], trajectory, target_rs)
    plot_xy_trajectory   (axs[0, 1], trajectory, target_rs)
    plot_xz_trajectory   (axs[0, 2], trajectory, target_rs)
    plot_wire_currents   (axs[1, 0], current_log, wire_indices=[0, 1, 2, 3, 4, 5])
    plot_trap_potential  (axs[1, 1], U0_vals)
    plot_trap_frequencies(axs[1, 2], omega_vals)
    plot_wire_currents   (axs[2, 0], current_log, wire_indices=[6, 14])
    plot_wire_currents   (axs[2, 1], current_log, wire_indices=[7, 9, 12, 13])
    plot_wire_currents   (axs[2, 2], current_log, wire_indices=[9, 10, 11])
    plot_trap_radii      (axs[3, 0], radii_vals[:, 0], label="$r_x$")
    plot_trap_radii      (axs[3, 1], radii_vals[:, 1], label="$r_y$")
    plot_trap_radii      (axs[3, 2], radii_vals[:, 2], label="$r_z$")

    plt.tight_layout()
    plt.savefig("transport.png", dpi=300)
    plt.show()
# fmt: on


def plot_x_over_time(ax: plt.Axes, trajectory: jnp.ndarray, trajectory_ref: jnp.ndarray):
    # Plot x over time.
    x = trajectory[:, 0]
    x_ref = trajectory_ref[:, 0]
    ax.plot(x_ref, linestyle="-", label="Target x position", color="orange")
    ax.plot(x, marker="o", label="x position", markersize=0.5)
    ax.set_title("Trap x Position Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("x (mm)")
    ax.legend()


def plot_xy_trajectory(ax: plt.Axes, trajectory: jnp.ndarray, trajectory_ref: jnp.ndarray):
    # Plot x-y trap trajectory.
    x, y = trajectory[:, 0], trajectory[:, 1]
    x_ref, y_ref = trajectory_ref[:, 0], trajectory_ref[:, 1]
    ax.plot(x_ref, y_ref, linestyle="-", label="Target x-y position", color="orange")
    ax.plot(x, y, marker="o", label="x-y position", markersize=0.5)
    ax.set_title("Trap Trajectory (x-y)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend()
    ax.axis("equal")


def plot_xz_trajectory(ax: plt.Axes, trajectory: jnp.ndarray, trajectory_ref: jnp.ndarray):
    # Plot x-z trap trajectory.
    x, z = trajectory[:, 0], trajectory[:, 2]
    x_ref, z_ref = trajectory_ref[:, 0], trajectory_ref[:, 2]
    ax.plot(x_ref, z_ref, linestyle="-", label="Target x-z position", color="orange")
    ax.plot(x, z, marker="o", label="x-z position", markersize=0.5)
    ax.set_title("Trap Trajectory (x-z)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.legend()


def plot_wire_currents(ax: plt.Axes, current_log: jnp.ndarray, wire_indices: List[int]):
    # Plot wire currents over time.
    for i in wire_indices:
        ax.plot(current_log[:, i], label=f"Wire {i}")
    ax.set_title("Wire Currents Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Current (A)")
    ax.legend(fontsize="small", ncol=2)


def plot_trap_potential(ax: plt.Axes, U0_vals: List[float]):
    # Plot U0 (trap potential energy) over time.
    ax.plot(U0_vals, marker="o", label="U0", markersize=0.5)
    ax.set_title("Trap Potential Energy (U0) Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("U0 (J)")
    ax.legend()


def plot_trap_frequencies(ax: plt.Axes, omega_vals: jnp.ndarray):
    # Plot trap frequencies over time.
    for i, label in enumerate(["$\\omega_x$", "$\\omega_y$", "$\\omega_z$"]):
        ax.plot(omega_vals[:, i], marker="o", label=label, markersize=0.5)
    ax.set_title("Trap Frequencies Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend()


def plot_trap_radii(ax: plt.Axes, radii_vals: jnp.ndarray, label: str):
    # Plot BEC radii over time.
    ax.plot(radii_vals, marker="o", label=label, markersize=0.5)
    ax.set_title("BEC Radii Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Radius (m)")
    ax.set_ylim(0.0, 2.0e-6)  # Adjust y-limits for better visibility
    ax.legend()


if __name__ == "__main__":
    main()
