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
# Generate a linear trajectory for the trap center position.
#
# This can be JIT-compiled but it's not necessary since it runs only once.
# ----------------------------------------------------------------------------------------------------
def generate_reference_trajectory(
    r0: jnp.ndarray,
    num_shifts: int,
    shifting_wire_distance: float,  # mm
    steps_per_wire_distance: int,
) -> jnp.ndarray:
    """
    Generates a linear trajectory for the trap center position, and start and end currents for the wires.
    """
    step_size = shifting_wire_distance / steps_per_wire_distance
    num_steps = num_shifts * steps_per_wire_distance

    steps = jnp.arange(0, num_steps + 1).reshape(-1, 1)
    shift = steps * step_size
    trajectory = r0 + shift * jnp.array([1.0, 0.0, 0.0])

    return trajectory


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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate potential energy and curvature at a fixed position."""

    def objective(r):
        """Compute potential energy at position r."""
        U, _, _ = ac.atom_chip.trap_potential_energies(jnp.atleast_2d(r), atom, wire_config, wire_currents, bias_config)
        return U[0]

    U0 = objective(trap_position)
    H = ac.potential.hessian_by_finite_difference(
        objective,
        jnp.ravel(trap_position),
        step=1e-3,
    )
    eigenvalues = H.eigenvalues * 1e6  # # J/mm^2 -> J/m^2
    omega = jnp.sqrt(eigenvalues / atom.mass) / (2 * jnp.pi)
    return U0, omega


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
# Optimize the transport of the atom chip trap.
# ----------------------------------------------------------------------------------------------------
# fmt: off
def optimize_transport(
    atom           : ac.Atom,
    wire_config    : ac.atom_chip.WireConfig,
    bias_config    : ac.field.BiasConfig,
    I_start        : jnp.ndarray,  # Initial currents for the wires (shape: (15,))
    I_end          : jnp.ndarray,  # Final currents for the wires (shape: (15,))
    mask           : jnp.ndarray,  # Mask to disable guiding wires during optimization
    trajectory_ref : jnp.ndarray,  # Reference trajectory for the trap (shape: (T, 3))
    r0_ref         : jnp.ndarray,  # Reference position of the trap at t=0 (shape: (3,))
    U0_ref         : float,        # Reference potential energy of the trap
    omega_ref      : jnp.ndarray,  # Reference trap frequencies (shape: (3,))
    T              : int,          # Number of time steps
    α              : float,        # Schedule-following weight
    λ              : float,        # Regularization parameter
):
# fmt: on
    # Cosine schedule (range [0, 1] over finite time steps t over duration T)
    def cosine_schedule(t: int, T: int) -> jnp.ndarray:
        return 0.5 * (1 - jnp.cos(jnp.pi * t / T))

    def r_target(t: int, T: int) -> jnp.ndarray:
        r_start = trajectory_ref[0]
        r_end   = trajectory_ref[-1]
        return r_start + cosine_schedule(t, T) * (r_end - r_start)

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
    current_log = [ I_wires   ]
    U0_vals     = [ U0_ref    ]  # to log trap potential energy U0
    omega_vals  = [ omega_ref ]  # to log trap frequencies
    # fmt: on

    # Control loop
    for t in range(T):
        # Compute the current target position
        r_now = α * r_target(t, T) + (1 - α) * trajectory[-1]  # Average with last position
        r_next = r_target(t + 1, T)
        delta_r = r_next - r_now

        # Compute the implicit gradient and update currents
        J = compute_dr_dI(r_now, I_wires)
        omega = omega_vals[-1]  # Use the last computed frequency
        adjusted_λ = λ * (1 + jnp.max(omega / omega_ref))  # if trap is tight, increase regularization
        delta_I = jnp.linalg.solve(J.T @ J + adjusted_λ * jnp.eye(J.shape[1]), J.T @ delta_r)
        I_wires = I_wires + delta_I * mask  # Apply mask to restrict current updates

        # Find the minimum trap position and evaluate the trap
        wire_currents = distribute_currents_to_wires(I_wires)
        r_min = find_trap_minimum(atom, wire_config, bias_config, wire_currents, r_now)
        U0, omega = evaluate_trap(atom, wire_config, bias_config, wire_currents, r_min)

        # Log the results
        trajectory.append(r_min)
        current_log.append(I_wires)
        U0_vals.append(U0)
        omega_vals.append(omega)

        print(f"Step {t + 1}: r_min = {r_min}, U0 = {U0:.4g}, omega = {omega}")
    return trajectory, current_log, U0_vals, jnp.stack(omega_vals)


def main():
    # Set up reference variables and configurations
    atom_chip = transport_initializer.build_atom_chip()
    analysis = transport_initializer.analyze_atom_chip(atom_chip)
    r0_ref = analysis.potential.minimum.position
    U0_ref = analysis.potential.minimum.value
    omega_ref = analysis.potential.trap.frequency

    num_shifts = 6
    trajectory_ref = generate_reference_trajectory(
        r0=r0_ref,
        num_shifts=num_shifts,
        shifting_wire_distance=0.4,  # mm
        steps_per_wire_distance=20,
    )

    print(f"Initial trap pos: {r0_ref}, U0: {U0_ref:.4g}, omega: {omega_ref}")
    print(f"Desired final position: {trajectory_ref[-1]}")

    # Get the wire layout and initial conditions
    wire_config = setup_wire_config()
    bias_config = transport_initializer.setup_bias_config()

    I_shifting_wires = jnp.array(transport_initializer.SHIFTING_WIRE_CURRENTS, dtype=jnp.float64)
    I_guiding_wires = jnp.array(transport_initializer.GUIDING_WIRE_CURRENTS, dtype=jnp.float64)
    I_start = jnp.concatenate([I_shifting_wires, I_guiding_wires])
    I_end = jnp.concatenate([jnp.roll(I_shifting_wires, num_shifts), I_guiding_wires])

    # Mask to disable guiding wires during the optimization
    masked_wire_ids = jnp.arange(6, 15, dtype=jnp.int32)  # Guiding wires
    mask = jnp.ones_like(I_start).at[masked_wire_ids].set(0.0)  # Disable guiding wires

    T = 20
    α = 0.65  # schedule-following weight
    λ = 1e-4  # λ ∈ [1e-4, 1e-1]

    trajectory, current_log, U0_vals, omega_vals = optimize_transport(
        atom=atom_chip.atom,
        wire_config=wire_config,
        bias_config=bias_config,
        I_start=I_start,
        I_end=I_end,
        mask=mask,
        trajectory_ref=trajectory_ref,
        r0_ref=r0_ref,
        U0_ref=U0_ref,
        omega_ref=omega_ref,
        T=T,
        α=α,
        λ=λ,
    )
    # Convert lists to arrays for plotting
    plot_results(trajectory, current_log, U0_vals, omega_vals)


# fmt: off
def plot_results(
    trajectory: List[jnp.ndarray],
    current_log: List[jnp.ndarray],
    U0_vals: List[float],
    omega_vals: List[jnp.ndarray],
):
# fmt: on

# fmt: off
    trajectory  = jnp.stack(trajectory)
    current_log = jnp.stack(current_log)
    U0_vals     = jnp.array(U0_vals)
    omega_vals  = jnp.stack(omega_vals)  # shape (N, 3)
# fmt: on

    # Plotting the trap trajectory, currents, U0, and omega.
    fig, axs = plt.subplots(3, 3, figsize=(14, 10))

    # Plot x-y trap trajectory.
    axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1], marker="o", label="x-y position")
    axs[0, 0].set_title("Trap Trajectory (x-y)")
    axs[0, 0].set_xlabel("x (mm)")
    axs[0, 0].set_ylabel("y (mm)")
    axs[0, 0].legend()
    axs[0, 0].axis("equal")

    # Plot x-z trap trajectory.
    axs[1, 0].plot(trajectory[:, 0], trajectory[:, 2], marker="o", label="x-z position")
    axs[1, 0].set_title("Trap Trajectory (x-z)")
    axs[1, 0].set_xlabel("x (mm)")
    axs[1, 0].set_ylabel("z (mm)")
    axs[1, 0].legend()

    # Plot x over time.
    axs[2, 0].plot(trajectory[:, 0], marker="o", label="x position")
    axs[2, 0].set_title("Trap x Position Over Time")
    axs[2, 0].set_xlabel("Step")
    axs[2, 0].set_ylabel("x (mm)")
    axs[2, 0].set_xticks(range(len(trajectory)))
    axs[2, 0].legend()

    # Plot wire currents over time.
    for i in range(6):
        axs[0, 1].plot(current_log[:, i], label=f"Wire {i}")
    axs[0, 1].set_title("Wire Currents Over Time")
    axs[0, 1].set_xlabel("Step")
    axs[0, 1].set_ylabel("Current (A)")
    axs[0, 1].set_xticks(range(len(current_log)))
    axs[0, 1].legend(fontsize="small", ncol=2)

    for i in [6, 14]:
        axs[1, 1].plot(current_log[:, i], label=f"Wire {i}")
    axs[1, 1].set_title("Wire Currents Over Time")
    axs[1, 1].set_xlabel("Step")
    axs[1, 1].set_ylabel("Current (A)")
    axs[1, 1].set_xticks(range(len(current_log)))
    axs[1, 1].legend(fontsize="small", ncol=2)

    for i in [7, 8, 12, 13]:
        axs[2, 1].plot(current_log[:, i], label=f"Wire {i}")
    axs[2, 1].set_title("Wire Currents Over Time")
    axs[2, 1].set_xlabel("Step")
    axs[2, 1].set_ylabel("Current (A)")
    axs[2, 1].set_xticks(range(len(current_log)))
    axs[2, 1].legend(fontsize="small", ncol=2)

    for i in [9, 10, 11]:
        axs[0, 2].plot(current_log[:, i], label=f"Wire {i}")
    axs[0, 2].set_title("Wire Currents Over Time")
    axs[0, 2].set_xlabel("Step")
    axs[0, 2].set_ylabel("Current (A)")
    axs[0, 2].set_xticks(range(len(current_log)))
    axs[0, 2].legend(fontsize="small", ncol=2)

    # Plot U0 (trap potential energy) over time.
    axs[1, 2].plot(U0_vals, marker="o", label="U0")
    axs[1, 2].set_title("Trap Potential Energy (U0) Over Time")
    axs[1, 2].set_xlabel("Step")
    axs[1, 2].set_ylabel("U0 (J)")
    axs[1, 2].set_xticks(range(len(U0_vals)))
    axs[1, 2].legend()

    # Plot trap frequencies over time.
    for i, label in enumerate(["$\\omega_x$", "$\\omega_y$", "$\\omega_z$"]):
        axs[2, 2].plot(omega_vals[:, i], marker="o", label=label)
    axs[2, 2].set_title("Trap Frequencies Over Time")
    axs[2, 2].set_xlabel("Step")
    axs[2, 2].set_ylabel("Frequency (arb. units)")
    axs[2, 2].set_xticks(range(len(omega_vals)))
    axs[2, 2].legend()

    plt.tight_layout()
    plt.savefig("transport.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
