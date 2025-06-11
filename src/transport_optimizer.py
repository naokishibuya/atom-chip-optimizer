import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import atom_chip as ac
import transport_initializer


def main():
    # Get the wire layout and initial conditions
    bias_config = transport_initializer.make_bias_config()
    shifting_wires, guiding_wires = transport_initializer.setup_wire_layout()
    wire_config = transport_initializer.setup_wire_config(
        shifting_wires=shifting_wires,
        guiding_wires=guiding_wires,
    )

    I_shifting_wires = jnp.array(transport_initializer.SHIFTING_WIRE_CURRENTS, dtype=jnp.float64)
    I_guiding_wires = jnp.array(transport_initializer.GUIDING_WIRE_CURRENTS, dtype=jnp.float64)
    I_start = jnp.concatenate([I_shifting_wires, I_guiding_wires])

    currents = transport_initializer.distribute_currents_to_wires(I_start)
    options = ac.potential.AnalysisOptions(
        search=make_search_options(initial_guess=[0.0, 0.0, 0.5]),
        hessian=dict(
            # method = "jax",
            method="finite-difference",
            hessian_step=1e-5,  # Step size for Hessian calculation
        ),
        # for the trap analayis (not used for field analysis)
        total_atoms=1e5,
        condensed_atoms=1e5,
    )
    trap_analysis = ac.atom_chip.analyze_trap(ac.rb87, wire_config, currents, bias_config, options)

    # Initial reference values for the trap
    r0_ref = trap_analysis.minimum.position
    U0_ref = trap_analysis.minimum.value
    omega_ref = trap_analysis.trap.frequency
    num_shifts = 6
    trap_trajectory = generate_trajectory(
        r0=r0_ref,
        num_shifts=num_shifts,
        shifting_wire_distance=0.4,  # mm
        steps_per_wire_distance=20,
    )
    I_end = jnp.concatenate([jnp.roll(I_shifting_wires, num_shifts), I_guiding_wires])

    print(f"Initial trap pos: {r0_ref}, U0: {U0_ref:.4g}, omega: {omega_ref}")
    print(f"Desired final position: {trap_trajectory[-1]}")

    fit(
        trap_trajectory=trap_trajectory,
        I_start=I_start,
        I_end=I_end,
        wire_config=wire_config,
        bias_config=bias_config,
        r0_ref=r0_ref,
        U0_ref=U0_ref,
        omega_ref=omega_ref,
    )


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


def generate_trajectory(
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


def fit(
    trap_trajectory: jnp.ndarray,
    I_start: jnp.ndarray,
    I_end: jnp.ndarray,
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.atom_chip.BiasConfig,
    r0_ref: jnp.ndarray,
    U0_ref: float,
    omega_ref: jnp.ndarray,
):
    def U(r, I_wires):
        wire_currents = transport_initializer.distribute_currents_to_wires(I_wires)
        U, _, _ = ac.atom_chip.trap_potential_energies(
            jnp.atleast_2d(r), ac.rb87, wire_config, wire_currents, bias_config
        )
        return U[0]

    def s(t, T):
        # Cosine schedule
        return 0.5 * (1 - jnp.cos(jnp.pi * t / T))

    def r_target(t, T, r_start, r_end):
        return r_start + s(t, T) * (r_end - r_start)

    # Trap gradient
    grad_U_r = jax.grad(U, argnums=0)
    hess_U_r = jax.jacfwd(grad_U_r, argnums=0)
    cross_jac = jax.jacfwd(grad_U_r, argnums=1)

    # Implicit gradient dr/dI = -H^{-1} @ dgrad/dI
    def compute_dr_dI(r0, I_wires):
        H = hess_U_r(r0, I_wires)
        J = cross_jac(r0, I_wires)
        return -jnp.linalg.solve(H, J)

    # Control loop
    I_wires = I_start

    r_start = trap_trajectory[0]
    r_end = trap_trajectory[-1]

    trajectory = [r0_ref]
    current_log = [I_wires]
    U0_vals = [U0_ref]  # to log trap potential energy U0
    omega_vals = [omega_ref]  # to log trap frequencies

    T = 20
    λ = 1e-4  # λ ∈ [1e-4, 1e-1]
    base_lambda = λ  # Base regularization parameter

    mask = jnp.ones_like(I_start).at[jnp.arange(6, 15, dtype=jnp.int32)].set(0.0)  # Disable guiding wires

    for t in range(T):
        r_now = 0.6 * r_target(t, T, r_start, r_end) + 0.4 * trajectory[-1]  # Average with last position
        r_next = r_target(t + 1, T, r_start, r_end)
        delta_r = r_next - r_now

        J = compute_dr_dI(r_now, I_wires)

        omega = omega_vals[-1]  # Use the last computed frequency
        λ = base_lambda * (1 + jnp.max(omega / omega_ref))  # if trap is tight, increase regularization

        delta_I = jnp.linalg.solve(J.T @ J + λ * jnp.eye(J.shape[1]), J.T @ delta_r)

        I_wires = I_wires + delta_I * mask  # Apply mask to restrict current updates

        wire_currents = transport_initializer.distribute_currents_to_wires(I_wires)
        r_min = find_trap_minimum(
            atom=ac.rb87,
            wire_config=wire_config,
            bias_config=bias_config,
            wire_currents=wire_currents,
            guess=r_next,
        )
        trajectory.append(r_min)
        current_log.append(I_wires)

        U0, omega = evaluate_trap(
            atom=ac.rb87,
            wire_config=wire_config,
            bias_config=bias_config,
            wire_currents=wire_currents,
            trap_position=r_min,
        )
        U0_vals.append(U0)
        omega_vals.append(omega)
        print(f"Step {t + 1}: r_min = {r_min}, U0 = {U0:.4g}, omega = {omega}")

    # Convert lists to arrays for plotting
    trajectory = jnp.stack(trajectory)
    current_log = jnp.stack(current_log)
    U0_vals = jnp.array(U0_vals)
    omega_vals = jnp.stack(omega_vals)  # shape (N, 3)
    plot_results(trajectory=trajectory, current_log=current_log, U0_vals=U0_vals, omega_vals=omega_vals)


# This is not a JIT function because it uses `minimize` from SciPy
def find_trap_minimum(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    wire_currents: jnp.ndarray,
    bias_config: ac.field.BiasConfig,
    guess: jnp.ndarray,
) -> jnp.ndarray:
    """
    Find the local minimum of the magnetic potential energy.
    """

    def potential_energy_fn(r):
        """Compute potential energy at position r."""
        U, _, _ = ac.atom_chip.trap_potential_energies(jnp.atleast_2d(r), atom, wire_config, wire_currents, bias_config)
        return U[0]

    # scipy minimize
    result = minimize(potential_energy_fn, **make_search_options(initial_guess=guess))
    if not result.success:
        print(f"Minimization failed: {result.message} result.x={result.x} guess={guess}")

    return result.x  # shape (3,)


@jax.jit
def evaluate_trap(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.field.BiasConfig,
    wire_currents: jnp.ndarray,
    trap_position: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate potential energy and curvature at a fixed position."""

    def potential_energy_fn(r):
        """Compute potential energy at position r."""
        U, _, _ = ac.atom_chip.trap_potential_energies(jnp.atleast_2d(r), atom, wire_config, wire_currents, bias_config)
        return U[0]

    U0 = potential_energy_fn(trap_position)

    H = ac.potential.hessian_by_finite_difference(
        potential_energy_fn,
        jnp.ravel(trap_position),
        step=1e-3,
    )
    eigenvalues = H.eigenvalues * 1e6  # # J/mm^2 -> J/m^2
    omega = jnp.sqrt(eigenvalues / atom.mass) / (2 * jnp.pi)

    return U0, omega


def plot_results(
    trajectory: jnp.ndarray,
    current_log: jnp.ndarray,
    U0_vals: jnp.ndarray,
    omega_vals: jnp.ndarray,
):
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
