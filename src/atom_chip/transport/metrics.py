from scipy.optimize import minimize
import jax
import jax.numpy as jnp
import atom_chip as ac
from atom_chip.transport import planner


@jax.jit
def evaluate_trap(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.field.BiasFieldConfig,
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
    eigenvalues = H.eigenvalues
    # TODO: Investigate why this can be negative (numerical issues?)
    omega = jnp.sqrt(jnp.abs(eigenvalues) / atom.mass) / (2 * jnp.pi)

    return U0, omega


@jax.jit
def simulate_trap_dynamics(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.field.BiasFieldConfig,
    trap_trajectory: jnp.ndarray,
    I_schedule: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simulate the trap dynamics over a trajectory given a schedule of wire currents.
    """

    wire_currents = planner.distribute_current_schedule_to_wires(I_schedule)

    def eval_step(I_t: jnp.ndarray, trap_position: jnp.ndarray):
        return evaluate_trap(atom, wire_config, bias_config, I_t, trap_position)

    U0s, omegas = jax.vmap(eval_step)(wire_currents, trap_trajectory)
    return U0s, omegas


def reconstruct_trajecotry(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.field.BiasFieldConfig,
    trap_trajectory: jnp.ndarray,
    I_schedule: jnp.ndarray,
) -> jnp.ndarray:
    """
    Reconstruct the real trap path from the optimized currents
    """
    wire_currents = planner.distribute_current_schedule_to_wires(I_schedule)

    r0s = []
    guess = trap_trajectory[0]  # use desired initial position as the starting guess
    dx = trap_trajectory[1] - trap_trajectory[0]  # step size for warm-starting

    for I_t in wire_currents:
        r0 = find_trap_minimum(
            atom,
            wire_config,
            I_t,
            bias_config,
            guess,
        )
        r0s.append(r0)
        guess = r0 + dx  # warm-start the next step with current solution

    r0s = jnp.stack(r0s)

    U0s, omegas = simulate_trap_dynamics(
        atom=ac.rb87,
        wire_config=wire_config,
        bias_config=bias_config,
        trap_trajectory=r0s,
        I_schedule=I_schedule,
    )
    return r0s, U0s, omegas


# This is not a JIT function because it uses `minimize` from SciPy
def find_trap_minimum(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    wire_currents: jnp.ndarray,
    bias_config: ac.field.BiasFieldConfig,
    guess: jnp.ndarray,
) -> jnp.ndarray:
    """
    Find the local minimum of the magnetic potential energy.
    """

    def potential_energy_fn(r):
        """Compute potential energy at position r."""
        U, _, _ = ac.atom_chip.trap_potential_energies(jnp.atleast_2d(r), atom, wire_config, wire_currents, bias_config)
        return U[0]

    # Bound the search to a small region around the guess
    bounds = jnp.array(
        [
            [guess[0] - 0.5, guess[0] + 0.5],
            [guess[1] - 0.5, guess[1] + 0.5],
            [max(0, guess[2] - 0.5), guess[2] + 0.5],
        ]
    )

    # scipy minimize
    result = minimize(
        potential_energy_fn,
        x0=guess,
        method="Nelder-Mead",
        bounds=bounds,
        options=dict(
            xatol=1e-9,
            fatol=1e-9,
            maxiter=1000,
            maxfev=3000,
            disp=False,
        ),
    )
    if not result.success:
        print(f"Minimization failed: {result.message} result.x={result.x} guess={guess}")

    return result.x  # shape (3,)
