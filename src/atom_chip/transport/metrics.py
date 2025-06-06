from functools import partial
import jax
import jax.numpy as jnp
import atom_chip as ac
from .scheduler import ScheduleFn


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
    omega = jnp.sqrt(jnp.abs(eigenvalues) / atom.mass) / (2 * jnp.pi)

    return U0, omega


@partial(jax.jit, static_argnames=("schedule_fn",))
def simulate_trap_dynamics(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.field.BiasFieldConfig,
    trap_trajectory: jnp.ndarray,
    anchor_currents: jnp.ndarray,
    schedule_fn: ScheduleFn,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generates a full schedule of trap parameters (I_schedule, r0, U0, omega)
    by applying the schedule_fn to anchor_currents and evaluating the trap at each step.
    """
    # Generate the current schedule
    I_schedule = schedule_fn(trap_trajectory, anchor_currents)

    def eval_step(wire_currents: jnp.ndarray, trap_position: jnp.ndarray):
        return evaluate_trap(atom, wire_config, bias_config, wire_currents, trap_position)

    U0s, omegas = jax.vmap(eval_step)(I_schedule, trap_trajectory)

    return I_schedule, U0s, omegas
