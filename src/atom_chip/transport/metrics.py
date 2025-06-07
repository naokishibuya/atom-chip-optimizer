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
    omega = jnp.sqrt(jnp.abs(eigenvalues) / atom.mass) / (2 * jnp.pi)

    return U0, omega


# @jax.jit
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

    def eval_step(wire_currents: jnp.ndarray, trap_position: jnp.ndarray):
        return evaluate_trap(atom, wire_config, bias_config, wire_currents, trap_position)

    wire_currents = jax.vmap(
        planner.calculate_wire_currents, in_axes=(0, 0)
    )(
        I_schedule[:, :6],  # shape (n_steps, 6) shifting wires
        I_schedule[:, 6:],  # shape (n_steps, 9) guiding wires
    )

    U0s, omegas = jax.vmap(eval_step)(wire_currents, trap_trajectory)
    return U0s, omegas
