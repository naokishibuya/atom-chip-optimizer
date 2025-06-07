import jax
import jax.numpy as jnp
import atom_chip as ac

# from .scheduler import ScheduleFn
from atom_chip.transport import metrics


# ---------- Energy/Spatial Stability Losses ----------


@jax.jit
def energy_minimum_loss(U0s: jnp.ndarray, U0_ref: jnp.ndarray):
    """Penalizes trap depth below a minimum threshold."""
    return jnp.sum(jnp.square((U0s / U0_ref) - 1.0))


@jax.jit
def energy_smoothness_loss(U0s):
    return jnp.mean(jnp.square(U0s[1:] - U0s[:-1]))


@jax.jit
def trap_frequency_loss(omegas: jnp.ndarray, omega_ref: jnp.ndarray):
    """Penalizes deviation of trap frequency from reference."""
    return jnp.sum(jnp.square((omegas / omega_ref) - 1.0))


# ---------- Time-Dependent (Adiabatic) Losses ----------


@jax.jit
def frequency_change_loss(omegas: jnp.ndarray) -> jnp.ndarray:
    """Penalizes rapid changes in trap frequency."""
    return jnp.mean(jnp.square(omegas[1:] - omegas[:-1]))


@jax.jit
def jerk_loss(I_schedule: jnp.ndarray) -> jnp.ndarray:
    """Penalizes rapid changes in current."""
    return jnp.mean(jnp.square(I_schedule[2:] - 2 * I_schedule[1:-1] + I_schedule[:-2]))


# Total loss function that combines all components
@jax.jit
def total_loss_fn(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.field.BiasFieldConfig,
    trap_trajectory: jnp.ndarray,
    I_schedule: jnp.ndarray,
    omega_ref: jnp.ndarray,
    U0_ref: jnp.ndarray,
    loss_weights: dict,
) -> jnp.ndarray:
    """Computes the weighted sum of all physics-informed loss components."""

    U0s, omegas = metrics.simulate_trap_dynamics(
        atom,
        wire_config,
        bias_config,
        trap_trajectory,
        I_schedule,
    )

    # Compute losses
    # fmt: off
    losses = dict(
        U     = energy_minimum_loss(U0s, U0_ref),
        dU    = energy_smoothness_loss(U0s),
        freq  = trap_frequency_loss(omegas, omega_ref),
        dfreq = frequency_change_loss(omegas),
        jerk  = jerk_loss(I_schedule),
    )
    # fmt: on

    total_loss = sum(loss_weights[key] * val for key, val in losses.items())
    return total_loss, losses
