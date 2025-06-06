from functools import partial
import jax
import jax.numpy as jnp
import atom_chip as ac
from .scheduler import ScheduleFn
from .metrics import simulate_trap_dynamics


# ---------- Spatial Stability Losses ----------


@jax.jit
def trap_depth_loss(U0s: jnp.ndarray, U0_ref: jnp.ndarray):
    """Penalizes trap depth below a minimum threshold."""
    return jnp.sum(jnp.square((U0s / U0_ref) - 1.0))


@jax.jit
def energy_smoothness_loss(U0s):
    return jnp.mean(jnp.square(U0s[1:] - U0s[:-1]))


@jax.jit
def trap_frequency_loss(omegas: jnp.ndarray, omega_ref: jnp.ndarray):
    """Penalizes deviation of trap frequency from reference."""
    return jnp.sum(jnp.square((omegas / omega_ref) - 1.0))


# ---------- Current Bound Losses ----------


@jax.jit
def current_bound_loss(currents: jnp.ndarray, I_max: jnp.ndarray):
    """Penalizes currents that exceed the maximum allowed value."""
    return jnp.sum(jnp.square(jnp.maximum(0.0, jnp.abs(currents) - I_max)))


# ---------- Time-Dependent (Adiabatic) Losses ----------


@jax.jit
def velocity_loss(r0s: jnp.ndarray) -> jnp.ndarray:
    diffs = r0s[1:] - r0s[:-1]
    return jnp.mean(jnp.square(diffs[:, 0]))  # only x-axis smoothing


@jax.jit
def frequency_change_loss(omegas: jnp.ndarray) -> jnp.ndarray:
    """Penalizes rapid changes in trap frequency."""
    return jnp.mean(jnp.square(omegas[1:] - omegas[:-1]))


@jax.jit
def jerk_loss(I_schedule: jnp.ndarray) -> jnp.ndarray:
    """Penalizes rapid changes in current."""
    return jnp.mean(jnp.square(I_schedule[2:] - 2 * I_schedule[1:-1] + I_schedule[:-2]))


# Total loss function that combines all components
@partial(jax.jit, static_argnames=("schedule_fn",))
def total_loss_fn(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.field.BiasFieldConfig,
    anchor_currents: jnp.ndarray,
    schedule_fn: ScheduleFn,
    trap_trajectory: jnp.ndarray,
    omega_ref: jnp.ndarray,
    U0_ref: jnp.ndarray,
    I_max: jnp.ndarray,
    loss_weights: dict,
) -> jnp.ndarray:
    """Computes the weighted sum of all physics-informed loss components."""

    I_schedule, U0s, omegas = simulate_trap_dynamics(
        atom,
        wire_config,
        bias_config,
        trap_trajectory,
        anchor_currents,
        schedule_fn,
    )

    # Compute losses
    # fmt: off
    losses = dict(
        U     = trap_depth_loss(U0s, U0_ref),
        dU    = energy_smoothness_loss(U0s),
        freq  = trap_frequency_loss(omegas, omega_ref),
        bound = current_bound_loss(I_schedule, I_max),
        vel   = velocity_loss(trap_trajectory),
        dfreq = frequency_change_loss(omegas),
        jerk  = jerk_loss(I_schedule),
    )
    # fmt: on

    total_loss = sum(loss_weights[key] * val for key, val in losses.items())
    return total_loss, losses
