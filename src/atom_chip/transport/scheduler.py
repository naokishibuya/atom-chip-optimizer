from typing import Protocol
import jax
import jax.numpy as jnp


# Scheduler function signature
class ScheduleFn(Protocol):
    def __call__(self, trajectory: jnp.ndarray, anchor_currents: jnp.ndarray) -> jnp.ndarray: ...


def interpolate_currents_cosine(trajectory: jnp.ndarray, anchor_currents: jnp.ndarray) -> jnp.ndarray:
    """
    Cosine interpolation between anchor points using JAX-compatible operations.
    Args:
        anchor_currents: shape (n_anchors, n_wires)
        n_steps: number of output steps
    Returns:
        schedule: shape (n_steps, n_wires)
    """
    n_steps = trajectory.shape[0]
    n_anchors, n_wires = anchor_currents.shape
    n_segments = n_anchors - 1

    steps_per_segment = n_steps // n_segments
    extra_steps = n_steps % n_segments

    # Compute interpolation fractions t in [0, 1) for each segment
    t = jnp.linspace(0, 1, steps_per_segment, endpoint=False)
    f = 0.5 * (1 - jnp.cos(jnp.pi * t))  # shape (steps_per_segment,)

    # Broadcast to all segments
    def interpolate_pair(pair):
        I0, I1 = pair
        return I0[None, :] + (I1 - I0)[None, :] * f[:, None]  # shape (steps_per_segment, n_wires)

    I0s = anchor_currents[:-1]  # shape (n_segments, n_wires)
    I1s = anchor_currents[1:]  # shape (n_segments, n_wires)
    segment_pairs = jnp.stack([I0s, I1s], axis=1)  # shape (n_segments, 2, n_wires)

    interpolated_segments = jax.vmap(interpolate_pair)(segment_pairs)  # (n_segments, steps_per_segment, n_wires)
    I_schedule = interpolated_segments.reshape(-1, n_wires)  # shape (n_segments * steps_per_segment, n_wires)

    # Pad if needed
    if extra_steps > 0:
        pad = jnp.repeat(anchor_currents[-1][None, :], extra_steps, axis=0)
        I_schedule = jnp.vstack([I_schedule, pad])

    return I_schedule


def select_schedule_fn(name: str) -> ScheduleFn:
    """
    Select a schedule function by name.
    """
    functions = {
        "cosine": interpolate_currents_cosine,
    }
    if name not in functions:
        raise ValueError(f"Unknown schedule: {name}. Available options: {list(functions.keys())}")
    return jax.jit(functions[name])  # Apply JIT only to selected schedule
