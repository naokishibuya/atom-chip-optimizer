import jax
import jax.numpy as jnp
import flax.linen as nn
import atom_chip as ac
from . import builder


SHIFTING_WIRES, GUIDING_WIRES = builder.setup_wire_layout()
GUIDING_WIRE_SEGMENT_COUNTS = jnp.array([len(wire) for wire in GUIDING_WIRES], dtype=jnp.int32)


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


def setup_bias_config() -> ac.atom_chip.BiasFieldConfig:
    """
    Build the bias fields for the atom chip.
    """
    bias_fields = builder.make_bias_fields()
    return bias_fields.config


def generate_input_values(
    r0: jnp.ndarray,
    num_shifts: int = 3,
    shifting_wire_distance: float = 0.4,  # mm
    steps_per_wire_distance: int = 20,
) -> jnp.ndarray:
    """
    Generates a linear trajectory for the trap center position, and start and end currents for the wires.
    """
    step_size = shifting_wire_distance / steps_per_wire_distance
    num_steps = num_shifts * steps_per_wire_distance

    steps = jnp.arange(0, num_steps + 1).reshape(-1, 1)
    shift = steps * step_size
    trajectory = r0 + shift * jnp.array([1.0, 0.0, 0.0])

    I_shifting_wires = jnp.array(builder.SHIFTING_WIRE_CURRENTS, dtype=jnp.float64)
    I_guiding_wires = jnp.array(builder.GUIDING_WIRE_CURRENTS, dtype=jnp.float64)

    I_start = jnp.concatenate([I_shifting_wires, I_guiding_wires])
    I_end = jnp.concatenate([jnp.roll(I_shifting_wires, num_shifts), I_guiding_wires])
    return trajectory, I_start, I_end


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


# Vectorized version for batch (current schedule) processing
distribute_current_schedule_to_wires = jax.vmap(distribute_currents_to_wires)


def calculate_initial_wire_currents() -> jnp.ndarray:
    I_wires = jnp.concatenate(
        [
            jnp.array(builder.SHIFTING_WIRE_CURRENTS, dtype=jnp.float64),
            jnp.array(builder.GUIDING_WIRE_CURRENTS, dtype=jnp.float64),
        ]
    )
    return distribute_currents_to_wires(I_wires)


class WireCurrentPlanner(nn.Module):
    n_wires: int
    n_steps: int
    hidden_dim: int
    n_layers: int
    I_limits: jnp.ndarray  # shape (n_wires,)

    # Constants (define as class variables)
    SPLIT_INDEX_1 = 7  # 6 shifting wires + 1 guiding wire
    SPLIT_INDEX_2 = 14  # last guiding wire index = 14
    N_PREDICTED_WIRES = SPLIT_INDEX_1 + (15 - SPLIT_INDEX_2)

    @nn.compact
    def __call__(self, trajectory: jnp.ndarray, I_start: jnp.ndarray, I_end: jnp.ndarray) -> jnp.ndarray:
        # input shape: (n_steps, 3), (n_wires,), (n_wires,)
        x = trajectory.reshape(-1)  # flatten trajectory to 1D vector (n_steps * 3,)
        x = jnp.concatenate([x, I_start, I_end])  # (n_steps * 3 + n_wires + n_wires,)

        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)

        x = nn.Dense((self.n_steps - 2) * self.N_PREDICTED_WIRES)(x)
        x = jnp.tanh(x).reshape(self.n_steps - 2, self.N_PREDICTED_WIRES)

        # Limits for predicted wires
        limits_1 = self.I_limits[: self.SPLIT_INDEX_1]
        limits_2 = self.I_limits[self.SPLIT_INDEX_2 :]
        x = x * jnp.concatenate([limits_1, limits_2], axis=0)

        # Split the output into the first and second predicted segments
        I_pred_1 = x[:, : self.SPLIT_INDEX_1]
        I_pred_2 = x[:, self.SPLIT_INDEX_1 :]

        # Fixed guiding wires (guiding 1–7)
        I_fixed_guiding = I_start[self.SPLIT_INDEX_1 : self.SPLIT_INDEX_2]  # guiding wire currents 1–7
        I_fixed_block = jnp.tile(I_fixed_guiding[None, :], (self.n_steps - 2, 1))

        I_middle = jnp.concatenate([I_pred_1, I_fixed_block, I_pred_2], axis=1)
        I_schedule = jnp.concatenate([I_start[None, :], I_middle, I_end[None, :]], axis=0)
        return I_schedule
