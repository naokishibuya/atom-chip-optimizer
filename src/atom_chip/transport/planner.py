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


def generate_desired_positions(
    r0: jnp.ndarray,
    num_shifts: int = 3,
    shifting_wire_distance: float = 0.4,  # mm
    steps_per_wire_distance: int = 20,
) -> jnp.ndarray:
    """
    Generates a linear trajectory for the trap center position.
    """
    step_size = shifting_wire_distance / steps_per_wire_distance
    num_steps = num_shifts * steps_per_wire_distance

    steps = jnp.arange(0, num_steps + 1).reshape(-1, 1)
    shift = steps * step_size
    return r0 + shift * jnp.array([1.0, 0.0, 0.0])


@jax.jit
def calculate_wire_currents(
    I_shifting_wires: jnp.ndarray,  # shape: (6,)
    I_guiding_wires: jnp.ndarray,  # shape: (9,)
) -> jnp.ndarray:
    """
    Expands 6 shifting and 9 guiding wires into full 73-element current vector.
    """
    # Expand shifting wires
    total_shifting = jnp.concatenate(
        [I_shifting_wires, -I_shifting_wires, I_shifting_wires, -I_shifting_wires, I_shifting_wires]
    )  # (30,)

    # Expand guiding wires
    total_guiding = jnp.repeat(I_guiding_wires, GUIDING_WIRE_SEGMENT_COUNTS)  # (43,)

    return jnp.concatenate([total_shifting, total_guiding])  # (73,)


def calculate_initial_wire_currents() -> jnp.ndarray:
    return calculate_wire_currents(
        I_shifting_wires=jnp.array(builder.SHIFTING_WIRE_CURRENTS, dtype=jnp.float64),
        I_guiding_wires=jnp.array(builder.GUIDING_WIRE_CURRENTS, dtype=jnp.float64),
    )


class WireCurrentPlanner(nn.Module):
    n_wires: int
    n_steps: int
    hidden_dim: int
    n_layers: int
    current_limits: jnp.ndarray  # shape (n_wires,)

    @nn.compact
    def __call__(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        # trajectory: (n_steps, 3)
        x = trajectory.reshape(-1)  # (n_steps * 3,)
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.n_wires * self.n_steps)(x)
        x = jnp.tanh(x).reshape(self.n_steps, self.n_wires)
        return x * self.current_limits  # shape: (n_steps, n_wires)
