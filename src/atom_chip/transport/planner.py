import jax.numpy as jnp
import atom_chip as ac
from . import builder


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


def initialize_anchor_currents(initial_currents: jnp.ndarray, n_anchors: int) -> jnp.ndarray:
    """
    Uniformly duplicate the initial wire currents across anchor points.
    """
    return jnp.tile(initial_currents[None, :], (n_anchors, 1))


def setup_wire_config() -> ac.atom_chip.WireConfig:
    """
    Build the wire configuration for the atom chip.
    """
    starts = []
    ends = []
    widths = []
    heights = []

    # Collect segments for shifting and guiding wires
    shifting_wires, guiding_wires = builder.setup_wire_layout()

    for segment in shifting_wires:
        s, e, w, h = segment
        starts.append(jnp.array(s, dtype=jnp.float64))
        ends.append(jnp.array(e, dtype=jnp.float64))
        widths.append(jnp.array(w, dtype=jnp.float64))
        heights.append(jnp.array(h, dtype=jnp.float64))

    for wire in guiding_wires:
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

    # Convert lists to JAX arrays
    i_shifting_wires = jnp.array(builder.SHIFTING_WIRE_CURRENTS, dtype=jnp.float64)
    total_shifting = jnp.concatenate(
        [
            i_shifting_wires,
            -i_shifting_wires,
            i_shifting_wires,
            -i_shifting_wires,
            i_shifting_wires,
        ]
    )

    i_guiding_wires = jnp.array(builder.GUIDING__WIRE_CURRENTS, dtype=jnp.float64)
    guiding_wire_segment_counts = jnp.array([len(wire) for wire in guiding_wires])
    total_guiding = jnp.repeat(i_guiding_wires, guiding_wire_segment_counts)

    initial_currents = jnp.concatenate([total_shifting, total_guiding])

    return wire_config, initial_currents


def setup_bias_config() -> ac.atom_chip.BiasFieldConfig:
    """
    Build the bias fields for the atom chip.
    """
    bias_fields = builder.make_bias_fields()
    return bias_fields.config
