import jax.numpy as jnp
import atom_chip as ac
from . import builder


# Configuration parameters for the transport optimization
NUM_SHIFTS = 3
SHIFTING_WIRE_DISTANCE = 0.4  # mm
TOTAL_DISTANCE = NUM_SHIFTS * SHIFTING_WIRE_DISTANCE
STEPS_PER_WIRE_DISTANCE = 20
STEP_SIZE = SHIFTING_WIRE_DISTANCE / STEPS_PER_WIRE_DISTANCE
NUM_STEPS = NUM_SHIFTS * STEPS_PER_WIRE_DISTANCE

# Define the wire currents and their initial and final states
I_SHIFTING_WIRES = jnp.array(builder.SHIFTING_WIRE_CURRENTS, dtype=jnp.float64)
I_GUIDING_WIRES = jnp.array(builder.GUIDING__WIRE_CURRENTS, dtype=jnp.float64)
NUM_SHIFTING_WIRES = len(I_SHIFTING_WIRES)
NUM_GUIDING_WIRES = len(I_GUIDING_WIRES)
NUM_WIRES = NUM_SHIFTING_WIRES + NUM_GUIDING_WIRES  # Total number of wires (shifting wires are connected in series)

# Initial, shifting and final current states for the optimization
I_INIT = jnp.concatenate([I_SHIFTING_WIRES, I_GUIDING_WIRES])
I_FINAL = jnp.concatenate([jnp.roll(I_SHIFTING_WIRES, NUM_SHIFTS), I_GUIDING_WIRES])
I_SHIFTING_TARGETS = jnp.stack([jnp.roll(I_SHIFTING_WIRES, shift) for shift in range(NUM_SHIFTS + 1)])

SHIFTING_WIRES, GUIDING_WIRES = builder.setup_wire_layout()
GUIDING_WIRES_SEGMENT_COUNTS = jnp.array([len(wire) for wire in GUIDING_WIRES])


# Generate the desired positions for the trap at each step
def generate_desired_positions(r0: jnp.ndarray) -> jnp.ndarray:
    # this includes the initial position and the positions at each step afterwards
    steps = jnp.arange(0, NUM_STEPS + 1).reshape(-1, 1)
    shift = steps * STEP_SIZE
    return r0 + shift * jnp.array([1.0, 0.0, 0.0])


# Generate a grid of points around the trap minimum for evaluation
def trap_minimum_search_grid(grid_width: float, resolution: int) -> jnp.ndarray:
    grid_half_width = grid_width / 2.0
    lin = jnp.linspace(-grid_half_width, grid_half_width, resolution)
    dx, dy, dz = jnp.meshgrid(lin, lin, lin, indexing="ij")
    return jnp.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=-1)


def build_atom_chip_wires() -> ac.atom_chip.AtomChipWires:
    starts = []
    ends = []
    widths = []
    heights = []

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

    return ac.atom_chip.AtomChipWires(
        starts=jnp.stack(starts),
        ends=jnp.stack(ends),
        widths=jnp.stack(widths),
        heights=jnp.stack(heights),
    )


def initial_atom_chip_currents() -> jnp.ndarray:
    """
    Reconstruct the current schedule for the atom chip based on shifting and guiding wire currents.

    guiding_wire_segment_counts: jnp.ndarray shape (num_guiding_wires,), giving segment count per guiding wire.
    """
    total_shifting = jnp.concatenate(
        [
            I_SHIFTING_WIRES,
            -I_SHIFTING_WIRES,
            I_SHIFTING_WIRES,
            -I_SHIFTING_WIRES,
            I_SHIFTING_WIRES,
        ]
    )

    total_guiding = jnp.repeat(I_GUIDING_WIRES, GUIDING_WIRES_SEGMENT_COUNTS)

    return jnp.concatenate([total_shifting, total_guiding])


def build_atom_chip_bias_fields() -> ac.atom_chip.BiasFieldParams:
    """
    Build the bias fields for the atom chip.
    """
    bias_fields = builder.make_bias_fields()
    return bias_fields.params
