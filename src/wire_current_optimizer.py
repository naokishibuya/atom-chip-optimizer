"""
For BEC transport inverse optimization for shifting and guiding wire currents.
"""

from typing import Tuple, NamedTuple
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
import jax
import jax.numpy as jnp
import optax
import atom_chip as ac
import transport  # BEC transport module


# Define the configuration for the transport optimization
class TransportConfig:
    NUM_SHIFTS = 3
    SHIFTING_WIRE_DISTANCE = 0.4  # mm
    TOTAL_DISTANCE = NUM_SHIFTS * SHIFTING_WIRE_DISTANCE
    STEPS_PER_WIRE_DISTANCE = 20
    STEP_SIZE = SHIFTING_WIRE_DISTANCE / STEPS_PER_WIRE_DISTANCE
    NUM_STEPS = NUM_SHIFTS * STEPS_PER_WIRE_DISTANCE
    # Initial guess for the trap position (in mm) at the start of the transport
    INITIAL_POSITION = jnp.array([1.112e-07, -2.93e-06, 0.34725])  # from transport.py results


config = TransportConfig()

# Define the wire currents and their initial and final states
I_SHIFTING_WIRES = jnp.array(transport.SHIFTING_WIRE_CURRENTS, dtype=jnp.float64)
I_GUIDING_WIRES = jnp.array(transport.GUIDING__WIRE_CURRENTS, dtype=jnp.float64)
NUM_SHIFTING_WIRES = len(I_SHIFTING_WIRES)
NUM_GUIDING_WIRES = len(I_GUIDING_WIRES)
NUM_WIRES = NUM_SHIFTING_WIRES + NUM_GUIDING_WIRES  # Total number of wires (shifting wires are connected in series)

# Initial, shifting and final current states for the optimization
I_INIT = jnp.concatenate([I_SHIFTING_WIRES, I_GUIDING_WIRES])
I_FINAL = jnp.concatenate([jnp.roll(I_SHIFTING_WIRES, config.NUM_SHIFTS), I_GUIDING_WIRES])
I_SHIFTING_TARGETS = jnp.stack([jnp.roll(I_SHIFTING_WIRES, shift) for shift in range(config.NUM_SHIFTS + 1)])

# Setup the wire layout and bias fields for the atom chip
SHIFTING_WIRES, GUIDING_WIRES = transport.setup_wire_layout()
BIAS_FIELDS = transport.make_bias_fields()


# Optimization parameters
class TransportParams(NamedTuple):
    shifting_current_deltas: jnp.ndarray  # [NUM_SHIFTING, NUM_STEPS]
    guiding_currents: jnp.ndarray  # [NUM_GUIDING]


# Generate the desired positions for the trap at each step
def generate_desired_positions(r0: jnp.ndarray) -> jnp.ndarray:
    # this includes the initial position and the positions at each step afterwards
    steps = jnp.arange(0, config.NUM_STEPS + 1).reshape(-1, 1)
    shift = steps * config.STEP_SIZE
    return r0 + shift * jnp.array([1.0, 0.0, 0.0])


# Generate a grid of points around the trap minimum for evaluation
def trap_minimum_grid(grid_width: float, resolution: int) -> jnp.ndarray:
    grid_half_width = grid_width / 2.0
    lin = jnp.linspace(-grid_half_width, grid_half_width, resolution)
    dx, dy, dz = jnp.meshgrid(lin, lin, lin, indexing="ij")
    return jnp.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=-1)


# Get the current schedule based on the transport parameters
@jax.jit
def get_current_schedule(params: TransportParams) -> jnp.ndarray:
    I_shifting = [I_SHIFTING_WIRES[:, None]]  # Start with initial currents
    for i, (start, end) in enumerate(zip(I_SHIFTING_TARGETS[:-1], I_SHIFTING_TARGETS[1:])):
        idx = i * config.STEPS_PER_WIRE_DISTANCE
        deltas = jax.nn.softplus(params.shifting_current_deltas[:, idx : idx + config.STEPS_PER_WIRE_DISTANCE])
        cumulative = jnp.cumsum(deltas, axis=1)
        scale = (end - start) / (cumulative[:, -1] + 1e-6)  # Avoid division by zero
        block_schedule = start[:, None] + scale[:, None] * cumulative
        I_shifting.append(block_schedule)

    I_shifting = jnp.concatenate(I_shifting, axis=1)
    I_guiding = params.guiding_currents[:, None].repeat(I_shifting.shape[1], axis=1)
    return jnp.concatenate([I_shifting, I_guiding], axis=0)


# Evaluate the trap position, potential energy, and frequency at a given set of currents
@jax.jit
def evaluate_trap(
    currents: jnp.ndarray,
    initial_guess: jnp.ndarray,
    search_grid: jnp.ndarray,
) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
    # Build the atom chip with the given currents
    atom_chip = transport.build_atom_chip(
        shifting_wires=SHIFTING_WIRES,
        shifting_wire_currents=currents[:NUM_SHIFTING_WIRES],
        guiding_wires=GUIDING_WIRES,
        guiding_wire_currents=currents[NUM_SHIFTING_WIRES:],
        bias_fields=BIAS_FIELDS,
    )

    # Define a function to compute the potential energy at a point
    def potential_energy(point):
        E, _, _ = atom_chip.get_potentials(point)
        return E[0]

    grid_points = search_grid + initial_guess
    potentials = jax.vmap(potential_energy)(grid_points)
    min_idx = jnp.argmin(potentials)
    r_trap = grid_points[min_idx]
    U0 = potentials[min_idx]

    # Compute the Hessian matrix at the trap minimum to get the trap frequencies
    # H = jax.hessian(potential_energy)(r_trap)
    # eigenvalues, _ = jnp.linalg.eigh(H)
    H = ac.potential.hessian_by_finite_difference(potential_energy, r_trap, step=config.STEP_SIZE / 2.0)
    eigenvalues = H.eigenvalues

    # trap frequencies in Hz (x, y, z directions)
    m = atom_chip.atom.mass
    omega = jnp.sqrt(eigenvalues / m) / (2 * jnp.pi)
    return r_trap, U0, omega


def optimize_transport():
    # Initial trap position and potential
    search_grid = trap_minimum_grid(config.STEP_SIZE, config.STEPS_PER_WIRE_DISTANCE)
    init_r, init_U0, init_omega = evaluate_trap(I_INIT, config.INITIAL_POSITION, search_grid)
    print(f"Initial trap position: {init_r}, U0: {init_U0}, omega: {init_omega}")

    # Generate the desired positions for the trap at each step
    desired_r = generate_desired_positions(init_r)

    # Initialize the transport parameters with random deltas for shifting wires
    key = jax.random.PRNGKey(0)
    shifting_current_deltas = jax.random.normal(key, (NUM_SHIFTING_WIRES, config.NUM_STEPS)) * 0.01
    params = TransportParams(
        shifting_current_deltas=shifting_current_deltas,
        guiding_currents=I_GUIDING_WIRES,
    )

    # Define the loss function for optimization
    @jax.jit
    def loss_fn(params: TransportParams):
        I_schedule = get_current_schedule(params)

        def step_loss(step):
            r_target = desired_r[step]
            currents = I_schedule[:, step]
            r_trap, U0, omega = evaluate_trap(currents, r_target, search_grid)
            pos_loss = jnp.sum((r_trap - r_target) ** 2)
            depth_loss = jnp.maximum(init_U0 - U0, 0.0) ** 2
            freq_loss = jnp.maximum(init_omega - omega, 0.0) ** 2
            return pos_loss + 3.0 * depth_loss + 1.0 * freq_loss

        # Compute the total loss over all steps
        transport_loss = jnp.sum(jax.vmap(step_loss)(jnp.arange(1, config.NUM_STEPS + 1)))
        final_loss = jnp.sum((I_schedule[:, -1] - I_FINAL) ** 2)
        return transport_loss + 10.0 * final_loss

    # Optimization setup
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = optimizer.update(grads, state)
        new_params = optax.apply_updates(params, updates)
        return new_params, state, loss

    for i in range(1000):
        params, opt_state, loss = step(params, opt_state)
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")

    # After optimization, evaluate the trap positions and potentials (including the initial currents)
    I_final_schedule = get_current_schedule(params)

    positions = []
    depths = []
    frequencies = []
    for step in range(config.NUM_STEPS + 1):
        r_trap, U0, omega = evaluate_trap(I_final_schedule[:, step], desired_r[step], search_grid)
        positions.append(r_trap)
        depths.append(U0)
        frequencies.append(omega)

    positions = jnp.stack(positions)
    depths = jnp.stack(depths)
    frequencies = jnp.stack(frequencies)
    print(f"Final trap position: {positions[-1]}, U0: {depths[-1]}, omega: {frequencies[-1]}")

    # Plot the final trap positions and current schedules
    fig1 = plot_trap_positions(positions, desired_r)
    fig2 = plot_current_schedule(I_final_schedule)
    fig3 = plot_trap_depths(depths)
    fig4 = plot_trap_frequencies(frequencies)

    # Adjust the plot windows to be side by side
    start_top, start_left = 100, 100
    geo1 = adjust_plot_window(fig1, top=start_top, left=start_left)
    geo2 = adjust_plot_window(fig2, top=geo1.top(), left=geo1.left() + geo1.width() + 10)
    adjust_plot_window(fig3, top=geo2.top(), left=geo2.left() + geo2.width() + 10)
    adjust_plot_window(fig4, top=geo1.top() + geo1.height() + 20, left=start_left)
    plt.show(block=False)

    input("Press Enter to close the plots...")


# Plot the actual and desired trap positions over time
def plot_trap_positions(positions, desired):
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, label in enumerate(["x", "y", "z"]):
        axs[i].plot(positions[:, i], label="Actual", color="blue")
        axs[i].plot(desired[:, i], "--", label="Desired", color="green")
        axs[i].set_ylabel(label + " (mm)")
        axs[i].legend()
        axs[i].grid()
    axs[-1].set_xlabel("Step")
    plt.tight_layout()
    return fig


# Plot current schedules for shifting and guiding wires
def plot_current_schedule(I_schedule):
    # Plot shifting wire currents
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for i in range(NUM_SHIFTING_WIRES):
        ax1.plot(I_schedule[i], label=f"Shifting Wire {i}")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Current (A)")
    ax1.set_title("Shifting Currents")
    ax1.grid()

    # Plot guiding wire currents
    x = jnp.arange(I_schedule.shape[1])
    shift_amt = 0.4  # how much to shift in x
    for i in range(NUM_GUIDING_WIRES):
        x_shifted = x + (i - NUM_GUIDING_WIRES // 2) * shift_amt / NUM_GUIDING_WIRES
        ax2.plot(
            x_shifted,
            I_schedule[NUM_SHIFTING_WIRES + i],
            label=f"Guiding Wire {i}",
            linestyle="None",
            marker=".",
            markersize=3,
        )
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Current (A)")
    ax2.set_title("Guiding Currents")
    ax2.grid()

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # leave space for legend
    return fig


def plot_trap_depths(depths):
    # Plot U0 (trap depth)
    fig = plt.figure(figsize=(8, 4))
    plt.plot(depths)
    plt.xlabel("Step")
    plt.ylabel("Trap Minimum $U_0$")
    plt.title("Trap Depth Over Transport")
    plt.grid()
    plt.tight_layout()
    return fig


def plot_trap_frequencies(frequencies):
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = jnp.arange(config.NUM_STEPS + 1)
    labels = ["$\\omega_x$", "$\\omega_y$", "$\\omega_z$"]
    for i in range(3):
        ax.plot(steps, frequencies[:, i], label=labels[i])
    ax.set_xlabel("Step")
    ax.set_ylabel("Trap Frequency (Hz)")
    ax.set_title("Trap Frequencies Over Transport")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    return fig


def adjust_plot_window(fig, top: int, left: int, z_top: bool = False):
    window = fig.canvas.manager.window
    if z_top:
        window.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    geometry = window.geometry()
    window.setGeometry(left, top, geometry.width(), geometry.height())
    return window.geometry()


if __name__ == "__main__":
    optimize_transport()
