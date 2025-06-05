"""
For BEC transport inverse optimization for shifting and guiding wire currents.
"""

from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
import optax
import atom_chip as ac
from atom_chip.transport import planner, plots


# Optimization parameters
class TransportParams(NamedTuple):
    shifting_current_deltas: jnp.ndarray  # [NUM_SHIFTING, NUM_STEPS]
    guiding_currents: jnp.ndarray  # [NUM_GUIDING, NUM_STEPS]


# @jax.jit
def reconstruct_currents(
    shifting_deltas: jnp.ndarray,  # (n_shifting, n_steps)
    guiding_currents: jnp.ndarray,  # (n_guiding, n_steps)
) -> jnp.ndarray:
    shifting = construct_shifting_schedule_from_deltas(shifting_deltas)
    shifting_full = jnp.concatenate(
        [
            shifting,
            -shifting,
            shifting,
            -shifting,
            shifting,
        ],
        axis=0,
    )  # (n_shifting_segments, n_steps)

    guiding_full = jnp.repeat(guiding_currents, planner.GUIDING_WIRES_SEGMENT_COUNTS, axis=0)
    # guiding_currents shape: (n_guiding, n_steps)
    # guiding_full shape: (n_guiding_segments, n_steps)

    return jnp.concatenate([shifting_full, guiding_full], axis=0)  # (n_total_segments, n_steps)


# @jax.jit
def construct_shifting_schedule_from_deltas(deltas: jnp.ndarray) -> jnp.ndarray:
    steps_per_block = planner.STEPS_PER_WIRE_DISTANCE
    targets = planner.I_SHIFTING_TARGETS
    num_blocks = len(targets) - 1

    I_blocks = []

    for i in range(num_blocks):
        start = targets[i]
        end = targets[i + 1]
        idx_start = i * steps_per_block
        idx_end = idx_start + steps_per_block

        delta_block = deltas[:, idx_start:idx_end]
        delta_block = jax.nn.softplus(delta_block)
        cumulative = jnp.cumsum(delta_block, axis=1)
        scale = (end - start) / (cumulative[:, -1] + 1e-6)
        block = start[:, None] + scale[:, None] * cumulative
        I_blocks.append(block)

    return jnp.concatenate(I_blocks, axis=1)


# @jax.jit
def evaluate_trap(
    wires: ac.atom_chip.AtomChipWires,
    currents: jnp.ndarray,
    bias: ac.field.BiasFieldParams,
    guess: jnp.ndarray,
    search_grid: jnp.ndarray,
) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
    """
    Locate trap minimum, evaluate energy and trap frequencies.

    Returns:
        r0: jnp.ndarray shape (3,)
        U0: float
        omega: jnp.ndarray shape (3,)
    """
    grid_points = guess + search_grid
    U, _, _ = ac.atom_chip.trap_potential_energies(grid_points, ac.rb87, wires, currents, bias)

    min_idx = jnp.argmin(U)
    r0 = grid_points[min_idx]
    U0 = U[min_idx]

    # Compute the Hessian matrix at the trap minimum to get the trap frequencies
    def potential_energy(r):
        U, _, _ = ac.atom_chip.trap_potential_energies(r, ac.rb87, wires, currents, bias)
        return U[0]

    H = ac.potential.hessian_by_finite_difference(potential_energy, r0, step=1e-3)
    eigenvalues = H.eigenvalues

    # trap frequencies in Hz (x, y, z directions)
    omega = jnp.sqrt(jnp.abs(eigenvalues) / ac.rb87.mass) / (2 * jnp.pi)
    return r0, U0, omega


# Define the loss function for optimization
@jax.jit
def loss_fn(
    params: TransportParams,
    wires: ac.atom_chip.AtomChipWires,
    bias: ac.atom_chip.BiasFieldParams,
    desired_positions: jnp.ndarray,
    U0_ref: float,
    omega_ref: jnp.ndarray,
    search_grid: jnp.ndarray,
):
    I_schedule = reconstruct_currents(params.shifting_current_deltas, params.guiding_currents)

    def step_loss(step):
        target_position = desired_positions[step]
        points = jnp.array([target_position])
        currents_t = I_schedule[:, step]

        r0_t, U0_t, omega_t = evaluate_trap(wires, currents_t, bias, points, search_grid)

        # loss calculation
        loss_pos = jnp.sum(((r0_t - target_position) / target_position) ** 2)
        loss_U = ((U0_t - U0_ref) / U0_ref) ** 2
        loss_freq = jnp.sum(((omega_t - omega_ref) / omega_ref) ** 2)

        loss = 0.001 * loss_pos + loss_U + loss_freq

        # guiding_diff = jnp.diff(params.guiding_currents, axis=1)
        # smoothness_penalty = jnp.sum(guiding_diff**2)
        # loss += 0.001 * smoothness_penalty

        # anchor_loss = jnp.sum((params.guiding_currents - guiding_anchor[:, None])**2)
        # loss += 0.01 * anchor_loss

        return loss

    # Compute the total loss over all steps
    transport_loss = jnp.sum(jax.vmap(step_loss)(jnp.arange(1, planner.NUM_STEPS + 1)))
    # TODO final_loss = jnp.sum((I_schedule[:, -1] - I_FINAL) ** 2)
    return transport_loss  # + 10.0 * final_loss


def build_anisotropic_search_grid(radii: jnp.ndarray, resolution: int) -> jnp.ndarray:
    lin_x = jnp.linspace(-radii[0], radii[0], resolution)
    lin_y = jnp.linspace(-radii[1], radii[1], resolution)
    lin_z = jnp.linspace(-radii[2], radii[2], resolution)
    grid_x, grid_y, grid_z = jnp.meshgrid(lin_x, lin_y, lin_z, indexing="ij")
    return jnp.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)


def main():
    wires = planner.build_atom_chip_wires()
    initial_currents = planner.initial_atom_chip_currents()
    bias = planner.build_atom_chip_bias_fields()

    # Initialize the reference energy and trap frequencies
    initial_position = jnp.array([1.112e-07, -2.93e-06, 0.34725])  # from transport.py results
    TF_radii = jnp.array([9.96, 3.185, 3.032]) * 1e-3  # in mm

    search_grid = build_anisotropic_search_grid(TF_radii, 10)

    r0_ref, U0_ref, omega_ref = evaluate_trap(wires, initial_currents, bias, initial_position, search_grid)
    desired_positions = planner.generate_desired_positions(r0_ref)
    print(f"Initial trap pos: {r0_ref}, U0: {U0_ref:.4g}, omega: {omega_ref}")
    print(f"Desired final position: {desired_positions[-1]}")

    # Initialize the transport parameters with random deltas for shifting wires
    key = jax.random.PRNGKey(0)
    shifting_current_deltas = jax.random.normal(key, (planner.NUM_SHIFTING_WIRES, planner.NUM_STEPS)) * 0.01
    guiding_currents = jnp.tile(planner.I_GUIDING_WIRES[:, None], (1, planner.NUM_STEPS))
    params = TransportParams(
        shifting_current_deltas=shifting_current_deltas,
        guiding_currents=guiding_currents,
    )

    # Optimization setup
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(
            params, wires, bias, desired_positions, U0_ref, omega_ref, search_grid
        )
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss, grad_norm

    for step in range(1000):
        params, opt_state, loss, grad_norm = update(params, opt_state)
        if step % 100 == 0:
            # Print the trap evaluation at the last step of the schedule
            I_schedule = reconstruct_currents(params.shifting_current_deltas, params.guiding_currents)
            r0_T, U0_T, omega_T = evaluate_trap(wires, I_schedule[:, -1], bias, desired_positions[-1], search_grid)
            print(
                " ".join(
                    [
                        f"Step {step:03d}:",
                        f"Trap position: {r0_T}",
                        f"Trap potential U0: {U0_T:.4g}",
                        f"Trap frequency omega: {omega_T}",
                        f"Loss: {loss:.4g}",
                        f"Grad norm: {grad_norm:.4g}",
                    ]
                )
            )

    # After optimization, evaluate the trap positions and potentials (including the initial currents)
    I_final_schedule = reconstruct_currents(params.shifting_current_deltas, params.guiding_currents)
    r0s, U0s, omegas = scan_trap_metrics(wires, I_final_schedule, bias, desired_positions, search_grid)
    print(f"Final trap pos: {r0s[-1]}, U0: {U0s[-1]:.04g}, omega: {omegas[-1]}")

    # final block schedule
    I_guiding_wires = construct_shifting_schedule_from_deltas(params.shifting_current_deltas)
    I_schedule = jnp.concatenate([I_guiding_wires, params.guiding_currents], axis=0)

    # Plot the results
    plots.show(r0s, desired_positions, I_schedule, U0s, omegas, search_grid)


def scan_trap_metrics(wires, I_schedule, bias, desired_positions, search_grid):
    def step_fn(carry, step):
        currents = I_schedule[:, step]
        r_target = desired_positions[step]
        r_trap, U0, omega = evaluate_trap(wires, currents, bias, r_target, search_grid)
        return carry, (r_trap, U0, omega)

    num_steps = I_schedule.shape[1]
    indices = jnp.arange(num_steps)

    # lax.scan requires outputs to be arrays of equal shape and dtype
    _, outputs = jax.lax.scan(step_fn, init=None, xs=indices)

    positions, depths, frequencies = outputs
    return jnp.stack(positions), jnp.stack(depths), jnp.stack(frequencies)


if __name__ == "__main__":
    main()
