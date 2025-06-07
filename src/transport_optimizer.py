from typing import NamedTuple
import jax
import jax.numpy as jnp
import optax
import atom_chip as ac
from atom_chip.transport import (
    loss_func,
    metrics,
    planner,
    plots,
)


class TransportParams(NamedTuple):
    anchor_currents: jnp.ndarray  # shape (n_anchors, n_wires)
    trap_trajectory: jnp.ndarray  # shape (n_steps, 3)


def main():
    wire_config = planner.setup_wire_config()
    bias_config = planner.setup_bias_config()
    initial_currents = planner.calculate_initial_wire_currents()

    # Initial reference values for the trap
    r0_ref = jnp.array([1.112e-07, -2.93e-06, 0.34725])  # from transport.py results
    U0_ref, omega_ref = metrics.evaluate_trap(
        atom=ac.rb87,
        wire_config=wire_config,
        bias_config=bias_config,
        wire_currents=initial_currents,
        trap_position=r0_ref,
    )
    trap_trajectory = planner.generate_desired_positions(r0_ref)

    print(f"Initial trap pos: {r0_ref}, U0: {U0_ref:.4g}, omega: {omega_ref}")
    print(f"Desired final position: {trap_trajectory[-1]}")

    # Hyperparameters for optimization
    loss_weights = dict(
        U=1.0,
        dU=1.0,
        freq=0.1,
        bound=0.0,  # TODO only for shifting wires
        vel=1e3,
        dfreq=1e3,
        jerk=1e4,
    )

    I_shifting_wire_max = jnp.array([1.0] * 6)  # Max current for shifting wires
    I_guiding_wire_max = jnp.array([15.0] * 9)  # Max current for guiding wires
    I_max = jnp.concatenate([I_shifting_wire_max, I_guiding_wire_max])

    final_I_schedule, loss_log = run_optimization(
        atom=ac.rb87,
        wire_config=wire_config,
        bias_config=bias_config,
        trap_trajectory=trap_trajectory,
        omega_ref=omega_ref,
        U0_ref=U0_ref,
        I_max=I_max,
        max_steps=1000,
        learning_rate=0.01,
        loss_weights=loss_weights,
    )

    # Plot the results
    plots.show(
        wire_config=wire_config,
        bias_config=bias_config,
        trap_trajectory=trap_trajectory,
        I_schedule=final_I_schedule,
        loss_log=loss_log,
    )


# Find the optimal anchor currents for the atom chip transport
def run_optimization(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.field.BiasFieldConfig,
    trap_trajectory: jnp.ndarray,
    omega_ref: jnp.ndarray,
    U0_ref: jnp.ndarray,
    I_max: jnp.ndarray,
    max_steps: int,
    learning_rate: float,
    loss_weights: dict,
) -> jnp.ndarray:
    # Initialize the planner model
    model = planner.WireCurrentPlanner(
        n_wires=15,  # 6 shifting + 9 guiding
        n_steps=trap_trajectory.shape[0],
        hidden_dim=128,
        n_layers=3,
        current_limits=I_max,  # shape: (n_wires,)
    )

    # Model init input should match __call__ signature: (n_steps, 3)
    rng = jax.random.PRNGKey(42)
    input_trajectory = trap_trajectory  # shape: (n_steps, 3)
    variables = model.init(rng, input_trajectory)
    params = variables["params"]

    # Define optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Define the loss function with aux outputs
    def loss_fn(params):
        I_schedule = model.apply({"params": params}, input_trajectory)  # shape: (n_wires, n_steps)

        loss, losses = loss_func.total_loss_fn(
            atom,
            wire_config,
            bias_config,
            trap_trajectory,
            I_schedule,
            omega_ref,
            U0_ref,
            I_max,
            loss_weights,
        )
        return loss, losses

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # Tracking loss components
    loss_log = {key: [] for key in loss_weights}
    loss_log["total"] = []

    for step in range(1, max_steps + 1):
        (total_loss, losses), grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Log
        loss_log["total"].append(total_loss)
        for key in loss_weights:
            loss_log[key].append(losses[key])

        if step % 50 == 0 or step == max_steps:
            detail_str = " ".join(f"{k}: {losses[k]:.2e}" for k in loss_weights)
            print(f"Step {step:04d}: Loss={total_loss:.4e} ({detail_str})")

    # Final I_schedule
    final_I_schedule = model.apply({"params": params}, input_trajectory)
    return final_I_schedule, loss_log


if __name__ == "__main__":
    main()
