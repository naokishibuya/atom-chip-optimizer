from typing import NamedTuple
import jax
import jax.numpy as jnp
import optax
import atom_chip as ac
from atom_chip.transport import (
    loss_func,
    metrics,
    planner,
    scheduler,
    plots,
)


class TransportParams(NamedTuple):
    anchor_currents: jnp.ndarray  # shape (n_anchors, n_wires)
    trap_trajectory: jnp.ndarray  # shape (n_steps, 3)


# Find the optimal anchor currents for the atom chip transport
def run_optimization(
    atom: ac.Atom,
    wire_config: ac.atom_chip.WireConfig,
    bias_config: ac.field.BiasFieldConfig,
    initial_trap_trajectory: jnp.ndarray,
    init_anchor_currents: jnp.ndarray,
    schedule_fn: scheduler.ScheduleFn,
    omega_ref: jnp.ndarray,
    U0_ref: jnp.ndarray,
    I_max: jnp.ndarray,
    max_steps: int,
    learning_rate: float,
    loss_weights: dict,
) -> jnp.ndarray:
    # Minimization objective function in terms of transport parameters
    def objective_fn(params: TransportParams) -> jnp.ndarray:
        return loss_func.total_loss_fn(
            atom,
            wire_config,
            bias_config,
            params.anchor_currents,
            schedule_fn,
            params.trap_trajectory,
            omega_ref,
            U0_ref,
            I_max,
            loss_weights,
        )

    loss_grad_fn = jax.value_and_grad(objective_fn, has_aux=True)

    # Initialize transport parameters
    params = TransportParams(
        anchor_currents=init_anchor_currents,
        trap_trajectory=initial_trap_trajectory,
    )

    # Initialize optimizer
    opt = optax.adam(learning_rate)
    opt_state = opt.init(params)

    # Run the optimization loop
    loss_log = {key: [] for key in loss_weights.keys()}
    loss_log["total"] = []
    for step in range(1, max_steps + 1):
        # Compute loss and gradients
        (total_loss, losses), grads = loss_grad_fn(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Store losses for analysis
        loss_log["total"].append(total_loss)
        for key in losses.keys():
            loss_log[key].append(losses[key])

        if step % 50 == 0 or step == max_steps - 1:
            # Individual loss terms
            loss_details = " ".join(f"{k}: {losses[k]:.2e}" for k in loss_weights.keys())

            # Gradient norms
            grad_norm_anchor = jnp.linalg.norm(grads.anchor_currents)
            grad_norm_trap = jnp.linalg.norm(grads.trap_trajectory)
            grad_details = f"Grad-Norm (Anchor={grad_norm_anchor:.4e} Traj={grad_norm_trap:.4e})"

            print(f"Step {step:04d}: Loss={total_loss:.4e} ({loss_details}) {grad_details}")

    return params.anchor_currents, params.trap_trajectory, loss_log


def main():
    wire_config, initial_currents = planner.setup_wire_config()
    bias_config = planner.setup_bias_config()

    # Initial reference values for the trap
    r0_ref = jnp.array([1.112e-07, -2.93e-06, 0.34725])  # from transport.py results
    U0_ref, omega_ref = metrics.evaluate_trap(
        atom=ac.rb87,
        wire_config=wire_config,
        bias_config=bias_config,
        wire_currents=initial_currents,
        trap_position=r0_ref,
    )
    initial_trap_trajectory = planner.generate_desired_positions(r0_ref)

    print(f"Initial trap pos: {r0_ref}, U0: {U0_ref:.4g}, omega: {omega_ref}")
    print(f"Desired final position: {initial_trap_trajectory[-1]}")

    # Hyperparameters for optimization
    schedule_fn = scheduler.select_schedule_fn("cosine")
    loss_weights = dict(
        U=1.0,
        dU=1.0,
        freq=0.1,
        bound=0.0,  # TODO only for shifting wires
        vel=1e3,
        dfreq=1e3,
        jerk=1e4,
    )

    init_anchor_currents = planner.initialize_anchor_currents(
        initial_currents=initial_currents,
        n_anchors=6,
    )

    anchor_currents, trap_trajectory, losses = run_optimization(
        atom=ac.rb87,
        wire_config=wire_config,
        bias_config=bias_config,
        initial_trap_trajectory=initial_trap_trajectory,
        init_anchor_currents=init_anchor_currents,
        schedule_fn=schedule_fn,
        omega_ref=omega_ref,
        U0_ref=U0_ref,
        I_max=1.0,
        max_steps=1000,
        learning_rate=0.01,
        loss_weights=loss_weights,
    )

    # Plot the results
    plots.show(
        initial_trap_trajectory=initial_trap_trajectory,
        trap_trajectory=trap_trajectory,
        anchor_currents=anchor_currents,
        wire_config=wire_config,
        bias_config=bias_config,
        schedule_fn=schedule_fn,
        losses=losses,
    )


if __name__ == "__main__":
    main()
