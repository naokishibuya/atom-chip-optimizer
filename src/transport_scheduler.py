import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


# linear scheduler function
@jax.jit
def linear_schedule(s: float) -> jnp.ndarray:
    return s


# 1st and 2nd derivatives are 0 at s = 0 and s = 1
@jax.jit
def smoothstep_quintic(s: float) -> jnp.ndarray:
    return 10 * s**3 - 15 * s**4 + 6 * s**5  # C² at both ends


# Cosine schedule (range [0, 1] over finite time steps t over duration T)
@jax.jit
def cosine_schedule(s: float) -> jnp.ndarray:
    return 0.5 * (1 - jnp.cos(jnp.pi * s))


@jax.jit
def trapezoid_schedule(s: jnp.ndarray, alpha: float = 0.1) -> jnp.ndarray:
    """
    s     ∈ [0,1] scalar or array of normalized time steps n/T
    alpha ∈ [0,0.5] (T_accel/T)

    Returns f(s) ∈ [0,1] with a quadratic ramp-up, flat middle, quadratic ramp-down.
    """
    # ramp-up:   0 ≤ s < alpha
    up = 0.5 * (s / alpha) ** 2
    # middle:   alpha ≤ s < 1 - alpha
    mid = s - 0.5 * alpha
    # ramp-down: 1 - alpha ≤ s ≤ 1
    down = 1.0 - 0.5 * ((1.0 - s) / alpha) ** 2
    # piecewise combine
    return jnp.where(s < alpha, up, jnp.where(s < 1.0 - alpha, mid, down))


SCHEDULER_FUNCS = {
    "linear": linear_schedule,
    "smoothstep": smoothstep_quintic,
    "cosine": cosine_schedule,
    # "trapezoid": trapezoid_schedule,
}


def plot_schedules():
    # Parameters
    N = 2500
    distance = 2.4  # total displacement in mm
    distance = distance * 1e3  # convert from mm to micrometers

    # Compute normalized positions
    s = np.linspace(0, 1, N + 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_schedule(axes[0], s, **SCHEDULER_FUNCS)
    plot_schedule_delta_r(axes[1], s, distance, **SCHEDULER_FUNCS)
    fig.tight_layout()
    plt.show()


def plot_schedule(ax, s, **schedulers):
    for name, scheduler in schedulers.items():
        x = scheduler(s)
        ax.plot(s, x, label=name + " Schedule")
    ax.set_title("Transport Scheduler Comparison")
    ax.set_xlabel("Normalized Position (s)")
    ax.set_ylabel("Schedule Value")
    ax.legend(fontsize=8)


def plot_schedule_delta_r(ax, s, distance, **schedulers):
    for name, scheduler in schedulers.items():
        delta_r = jnp.diff(scheduler(s)) * distance
        ax.plot(s[:-1], delta_r, label=name + " Schedule")
    ax.set_title(r"$\Delta x = 2.4$ mm, $N = 2500$ steps")
    ax.set_xlabel("Step Index")
    ax.set_ylabel(r"$\mu$m / step")
    ax.legend(fontsize=8)


if __name__ == "__main__":
    plot_schedules()
