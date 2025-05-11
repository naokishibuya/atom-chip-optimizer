"""
This script optimizes the current distribution in a quadrupole trap given a set of constraints.
"""

import logging
import os
from typing import Tuple
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
import jax
import jax.numpy as jnp
import optax
import atom_chip as ac


# Set up logging
logging.basicConfig(level=logging.INFO)


MATERIAL = "copper"


def get_matrial_color(current: float):
    return ac.visualization.layout_3d._get_material_color(MATERIAL, current)


def current_density(x: float, g: float, z0: float) -> jnp.ndarray:
    """
    Compute the current density profile j_y(x) for a quadrupole trap with an infinite flat sheet of current.
    (Solved with the calculus of variations)

    The trap is a 45°-rotated 2D quadrupole trap with the following properties:

    The field is zero at the trap center:

        B_x(x_0, z_0) = 0
        B_z(x_0, z_0) = 0

    The field gradient at the trap center is constant:

        dB_x/dx(x_0,z_0) = 0
        dB_x/dz(x_0,z_0) = g    where g is the field gradient [G/mm]

    The parameters are z_0 and g, and x_0 is assumed to be zero.

    These 4 constraints are satisfied by applying Biot-Savart's law to the infinite flat sheet of current in
    which the current flows in the positive/negative y-direction only, and solving the Euler-Lagrange equation
    with multiple Lagrange multipliers.

    Args:
        x: Position in x-axis.
        g: The field gradient (dBx/dz) at the center of the trap.
        z0: The height of the trap center above the wire.

    Returns:
        jnp.ndarray: Discretized current density profile.
    """
    return (2 * g * z0**3 / jnp.pi) * (3 * x**2 - z0**2) / (x**2 + z0**2) ** 2


def find_roots(func, x_min: float, x_max: float, num_points: int):
    """
    Find intervals where the function changes sign.

    Args:
        func: The function to evaluate.
        x_min: Minimum x value to search.
        x_max: Maximum x value to search.
        num_points: Number of points to sample in the range.

    Returns:
        list: List of tuples representing intervals where the function changes sign.
    """
    x_values = jnp.linspace(x_min, x_max, num_points)
    y_values = func(x_values)
    find_roots = []
    for i in range(len(x_values) - 1):
        if y_values[i] * y_values[i + 1] < 0:  # Sign change detected
            find_roots.append((x_values[i] + x_values[i + 1]) / 2)
    return find_roots


def integrate_current_density(
    func, x_min: float, x_max: float, num_points: int, roots: list
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Integrate the current density to find the total current within the brackets
    and find the mean x position of each bracket.
    Args:
        func: The function to evaluate.
        x_min: Minimum x value to integrate.
        x_max: Maximum x value to integrate.
        num_points: Number of points to sample in the range.
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Total current and mean x center position of each bracket.
    """
    points = [x_min, *roots, x_max]
    ranges = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    total_range = x_max - x_min
    currents = []
    centers = []
    for x_start, x_end in ranges:
        # Calculate the number of points in the current range
        num = int((x_end - x_start) / total_range * num_points)
        x_values = jnp.linspace(x_start, x_end, num)

        # Calculate the current density over the range
        density_values = func(x_values)

        # Integrate the current density over the range
        current = jax.scipy.integrate.trapezoid(density_values, x_values)
        currents.append(current)  # current density (A/m)

        # weighted average of x
        mean_x = jnp.sum(x_values * density_values) / jnp.sum(density_values)
        centers.append(mean_x)

        print(f"Range: ({x_start:8.2f}, {x_end:8.2f}) Current: {current:6.2f} Center: {mean_x:6.2f} with {num} points")
    return jnp.array(currents), jnp.array(centers)


def plot_current_density(func, g: float, z0: float, roots: list, currents: jnp.ndarray, centers: jnp.ndarray):
    """
    Plot the current density profile and mark the roots.
    Args:
        func: The function to evaluate.
        g: The field gradient (dBx/dz) at the center of the trap.
        z0: trap z value
        roots: List of roots to mark on the plot.
    """
    # Plot the current density profile with the roots
    x_values = jnp.linspace(-10, 10, 1000)
    j_y = func(x_values)

    fig = plt.figure(figsize=(7, 5))
    plt.title(f"Ideal 2D Quadrupole Trap (g={g} G/mm, $z_0$={z0} mm)\nLine Current Density Profile")
    plt.plot(x_values, j_y)
    plt.axhline(0, color="black", linestyle="--")
    plt.axvline(0, color="black", linestyle="--")
    for root in roots:
        plt.axvspan(root, root, color="red", alpha=0.3)
    # put shaded are between the x-axis and the line
    points = [x_values[0], *roots, x_values[-1]]
    ranges = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    for (x_start, x_end), current, center in zip(ranges, currents, centers):
        plt.fill_between(
            x_values,
            j_y,
            where=((x_start <= x_values) & (x_values <= x_end)),
            color=get_matrial_color(current),
            alpha=0.6,
        )
        plt.annotate(
            f"{current:.2f} A/m",
            xy=(center, func(center) / 2),
            xytext=(center, func(center) + 0.1),
            arrowprops=dict(arrowstyle="-", linestyle="dotted", lw=0.5),
            fontsize=8,
            ha="right" if center <= 0 else "left",
            va="top" if current < 0 else "bottom",
        )
    plt.xlabel("x' (mm)")
    plt.ylabel("$j_y$ (A/m)")  # It's a line density and not A/m^2
    plt.grid()
    return fig


def plot_field_magnitude(func, g: float, z0: float):
    """
    Plot the field magnitude profile.
    Args:
        func: The function to evaluate.
        g: The field gradient (dBx/dz) at the center of the trap.
        z0: trap z value
        num_points: Number of points to sample in the range.
    """
    # Compute B_x(z) from Biot–Savart integral at x = 0
    # B_x(0,z) = ∫ j_y(x') * 2z / (x'^2 + z^2) dx'
    # We'll evaluate this numerically for each z

    # Precompute j_y(x') over x' for integration
    x_prime = jnp.linspace(-100, 100, 1000)
    jy_vals = func(x_prime)

    # Define z range to evaluate B_x and its gradient
    z_min, z_max = z0 * 0.4, z0 * 1.6
    z_values = jnp.linspace(z_min, z_max, 500)  # Avoid z=0 to prevent singularity

    # Compute B_x(z) as an array
    Bx = jax.scipy.integrate.trapezoid(
        jy_vals * 2 * z_values[:, None] / (x_prime**2 + z_values[:, None] ** 2),
        x_prime,
        axis=1,
    )

    # Compute dB_x/dz using numerical differentiation
    dBx_dz = jnp.gradient(Bx, z_values)

    # Plot |B_x(z)| and dB_x/dz vs. z
    fig = plt.figure(figsize=(7, 5))
    plt.plot(z_values, jnp.abs(Bx), label=r"$|B_x(z)|$", color="blue")
    plt.plot(z_values, dBx_dz, label=r"$\frac{dB_x}{dz}$", color="green")
    plt.axhline(g, color="black", linestyle="--")
    plt.axvline(z0, color="black", linestyle="--")
    plt.grid(True)
    plt.xlim(z_min, z_max)
    plt.ylim(0, 2.5 * g)
    plt.xlabel("z [mm]")
    plt.ylabel(r"$|B_x|$ [G]")
    plt.legend()
    plt.gca().twinx().set_ylabel(r"$\frac{dB_x}{dz}$ [G/mm]")
    plt.title(f"Ideal 2D Quadrupole Trap (g={g} G/mm, $z_0$={z0} mm)\nMagnitude of $B_x$ and its Gradient vs $z$")
    plt.tight_layout()
    return fig


def adjust_plot_window(fig, top: int, left: int):
    window = fig.canvas.manager.window
    window.setWindowFlag(Qt.WindowStaysOnTopHint)
    geometry = window.geometry()
    window.setGeometry(left, top, geometry.width(), geometry.height())
    return window.geometry()


def calculate_quadrupole_currents(g: float, z0: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate the sheet currents for the quadrupole trap.

    Args:
        g: The field gradient (dBx/dz) at the center of the trap.
        z0: The height of the trap center above the wire.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Currents and centers of the quadrupole trap.
    """

    def func(x):
        return current_density(x, g=g, z0=z0)

    # Find the roots of the equation and integrate the current density with respect to x'
    roots = find_roots(func, x_min=-10.0, x_max=10.0, num_points=1000)
    currents, centers = integrate_current_density(func, x_min=-100.0, x_max=100.0, num_points=10000, roots=roots)

    # Plot the current density profile
    fig1 = plot_current_density(func, g=g, z0=z0, roots=roots, currents=currents, centers=centers)
    fig2 = plot_field_magnitude(func, g=g, z0=z0)

    geo1 = adjust_plot_window(fig1, top=100, left=100)
    adjust_plot_window(fig2, top=geo1.top(), left=geo1.left() + geo1.width() + 10)

    return currents, centers


# make atom chip for quadrupole trap
# fmt: off
def make_chip_q(
    currents    : jnp.ndarray,
    centers     : jnp.ndarray,
    length      : float,
    width       : float,
    height      : float,
) -> ac.AtomChip:
    # Create the atom chip wire segments
    components = []
    for current, center in zip(currents, centers):
        components.append(
            ac.components.RectangularConductor.create(
                material = MATERIAL,
                current  = current,
                segments = [
                    # [[start point], [end point], width, height]
                    [
                        jnp.array([center, -length / 2, 0]),
                        jnp.array([center,  length / 2, 0]),
                        width,
                        height,
                    ],
                ],
            )
        )

    return ac.AtomChip(
        name       = "Quadrupole",
        atom       = ac.rb87,
        components = components,
        bias_fields= ac.field.ZERO_BIAS_FIELD,
    )
# fmt: on


def optimize_params(make_chip, params, g, z0):
    def loss_fn(params):
        # Create the atom chip with the current configuration
        atom_chip = make_chip(params)

        # Field and grad functions
        def B_fn(r):
            _, B = atom_chip.get_fields(r[None])
            return B[0]

        # Compute the field and its gradient at the desired position
        r0 = jnp.array([0.0, 0.0, z0])  # Desired trap position
        B = B_fn(r0)
        grad_B_fn = jax.jacfwd(B_fn)
        grad_B = grad_B_fn(r0)

        # Compute the loss as the sum of the field and gradient terms
        field_term = jnp.sum(B[0] ** 2 + B[2] ** 2)
        grad_term = (grad_B[0, 0]) ** 2 + (grad_B[0, 2] - g) ** 2

        # Loss value
        λ1 = 0.7
        return field_term + λ1 * grad_term

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # Initialize the optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # optional
        optax.adamw(learning_rate=1e-4, weight_decay=1e-4),
    )
    opt_state = optimizer.init(params)

    # Perform optimization
    steps = 50000
    for step in range(1, steps + 1):
        loss, grads = loss_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if step % (steps // 10) == 0:
            print(f"Step {step}: Loss = {loss:.6g}")

    print(f"Optimized params: {params}")
    return params


def main():
    # Fix the field gradient (g=dBx/dz) and the trap position (z0)
    g = 4.0  # Gradient strength
    z0 = 1.0  # Distance from the center of the trap to the wire

    # Calculate the currents and centers for the quadrupole trap
    currents, centers = calculate_quadrupole_currents(g=g, z0=z0)

    # Build the atom chip with the initial currents and centers, and perform optimization
    length = 100.0  # mm
    width = jnp.array([5])  # mm
    height = 0.001  # mm

    def make_chip(params: jnp.array):
        centers, width = params[:3], params[3]
        return make_chip_q(currents, centers, length, width, height)

    # Optimize these parameters to replicate the desired field
    params = jnp.concatenate([centers, width])
    optimized_params = optimize_params(make_chip, params, g=g, z0=z0)

    # Build final chip
    atom_chip = make_chip(optimized_params)

    # fmt: off
    options = ac.potential.AnalysisOptions(
        search = dict(
            x0      = [0.0, 0.0, z0],  # Initial guess
            bounds  = [(-0.5, 0.5), (-0.5, 0.5), (0.0, 10.0)],
            method  = "Nelder-Mead",
            options = dict(
                xatol   = 1e-10,
                fatol   = 1e-10,
                maxiter = int(1e5),
                maxfev  = int(1e5),
                disp    = True,
            ),
        ),
        hessian = dict(
            method = "jax",
            # method = "finite-difference",
            # hessian_step = 1e-5,  # Step size for Hessian calculation
        ),
        # for the trap analayis (not used for field analysis)
        total_atoms=1e5,
        condensed_atoms=1e5,
    )
    # fmt: on

    # Analyze the atom chip
    analysis = atom_chip.analyze(options)

    # Export the atom chip to JSON
    directory = os.path.dirname(__file__)
    atom_chip.save(os.path.join(directory, "quadrupole.json"))

    # Perform the visualization
    ac.visualization.show(atom_chip, analysis, os.path.join(directory, "visualization.yaml"))


if __name__ == "__main__":
    main()
