"""
This script optimizes the current distribution in a quadrupole trap given a set of constraints.
"""

import logging
import os
from typing import Tuple
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import atom_chip as ac


# Set up logging
logging.basicConfig(level=logging.INFO)


def current_density(x: float, g: float, z0: float) -> jnp.ndarray:
    """
    Compute the current density profile j_y(x) for a quadrupole trap with an infinite flat sheet of current.
    (Solved with the calculus of variations)

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


def plot_current_density(func, x_min: float, x_max: float, num_points: int, roots: list):
    """
    Plot the current density profile and mark the roots.
    Args:
        func: The function to evaluate.
        x_min: Minimum x value to plot.
        x_max: Maximum x value to plot.
        num_points: Number of points to sample in the range.
        roots: List of roots to mark on the plot.
    """
    # Plot the current density profile with the roots
    x_values = jnp.linspace(x_min, x_max, num_points)
    j_y = func(x_values)

    plt.figure(figsize=(10, 6))
    plt.title("Current Density Profile")
    plt.plot(x_values, j_y)
    plt.axhline(0, color="black", linestyle="--")
    plt.axvline(0, color="black", linestyle="--")
    for root in roots:
        plt.axvspan(root, root, color="red", alpha=0.3)
    plt.xlabel("x' (mm)")
    plt.ylabel("j_y (A/m)")  # It's a line density and not A/m^2
    plt.grid()


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
    plt.figure(figsize=(10, 6))
    plt.plot(z_values, jnp.abs(Bx), label=r"$|B_x(z)|$", color="green")
    plt.plot(z_values, dBx_dz, label=r"$\frac{dB_x}{dz}$", color="red")
    plt.axhline(g, color="black", linestyle="--")
    plt.axvline(z0, color="black", linestyle="--")
    plt.grid(True)
    plt.xlim(z_min, z_max)
    plt.ylim(0, 2.5 * g)
    plt.xlabel("z [mm]")
    plt.ylabel(r"$|B_x|$ [G]")
    plt.legend()
    plt.gca().twinx().set_ylabel(r"$\frac{dB_x}{dz}$ [G/mm]")
    plt.title("Magnitude of $B_x$ and its Gradient vs $z$")
    plt.tight_layout()


def integrate_current_density(
    func, x_min: float, x_max: float, num_points: int, roots
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


def calculate_quadrupole_currents(g: float, z0: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate the currents for the quadrupole trap.

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
    print(f"Roots where current crosses zero: {roots}")

    # Plot the current density profile
    plot_current_density(func, x_min=-20, x_max=20, num_points=100, roots=roots)
    plot_field_magnitude(func, g=g, z0=z0)

    currents, centers = integrate_current_density(func, x_min=-100.0, x_max=100.0, num_points=10000, roots=roots)
    return currents, centers


# make atom chip for quadrupole trap
# fmt: off
def make_chip_q(
    currents    : jnp.ndarray,
    centers     : jnp.ndarray,
    length      : float,
    width       : float,
    height      : float,
    bias_fields : ac.field.BiasFields = ac.field.ZERO_BIAS_FIELD,
) -> ac.AtomChip:
    components = []
    for current, center in zip(currents, centers):
        components.append(
            ac.components.RectangularConductor.create(
                material = "gold",
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
        bias_fields= bias_fields,
    )
# fmt: on


def evaluate_field_and_gradient(atom_chip, r0: jnp.ndarray):
    """
    Compute the magnetic field and its spatial gradient at a given point.

    Args:
        atom_chip: An AtomChip object.
        r0: 3D position to evaluate the field [x, y, z] (in mm).

    Returns:
        Tuple of (B_field: [3], grad_B: [3,3])
    """
    r0 = jnp.array(r0)

    def B_fn(r):
        _, B = atom_chip.get_fields(r[None])
        return B[0]

    grad_B_fn = jax.jacfwd(B_fn)
    B = B_fn(r0)
    grad_B = grad_B_fn(r0)

    print(f"r0: {r0}")
    print(f"Field at r0: B = {B}")
    print(f"Gradient at r0:\n{grad_B}")

    return B, grad_B


def optimize_currents(initial_currents, centers, length, width, height, g, z0, steps, lr):
    def make_chip_from_currents(currents):
        return make_chip_q(currents=currents, centers=centers, length=length, width=width, height=height)

    @jax.jit
    def loss_fn(currents):
        chip = make_chip_from_currents(currents)
        r0 = jnp.array([0.0, 0.0, z0])  # Desired trap position

        def B_fn(r):
            _, B = chip.get_fields(r[None])
            return B[0]

        grad_B_fn = jax.jacfwd(B_fn)
        B = B_fn(r0)
        grad_B = grad_B_fn(r0)

        field_term = jnp.sum(B**2)
        grad_term = (grad_B[0, 0]) ** 2 + (grad_B[0, 2] - g) ** 2

        λ = 1.0
        return field_term + λ * grad_term

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(initial_currents)

    params = initial_currents

    for step in range(steps):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss:.6g}")
    return params


def main():
    # Fix the field gradient (g=dBx/dz) and the trap position (z0)
    g = 3.0  # Gradient strength
    z0 = 1.0  # Distance from the center of the trap to the wire

    # Calculate the currents and centers for the quadrupole trap
    currents, centers = calculate_quadrupole_currents(g=g, z0=z0)

    # Build the atom chip
    length = 100.0  # mm
    width = 5.0  # mm
    height = 0.01  # mm

    optimized_currents = optimize_currents(currents, centers, length, width, height, g, z0, steps=5000, lr=0.01)

    print(f"Optimized currents: {optimized_currents}")

    # Build final chip
    bias_fields = ac.field.BiasFields(
        coil_factors=jnp.array([0.0, 0.0, 0.0]),
        currents=jnp.array([0.0, 0.0, 0.0]),
        stray_fields=jnp.array([0.5, 0.5, 0.0]),
    )
    atom_chip = make_chip_q(optimized_currents, centers, length, width, height, bias_fields)

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
