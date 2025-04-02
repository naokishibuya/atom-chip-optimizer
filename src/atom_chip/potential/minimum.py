from dataclasses import dataclass
from typing import Callable, Optional
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
from .hessian import hessian_at_minimum, Hessian


@dataclass
class MinimumResult:
    """
    Result of the minimum search.
    """

    found: bool
    value: jnp.ndarray  # Value of the function at the minimum
    point: jnp.ndarray  # Position of the minimum [x, y, z] in mm
    hessian: Optional[Hessian] = None  # Hessian at the minimum


def search_minimum(
    objective_function: Callable[[jnp.ndarray], float],
    hessian_step: float = 1e-5,
    verbose: bool = False,
    **kwargs: dict,
) -> MinimumResult:
    """
    Search for the minimum of a given function.

    Args:
        function (Callable): Function to be minimized. It should take a single point and return the function value.
        hessian_step (float): Step size for finite difference Hessian calculation.
        verbose (bool): If True, print additional information about the optimization process.
        kwargs (dict): Options for the minimization algorithm. It should contain:

    Returns:
        MinimumResult: Result of the minimum search.
    """

    # perform the minimization
    optres = minimize(objective_function, **kwargs)
    result = MinimumResult(
        found=optres.success,
        value=optres.fun,
        point=optres.x,
    )
    if not result.found:
        print("Optimization failed:", optres.message)
        return result

    # Compute the Hessian at the minimum
    if verbose:
        print("Minimum {:.10g} found at: x={:.10g} mm, y={:.10g} mm, z={:.10g} mm".format(result.value, *result.point))
    result.hessian = hessian_at_minimum(objective_function, result.point, hessian_step)

    # TODO
    hessian_fn = jax.hessian(objective_function)
    hessian_matrix = hessian_fn(result.point)
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian_matrix)
    print("====JAX Hessian====")
    print(eigenvalues)
    print(eigenvectors)
    # print(hessian_matrix)
    print("--------------------")

    if verbose:
        print("Hessian:", result.hessian.eigenvalues)
        print(result.hessian.eigenvectors)
    return result
