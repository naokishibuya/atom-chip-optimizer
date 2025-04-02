from dataclasses import dataclass
from typing import Callable, Optional
from scipy.optimize import minimize
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


def search_minimum(objective_function: Callable[[jnp.ndarray], float], options: dict) -> MinimumResult:
    """
    Search for the minimum of a given function.

    Args:
        function (Callable): Function to be minimized. It should take a single point and return the function value.
        options (dict): Options for the minimization algorithm. It should contain:

    The options may include the below parameters that are not passed to the scipy minimize function:
        hessian_step (float): Step size for finite difference Hessian calculation.
        verbose (bool): If True, print additional information about the optimization process.

    Returns:
        MinimumResult: Result of the minimum search.
    """

    # remove hessian step from options if available
    hessian_step = options.pop("hessian_step", 1e-5)
    verbose = options.pop("verbose", False)

    # perform the minimization
    optres = minimize(objective_function, **options)
    result = MinimumResult(
        found=optres.success,
        value=optres.fun,
        point=optres.x,
    )
    if not result.found:
        return result

    # Compute the Hessian at the minimum
    if verbose:
        print("Minimum {:.10g} found at: x={:.10g} mm, y={:.10g} mm, z={:.10g} mm".format(result.value, *result.point))
    result.hessian = hessian_at_minimum(objective_function, result.point, hessian_step)
    if verbose:
        print("Hessian:", result.hessian.eigenvalues)
        print(result.hessian.eigenvectors)
    return result
