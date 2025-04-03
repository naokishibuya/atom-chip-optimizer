from dataclasses import dataclass
from typing import Callable
from scipy.optimize import minimize
import jax.numpy as jnp


@dataclass
class MinimumResult:
    """
    Result of the minimum search.
    """

    found: bool
    value: jnp.ndarray  # Value of the function at the minimum
    point: jnp.ndarray  # Position of the minimum [x, y, z] in mm


def search_minimum(function: Callable[[jnp.ndarray], float], **kwargs: dict) -> MinimumResult:
    """
    Search for the minimum of a given function.

    Args:
        function (Callable): Function to be minimized. It should take a single point and return the function value.
        kwargs (dict): Options for the minimization algorithm. It should contain:

    Returns:
        MinimumResult: Result of the minimum search.
    """

    # perform the minimization
    optres = minimize(function, **kwargs)
    result = MinimumResult(
        found=optres.success,
        value=optres.fun,
        point=optres.x,
    )
    if not result.found:
        print("Optimization failed:", optres.message)
    return result
