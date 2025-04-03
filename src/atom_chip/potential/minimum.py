from dataclasses import dataclass
from typing import Callable
from scipy.optimize import minimize
import jax.numpy as jnp
from jax.tree_util import register_dataclass


@register_dataclass
@dataclass
class MinimumResult:
    """
    Result of the minimum search.
    """

    found: bool
    value: jnp.ndarray  # Value of the function at the minimum
    position: jnp.ndarray  # Position of the minimum [x, y, z] in mm
    message: str = ""


def search_minimum(objective: Callable[[jnp.ndarray], float], **kwargs: dict) -> MinimumResult:
    """
    Search for the minimum of a given function.

    Args:
        function (Callable): Function to be minimized. It should take a single point and return the function value.
        kwargs (dict): Options for the minimization algorithm. It should contain:

    Returns:
        MinimumResult: Result of the minimum search.
    """

    # perform the minimization
    optres = minimize(objective, **kwargs)

    return MinimumResult(
        found=optres.success,
        value=optres.fun,
        position=optres.x,
        message=optres.message,
    )
