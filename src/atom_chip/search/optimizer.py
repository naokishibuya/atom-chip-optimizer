import logging
from dataclasses import dataclass
from typing import Callable
from scipy.optimize import minimize, NonlinearConstraint
import jax.numpy as jnp


@dataclass
class OptimizerResult:
    success: bool
    value: float
    params: jnp.ndarray
    message: str


def optimize(
    objective: Callable,
    constraints: list,
    lower_bounds: jnp.ndarray,
    upper_bounds: jnp.ndarray,
    params: jnp.ndarray,
    callback: Callable = None,
):
    nlc = NonlinearConstraint(
        fun=constraints,
        lb=lower_bounds,
        ub=upper_bounds,
    )

    result = minimize(
        fun=objective,
        x0=params,
        method="trust-constr",
        constraints=[nlc],
        callback=callback,
        options={"verbose": 3, "gtol": 1e-4},
    )

    if not result.success:
        logging.error("Optimization failed.")
        logging.error(result.message)

    return OptimizerResult(
        success=result.success,
        value=result.fun,
        params=result.x,
        message=result.message,
    )
