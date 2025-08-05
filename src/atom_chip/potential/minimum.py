from dataclasses import dataclass
from typing import Callable
from scipy.optimize import minimize
import jax
import jax.numpy as jnp


@dataclass(frozen=True)  # immutable = safer
class MinimumResult:
    """
    Result of the minimum search.
    """

    found: bool
    value: jnp.ndarray  # Value of the function at the minimum
    position: jnp.ndarray  # Position of the minimum [x, y, z] in mm
    message: str = ""

    # -- PyTree flatten/unflatten ------------------------------------
    def tree_flatten(self):
        # children that will travel through JIT/grad
        dynamic_children = (self.found, self.value, self.position)
        # aux / static data (ignored by JAX’s AD & device transfer)
        aux_data = dict(message=self.message)
        return dynamic_children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        found, value, position = children
        return cls(found=found, value=value, position=position, message=aux_data["message"])


# Register once; after this JAX recognises MinimumResult everywhere
jax.tree_util.register_pytree_node(
    MinimumResult,
    MinimumResult.tree_flatten,
    MinimumResult.tree_unflatten,
)


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
    result = minimize(objective, **kwargs)

    return MinimumResult(
        found=result.success,
        value=result.fun,
        position=result.x,
        message=result.message,
    )
