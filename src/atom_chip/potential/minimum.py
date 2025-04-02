from dataclasses import dataclass
from typing import Callable, Optional
from scipy.optimize import minimize
import numpy as np
from numpy.typing import ArrayLike
from .hessian import hessian_at_minimum


@dataclass
class Hessian:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    matrix: np.ndarray


@dataclass
class PotentialMinimum:
    """
    Result of the minimum search.
    """

    found: bool
    value: np.ndarray  # Value of the function at the minimum
    point: np.ndarray  # Position of the minimum [x, y, z] in mm
    hessian: Optional[Hessian] = None  # Hessian at the minimum


def search_minimum_potential(function: Callable[[np.ndarray], np.ndarray], **kwargs) -> PotentialMinimum:
    """
    Search for the minimum of a potential function.

    Args:
        function (Callable): Function to be minimized. It should take a single point and return the function value.
        **kwargs: Additional arguments for the minimization function.

    Returns:
        PotentialMinimum: Result of the minimum search.
    """

    def objective_function(point: ArrayLike) -> float:
        """
        Objective function to minimize. It takes a single point and returns the function value.
        """
        point = np.float64(point).reshape(1, 3)  # make it a single entry 2D array
        value = function(point)[0]  # get the first returned value
        return value  # get the first element of the single entry 2D array

    optres = minimize(objective_function, **kwargs)
    result = PotentialMinimum(
        found=optres.success,
        value=optres.fun,
        point=optres.x,
    )
    print(result.point)
    if result.found:
        # Compute the Hessian at the minimum
        eigenvalues, eigvectors, hessian_matrix = hessian_at_minimum(objective_function, result.point, step=1e-6)
        result.hessian = Hessian(
            eigenvalues=eigenvalues,
            eigenvectors=eigvectors,
            matrix=hessian_matrix,
        )
    return result
