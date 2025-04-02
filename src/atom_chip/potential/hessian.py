from dataclasses import dataclass
from typing import Callable
import jax.numpy as jnp


@dataclass
class Hessian:
    eigenvalues: jnp.ndarray
    eigenvectors: jnp.ndarray
    matrix: jnp.ndarray


def hessian_at_minimum(
    function: Callable[[jnp.ndarray], float],
    position: jnp.ndarray,
    step: float,  # finite difference step size
) -> Hessian:
    """
    Compute the Hessian matrix of a function at a given position using finite differences.

    Args:
        position: the function minimum position at which to compute the Hessian.
        function (Callable): Function to be computed and differentiated.
        step (float): Step size for finite differences.

    Returns:
        [Hessian eigenvectors, Hessian eigenvalues, Hessian matrix]
    """

    x0, y0, z0 = position

    def F(dx, dy, dz):
        p = jnp.float64([x0 + dx, y0 + dy, z0 + dz]).reshape(1, 3)  # make it a 2D array of one row
        return function(p)

    # Compute function at the given position
    f0 = F(0, 0, 0)

    # The second derivative ∂²f/∂x² is approximated by:
    # ∂²f/∂x² ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
    # ∂²f/∂y² ≈ [f(y+h) - 2f(y) + f(y-h)] / h²
    # ∂²f/∂z² ≈ [f(z+h) - 2f(z) + f(z-h)] / h²

    # Compute second derivatives (diagonal elements of the Hessian matrix)
    h = step
    denom = step**2
    d2fxx = (F(h, 0, 0) - 2 * f0 + F(-h, 0, 0)) / denom
    d2fyy = (F(0, h, 0) - 2 * f0 + F(0, -h, 0)) / denom
    d2fzz = (F(0, 0, h) - 2 * f0 + F(0, 0, -h)) / denom

    # Compute second derivatives (off-diagonal)
    # The mixed partial derivative ∂²f/∂x∂y is approximated by:
    # ∂²f/∂x∂y ≈ [f(x+h,y+h) - f(x+h,y-h) - f(x-h,y+h) + f(x-h,y-h)] / (4h²)
    # ∂²f/∂x∂z ≈ [f(x+h,z+h) - f(x+h,z-h) - f(x-h,z+h) + f(x-h,z-h)] / (4h²)
    # ∂²f/∂y∂z ≈ [f(y+h,z+h) - f(y+h,z-h) - f(y-h,z+h) + f(y-h,z-h)] / (4h²)

    # Compute mixed second derivatives (off-diagonal elements of the Hessian matrix)
    denom = 4 * step**2
    d2fxy = (F(h, h, 0) - F(h, -h, 0) - F(-h, h, 0) + F(-h, -h, 0)) / denom
    d2fxz = (F(h, 0, h) - F(h, 0, -h) - F(-h, 0, h) + F(-h, 0, -h)) / denom
    d2fyz = (F(0, h, h) - F(0, h, -h) - F(0, -h, h) + F(0, -h, -h)) / denom

    # Construct Hessian matrix (symmetric)
    # fmt: off
    hessian_matrix = jnp.float64([
        [d2fxx, d2fxy, d2fxz],
        [d2fxy, d2fyy, d2fyz],
        [d2fxz, d2fyz, d2fzz],
    ])
    # fmt: on

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian_matrix)

    return Hessian(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        matrix=hessian_matrix,
    )
