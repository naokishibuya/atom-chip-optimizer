# Test hessian computation
#
# The Hessian matrix is a square matrix of second-order partial derivatives of a scalar function.
# It is used to determine the local curvature of the function at a given point.
#
# In this test, we will compute the Hessian matrix of a simple function and compare it to the expected values.

import numpy as np
import jax
import jax.numpy as jnp
from atom_chip.potential import hessian_at_minimum


# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)


def mock_function1(points: jnp.ndarray) -> jnp.ndarray:
    """
    Mock compute function for testing.
    """

    points = jnp.atleast_2d(points)
    x, y, z = points[0]  # one point only
    value = (2 * x - 1.0) ** 2 + (3 * y - 2.0) ** 2 + (z - 3.0) ** 2
    return value


def evaluate_hessian(function, point: jnp.ndarray) -> jnp.ndarray:
    """JAX-based Hessian evaluation using automatic differentiation"""
    hessian_fn = jax.hessian(function)
    hessian_matrix = hessian_fn(point)
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian_matrix)
    return eigenvalues, eigenvectors, hessian_matrix


def test_compute_hessian1():
    """
    Test Hessian computation.
    """
    # Compute Hessian matrix
    position = jnp.float64([1.0, 2.0, 3.0])
    hessian = hessian_at_minimum(mock_function1, position, step=1e-5)

    print()
    print("-" * 100)
    print("Hessian Matrix:")
    print(hessian.matrix)
    print("Eigenvalues:")
    print(hessian.eigenvalues)
    print("Eigenvectors:")
    print(hessian.eigenvectors)

    # Expected eigenvalues and eigenvectors
    expected_eigvalues, expected_eigvectors, expected_hessian_matrix = evaluate_hessian(mock_function1, position)
    print(expected_hessian_matrix.shape)

    print("-" * 100)
    print("Expected Hessian Matrix:")
    print(expected_hessian_matrix)
    print("Expected Eigenvalues:")
    print(expected_eigvalues)
    print("Expected Eigenvectors:")
    print(expected_eigvectors)

    # Compare the results
    np.testing.assert_allclose(hessian.matrix, expected_hessian_matrix, atol=1e-4)
    np.testing.assert_allclose(hessian.eigenvalues, expected_eigvalues, atol=1e-4)
    compare_eigenvectors(hessian.eigenvectors, expected_eigvectors, atol=1e-4)


def compare_eigenvectors(computed, expected, atol):
    for v1, v2 in zip(computed.T, expected.T):
        assert np.allclose(np.abs(v1), np.abs(v2), atol=atol) or np.allclose(np.abs(v1), np.abs(-v2), atol=atol), (
            f"Eigenvector mismatch: {v1} vs {v2}"
        )


def mock_function2(points: np.ndarray) -> np.ndarray:
    """
    Mock compute function for testing.
    """
    points = jnp.atleast_2d(points)
    x, y, z = points[0]  # one point only
    value = 4 * x**2 + 3 * y**2 + 2 * z**2
    return value


def test_compute_hessian2():
    """
    Test Hessian computation.
    """
    # Compute Hessian matrix
    position = jnp.array([1.0, 2.0, 3.0])
    hessian = hessian_at_minimum(mock_function2, position, step=1e-5)

    print()
    print("-" * 100)
    print("Hessian Matrix:")
    print(hessian.matrix)
    print("Eigenvalues:")
    print(hessian.eigenvalues)
    print("Eigenvectors:")
    print(hessian.eigenvectors)

    # Expected eigenvalues and eigenvectors
    expected_eigvalues, expected_eigvectors, expected_hessian_matrix = evaluate_hessian(mock_function2, position)

    print("-" * 100)
    print("Expected Hessian Matrix:")
    print(expected_hessian_matrix)
    print("Expected Eigenvalues:")
    print(expected_eigvalues)
    print("Expected Eigenvectors:")
    print(expected_eigvectors)

    # Compare the results
    np.testing.assert_allclose(hessian.matrix, expected_hessian_matrix, atol=1e-4)
    np.testing.assert_allclose(hessian.eigenvalues, expected_eigvalues, atol=1e-4)
    compare_eigenvectors(hessian.eigenvectors, expected_eigvectors, atol=1e-4)
