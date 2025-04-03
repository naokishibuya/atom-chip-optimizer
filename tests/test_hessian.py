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
    hessian_fd = hessian_at_minimum(mock_function1, position, method="finite-difference", step=1e-5)

    print()
    print("-" * 100)
    print("Hessian (Finite Difference):")
    print(hessian_fd.matrix)
    print(hessian_fd.eigenvalues)
    print(hessian_fd.eigenvectors)

    hessian_jx = hessian_at_minimum(mock_function1, position, method="jax")

    print("-" * 100)
    print("Hessian (JAX):")
    print(hessian_jx.matrix)
    print(hessian_jx.eigenvalues)
    print(hessian_jx.eigenvectors)

    # Expected eigenvalues and eigenvectors
    expected_eigvalues, expected_eigvectors, expected_hessian_matrix = evaluate_hessian(mock_function1, position)

    print("-" * 100)
    print("Expected Hessian:")
    print(expected_hessian_matrix)
    print(expected_eigvalues)
    print(expected_eigvectors)

    # Compare the results
    np.testing.assert_allclose(hessian_fd.matrix, expected_hessian_matrix, atol=1e-4)
    np.testing.assert_allclose(hessian_fd.eigenvalues, expected_eigvalues, atol=1e-4)
    np.testing.assert_allclose(hessian_jx.matrix, expected_hessian_matrix, atol=1e-4)
    np.testing.assert_allclose(hessian_jx.eigenvalues, expected_eigvalues, atol=1e-4)

    # Compare eigenvectors
    compare_eigenvectors(hessian_fd.eigenvectors, expected_eigvectors, atol=1e-4)
    compare_eigenvectors(hessian_jx.eigenvectors, expected_eigvectors, atol=1e-4)


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
    hessian_fd = hessian_at_minimum(mock_function2, position, method="finite-difference", step=1e-5)

    print()
    print("-" * 100)
    print("Hessian Matrix (Finite Difference):")
    print(hessian_fd.matrix)
    print(hessian_fd.eigenvalues)
    print(hessian_fd.eigenvectors)

    hessian_jx = hessian_at_minimum(mock_function2, position, method="jax")

    print("-" * 100)
    print("Hessian Matrix (JAX):")
    print(hessian_jx.matrix)
    print(hessian_jx.eigenvalues)
    print(hessian_jx.eigenvectors)

    # Expected eigenvalues and eigenvectors
    expected_eigvalues, expected_eigvectors, expected_hessian_matrix = evaluate_hessian(mock_function2, position)

    print("-" * 100)
    print("Expected Hessian:")
    print(expected_hessian_matrix)
    print(expected_eigvalues)
    print(expected_eigvectors)

    # Compare the results
    np.testing.assert_allclose(hessian_fd.matrix, expected_hessian_matrix, atol=1e-4)
    np.testing.assert_allclose(hessian_fd.eigenvalues, expected_eigvalues, atol=1e-4)
    np.testing.assert_allclose(hessian_jx.matrix, expected_hessian_matrix, atol=1e-4)
    np.testing.assert_allclose(hessian_jx.eigenvalues, expected_eigvalues, atol=1e-4)

    # Compare eigenvectors
    compare_eigenvectors(hessian_fd.eigenvectors, expected_eigvectors, atol=1e-4)
    compare_eigenvectors(hessian_jx.eigenvectors, expected_eigvectors, atol=1e-4)
