import numpy as np
import jax.numpy as jnp
from atom_chip.potential import search_minimum


def mock_function(points: jnp.ndarray) -> float:
    """
    Mock compute function for testing.
    """

    points = jnp.atleast_2d(points)
    value = points - jnp.array([1.0, 2.0, 3.0])
    return jnp.linalg.norm(value)


def test_min_search_simple():
    """
    Test with a simple function.
    Global minimum should be at z = 0.
    """

    # Start away from the minimum
    options = dict(
        x0=[0.0, 0.0, 1.0],
        method="Nelder-Mead",
        options={
            "xatol": 1e-10,
            "fatol": 1e-10,
            "maxiter": int(1e5),
            "maxfev": int(1e5),
            "disp": True,
        },
    )

    result = search_minimum(mock_function, **options)

    assert result.found, f"Search failed: {result.message}"

    # The minimum should be near z=0
    expected_point = jnp.array([1.0, 2.0, 3.0])
    expected_min_value = mock_function(expected_point)

    np.testing.assert_allclose(result.point, expected_point, atol=1e-4)
    np.testing.assert_almost_equal(result.value, expected_min_value, decimal=4)


def test_min_search_with_bounds():
    """
    Test optimization with bounds applied.
    """

    options = dict(
        x0=[0.0, 0.0, 5.0],
        bounds=[(-1, 1), (-1, 1), (2, 6)],
    )
    result = search_minimum(mock_function, **options)
    assert result.found, f"Search failed: {result.message}"

    # The minimum is outside the bounds
    expected_point = jnp.array([1.0, 1.0, 3.0])  # y is clipped to 1
    expected_min_value = mock_function(expected_point)

    np.testing.assert_allclose(result.point, expected_point, atol=1e-4)
    np.testing.assert_almost_equal(result.value, expected_min_value, decimal=4)


def test_min_search_with_options():
    """
    Test optimizer with different options.
    """

    options = dict(
        x0=[0.0, 0.0, 5.0],
        method="Nelder-Mead",
        options={
            "xatol": 1e-6,
            "fatol": 1e-6,
            "maxiter": 500,
            "disp": True,
        },
    )
    result = search_minimum(mock_function, **options)
    assert result.found, f"Search failed: {result.message}"

    # The global minimum should be found
    expected_point = jnp.array([1.0, 2.0, 3.0])
    expected_min_value = mock_function(expected_point)

    np.testing.assert_allclose(result.point, expected_point, atol=1e-4)
    np.testing.assert_almost_equal(result.value, expected_min_value, decimal=4)


def test_min_search_with_flat_landscape():
    """
    Test behavior when the value is zero everywhere.
    """

    def flat_compute(points: jnp.ndarray) -> float:
        """
        Mock compute with zero everywhere.
        """
        return jnp.zeros((1, 1))

    initial_guess = (0.0, 0.0, 3.0)
    options = dict(
        x0=initial_guess,
    )
    result = search_minimum(flat_compute, **options)
    assert result.found, f"Search failed: {result.message}"

    # With no magnetic field, the minimum should be the initial gues
    np.testing.assert_allclose(result.point, initial_guess, atol=1e-4)
    np.testing.assert_almost_equal(result.value, 0.0, decimal=4)
