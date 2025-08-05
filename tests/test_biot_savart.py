import numpy as np
from atom_chip.field.biot_savart import biot_savart_rectangular, _compute_BY, _compute_BZ


def test_compute_BY_BZ():
    # Define wire segment
    L, W, H = 1.0, 0.1, 0.1  # Half-length, half-width, half-height
    x, y, z = 0.0, 0.0, 1.0  # Evaluation point on the z-axis

    # Compute BY and BZ
    BY = _compute_BY(x, y, z, L, W, H)
    BZ = _compute_BZ(x, y, z, L, W, H)

    print(f"BY: {BY}, BZ: {BZ}")
    print(BY.dtype, BZ.dtype)

    # Expected result (BY should be non-zero, BZ should be zero)
    assert abs(BY) > 1e-10, "BY should be non-zero"
    assert abs(BZ) < 1e-10, "BZ should be approximately zero"


def test_biot_savart_rectangular():
    # Test case: single wire segment
    # Define a simple wire along the x-axis
    starts = np.array([[0, 0, 0], [1, 1, 1]])
    ends = np.array([[1, 0, 0], [3, 1, 1]])
    widths = np.array([0.01, 0.02])
    heights = np.array([0.01, 0.02])
    current = np.array([1, 1])

    # Evaluation point above the wire
    # fmt: off
    eval_points = np.array([
        [ 0.5,  1.0, 0.0],
        [ 0.5, -1.0, 0.0],
        [-0.5,  1.0, 0.0],
        [-0.5, -1.0, 0.0],
        [ 0.5,  1.0, 1.0],
        [ 0.5, -1.0, 1.0],
        [-0.5,  1.0, 1.0],
        [-0.5, -1.0, 1.0],
        [ 1.5,  1.0, 1.0],
        [-1.5,  1.0, 1.0],
        [ 0.8,  1.0, 0.5],
        [ 0.8,  0.8, 0.8],
    ])
    # fmt: on

    # Compute the magnetic field
    field = biot_savart_rectangular(eval_points, starts, ends, widths, heights, current)

    print("Magnetic Field at Evaluation Point:")
    print(field)

    # Expected field
    # fmt: off
    expected = np.array([
        [0,  0.48124966696468,  0.89443434618299],
        [0,  0.10542751902487, -1.10528938429221],
        [0,  0.12946998850850,  0.38483410574970],
        [0,  0.05712260308071, -0.49907931192259],
        [0, -0.33333497946791,  0.33333497946791],
        [0, -0.33333497946485, -0.60250097209597],
        [0, -0.19713645137614,  0.19713645137614],
        [0, -0.19713645137825, -0.33125681523473],
        [0, -0.19713645137656,  0.19713645137656],
        [0, -0.07139035865133,  0.07139035865133],
        [0,  0.90417135895072,  0.60640700332542],
        [0,  0.56617022974114, -0.56617022974114],
    ])
    # fmt: on

    # Check if the field matches the expected field
    np.testing.assert_allclose(field, expected, rtol=1e-9)


def test_biot_savart_rectangular_with_rotation():
    # Test case: single wire segment at an angle
    starts = np.array([[0, 0, 0]])
    ends = np.array([[1, 1, 1]])
    widths = np.array([0.01])
    heights = np.array([0.01])
    currents = np.array([1])

    # Evaluation points (all outside the wire)
    eval_points = np.array(
        [
            [-0.1, 0.5, 0.5],  # point near the wire
            [0, 1, 0],  # point off the wire
            [1, 0, 1],  # another point off the wire
            [0.5, 0.5, 0.6],  # point above the wire
            [1.1, 1.1, 1.1],  # point beyond the end of the wire
        ]
    )

    # Calculate the magnetic field
    B_field = biot_savart_rectangular(eval_points, starts, ends, widths, heights, currents)

    # Assertions
    assert B_field.shape == (5, 3), "Output shape should be (5, 3)"
    assert not np.any(np.isnan(B_field)), "Output should not contain NaN values"
    assert not np.any(np.isinf(B_field)), "Output should not contain infinite values"

    # Check if the field is perpendicular to the wire direction
    wire_direction = ends[0] - starts[0]
    dot_products = np.abs(np.dot(B_field, wire_direction))
    assert np.all(dot_products < 1e-10), "Field should be perpendicular to wire direction"

    # Check expected values
    # fmt: off
    expected = [
        [-8.283559385299788e-09, -2.388481996642470e00 ,  2.388482004926028e00 ],
        [-1.207112421019265e00 , -6.193046157279211e-10,  1.207112421638569e00 ],
        [ 1.207112421009654e00 ,  6.246096169578927e-10, -1.207112421634263e00 ],
        [ 1.724296389440750e01 , -1.724296389440801e01 ,  5.099024833164444e-13],
        [-1.321515260168412e-14, -9.488052682171205e-16,  1.416395786990123e-14],
    ]
    # fmt: on
    np.testing.assert_allclose(B_field, expected, rtol=1e-6, atol=1e-10)
