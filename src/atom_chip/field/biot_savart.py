import jax
import jax.numpy as jnp


# fmt: off
@jax.jit
def biot_savart_rectangular(
    points  : jnp.ndarray,  # (M, 3) Evaluation points in 3D space [mm]
    starts  : jnp.ndarray,  # (N, 3) Start points of wire segments [mm]
    ends    : jnp.ndarray,  # (N, 3) End points of wire segments [mm]
    widths  : jnp.ndarray,  # (N,  ) Widths of wire segments [mm]
    heights : jnp.ndarray,  # (N,  ) Heights of wire segments [mm]
    currents: jnp.ndarray,  # (N,  ) Currents through wire segments [A]
) -> jnp.ndarray:
# fmt: on
    """
    Compute the magnetic field from multiple rectangular conductor segments at multiple points using JAX.

    Args:
        points: List of evaluation points in 3D space.
        wires: List of rectangular conductor segments.

    Returns:
        B_total: (M, 3) Total magnetic field at each evaluation point [G].
    """

    # Ensure float64 for precision
    # fmt: off
    points   = jnp.float64(points)
    starts   = jnp.float64(starts)
    ends     = jnp.float64(ends)
    widths   = jnp.float64(widths)
    heights  = jnp.float64(heights)
    currents = jnp.float64(currents)
    # fmt: on

    # Compute vectors and lengths
    vectors = ends - starts  # (N, 3)
    lengths = jnp.linalg.norm(vectors, axis=1)  # (N,)

    # Translate points
    midpoints = (starts + ends) / 2  # (N, 3)
    translated_points = points[:, None, :] - midpoints[None, :, :]  # (M, N, 3)

    # Rotate points
    rotation_matrices = _rotation_matrix(vectors)  # (N, 3, 3)
    rotated_points = jnp.einsum("nij,mnj->mni", rotation_matrices, translated_points)  # (M, N, 3)

    # Compute field components
    x, y, z = rotated_points[:, :, 0], rotated_points[:, :, 1], rotated_points[:, :, 2]  # (M, N)
    L, W, H = lengths / 2, widths / 2, heights / 2  # (N,)

    BY = _compute_BY(x, y, z, L, W, H)  # (M, N)
    BZ = _compute_BZ(x, y, z, L, W, H)  # (M, N)
    BX = jnp.zeros_like(BY)  # (M, N)

    # Rotate field components back
    B_rotated = jnp.stack([BX, BY, BZ], axis=-1)  # (M, N, 3)
    B = jnp.einsum("nij,mnj->mni", jnp.transpose(rotation_matrices, (0, 2, 1)), B_rotated)  # (M, N, 3)

    # Scale by current and conductor cross-section
    B_scaled = B * currents[None, :, None] / (widths * heights)[None, :, None]  # (M, N, 3)

    # Sum contributions from all conductors
    B_total = jnp.sum(B_scaled, axis=1)  # (M, 3)

    return B_total


def _rotation_matrix(vectors: jnp.ndarray) -> jnp.ndarray:
    """Compute rotation matrices."""
    vectors = vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)
    alpha = jnp.arctan2(vectors[:, 1], vectors[:, 0])
    beta = jnp.arctan2(vectors[:, 2], jnp.linalg.norm(vectors[:, :2], axis=1))

    cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
    cos_b, sin_b = jnp.cos(beta), jnp.sin(beta)
    zeros = jnp.zeros_like(alpha, dtype=jnp.float64)

    rotation_matrices = jnp.array([
        [cos_a * cos_b, sin_a * cos_b, sin_b],
        [-sin_a, cos_a, zeros],
        [-cos_a * sin_b, -sin_a * sin_b, cos_b]
    ]).transpose(2, 0, 1)
    return rotation_matrices


def _compute_BY(x, y, z, L, W, H):
    # Vectorized computation of BY component
    return -(
        -(x - L) * jnp.log(y + W + jnp.sqrt((y + W) ** 2 + (z - H) ** 2 + (x - L) ** 2))
        + (x + L) * jnp.log(y + W + jnp.sqrt((y + W) ** 2 + (z - H) ** 2 + (x + L) ** 2))
        + (x - L) * jnp.log(y - W + jnp.sqrt((y - W) ** 2 + (z - H) ** 2 + (x - L) ** 2))
        - (x + L) * jnp.log(y - W + jnp.sqrt((y - W) ** 2 + (z - H) ** 2 + (x + L) ** 2))
        + (x - L) * jnp.log(y + W + jnp.sqrt((y + W) ** 2 + (z + H) ** 2 + (x - L) ** 2))
        - (x + L) * jnp.log(y + W + jnp.sqrt((y + W) ** 2 + (z + H) ** 2 + (x + L) ** 2))
        - (x - L) * jnp.log(y - W + jnp.sqrt((y - W) ** 2 + (z + H) ** 2 + (x - L) ** 2))
        + (x + L) * jnp.log(y - W + jnp.sqrt((y - W) ** 2 + (z + H) ** 2 + (x + L) ** 2))
        - (y + W) * jnp.log(x - L + jnp.sqrt((y + W) ** 2 + (z - H) ** 2 + (x - L) ** 2))
        + (y + W) * jnp.log(x + L + jnp.sqrt((y + W) ** 2 + (z - H) ** 2 + (x + L) ** 2))
        + (y - W) * jnp.log(x - L + jnp.sqrt((y - W) ** 2 + (z - H) ** 2 + (x - L) ** 2))
        - (y - W) * jnp.log(x + L + jnp.sqrt((y - W) ** 2 + (z - H) ** 2 + (x + L) ** 2))
        + (y + W) * jnp.log(x - L + jnp.sqrt((y + W) ** 2 + (z + H) ** 2 + (x - L) ** 2))
        - (y + W) * jnp.log(x + L + jnp.sqrt((y + W) ** 2 + (z + H) ** 2 + (x + L) ** 2))
        - (y - W) * jnp.log(x - L + jnp.sqrt((y - W) ** 2 + (z + H) ** 2 + (x - L) ** 2))
        + (y - W) * jnp.log(x + L + jnp.sqrt((y - W) ** 2 + (z + H) ** 2 + (x + L) ** 2))
        + (z - H) * jnp.arctan((y + W) * (x - L) / ((z - H) * jnp.sqrt((y + W) ** 2 + (z - H) ** 2 + (x - L) ** 2)))
        - (z - H) * jnp.arctan((y + W) * (x + L) / ((z - H) * jnp.sqrt((y + W) ** 2 + (z - H) ** 2 + (x + L) ** 2)))
        - (z - H) * jnp.arctan((y - W) * (x - L) / ((z - H) * jnp.sqrt((y - W) ** 2 + (z - H) ** 2 + (x - L) ** 2)))
        + (z - H) * jnp.arctan((y - W) * (x + L) / ((z - H) * jnp.sqrt((y - W) ** 2 + (z - H) ** 2 + (x + L) ** 2)))
        - (z + H) * jnp.arctan((y + W) * (x - L) / ((z + H) * jnp.sqrt((y + W) ** 2 + (z + H) ** 2 + (x - L) ** 2)))
        + (z + H) * jnp.arctan((y + W) * (x + L) / ((z + H) * jnp.sqrt((y + W) ** 2 + (z + H) ** 2 + (x + L) ** 2)))
        + (z + H) * jnp.arctan((y - W) * (x - L) / ((z + H) * jnp.sqrt((y - W) ** 2 + (z + H) ** 2 + (x - L) ** 2)))
        - (z + H) * jnp.arctan((y - W) * (x + L) / ((z + H) * jnp.sqrt((y - W) ** 2 + (z + H) ** 2 + (x + L) ** 2)))
    )


def _compute_BZ(x, y, z, L, W, H):
    # Vectorized computation of BZ component
    return (
        -(x - L) * jnp.log(z + H + jnp.sqrt((z + H) ** 2 + (y - W) ** 2 + (x - L) ** 2))
        + (x + L) * jnp.log(z + H + jnp.sqrt((z + H) ** 2 + (y - W) ** 2 + (x + L) ** 2))
        + (x - L) * jnp.log(z - H + jnp.sqrt((z - H) ** 2 + (y - W) ** 2 + (x - L) ** 2))
        - (x + L) * jnp.log(z - H + jnp.sqrt((z - H) ** 2 + (y - W) ** 2 + (x + L) ** 2))
        + (x - L) * jnp.log(z + H + jnp.sqrt((z + H) ** 2 + (y + W) ** 2 + (x - L) ** 2))
        - (x + L) * jnp.log(z + H + jnp.sqrt((z + H) ** 2 + (y + W) ** 2 + (x + L) ** 2))
        - (x - L) * jnp.log(z - H + jnp.sqrt((z - H) ** 2 + (y + W) ** 2 + (x - L) ** 2))
        + (x + L) * jnp.log(z - H + jnp.sqrt((z - H) ** 2 + (y + W) ** 2 + (x + L) ** 2))
        - (z + H) * jnp.log(x - L + jnp.sqrt((z + H) ** 2 + (y - W) ** 2 + (x - L) ** 2))
        + (z + H) * jnp.log(x + L + jnp.sqrt((z + H) ** 2 + (y - W) ** 2 + (x + L) ** 2))
        + (z - H) * jnp.log(x - L + jnp.sqrt((z - H) ** 2 + (y - W) ** 2 + (x - L) ** 2))
        - (z - H) * jnp.log(x + L + jnp.sqrt((z - H) ** 2 + (y - W) ** 2 + (x + L) ** 2))
        + (z + H) * jnp.log(x - L + jnp.sqrt((z + H) ** 2 + (y + W) ** 2 + (x - L) ** 2))
        - (z + H) * jnp.log(x + L + jnp.sqrt((z + H) ** 2 + (y + W) ** 2 + (x + L) ** 2))
        - (z - H) * jnp.log(x - L + jnp.sqrt((z - H) ** 2 + (y + W) ** 2 + (x - L) ** 2))
        + (z - H) * jnp.log(x + L + jnp.sqrt((z - H) ** 2 + (y + W) ** 2 + (x + L) ** 2))
        + (y - W) * jnp.arctan((z + H) * (x - L) / ((y - W) * jnp.sqrt((z + H) ** 2 + (y - W) ** 2 + (x - L) ** 2)))
        - (y - W) * jnp.arctan((z + H) * (x + L) / ((y - W) * jnp.sqrt((z + H) ** 2 + (y - W) ** 2 + (x + L) ** 2)))
        - (y - W) * jnp.arctan((z - H) * (x - L) / ((y - W) * jnp.sqrt((z - H) ** 2 + (y - W) ** 2 + (x - L) ** 2)))
        + (y - W) * jnp.arctan((z - H) * (x + L) / ((y - W) * jnp.sqrt((z - H) ** 2 + (y - W) ** 2 + (x + L) ** 2)))
        - (y + W) * jnp.arctan((z + H) * (x - L) / ((y + W) * jnp.sqrt((z + H) ** 2 + (y + W) ** 2 + (x - L) ** 2)))
        + (y + W) * jnp.arctan((z + H) * (x + L) / ((y + W) * jnp.sqrt((z + H) ** 2 + (y + W) ** 2 + (x + L) ** 2)))
        + (y + W) * jnp.arctan((z - H) * (x - L) / ((y + W) * jnp.sqrt((z - H) ** 2 + (y + W) ** 2 + (x - L) ** 2)))
        - (y + W) * jnp.arctan((z - H) * (x + L) / ((y + W) * jnp.sqrt((z - H) ** 2 + (y + W) ** 2 + (x + L) ** 2)))
    )
