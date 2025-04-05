from typing import List, Tuple
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class RectangularSegment:
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    width: float
    height: float

    # make it zippable
    def __iter__(self):
        return iter((self.start, self.end, self.width, self.height))


class RectangularConductor:
    """
    A wire segment in 3D space defined by a rectangular cross-section.

    Attributes:
        segments (List[Rectangular3D]): List of rectangular segments in millimeters (mm).
        current (float): Current flowing through the wire in Amperes (A).
    """

    def __init__(
        self,
        material: str,
        current: float,
        segments: List[RectangularSegment],
    ):
        self.material = material
        self.current = current
        self.starts, self.ends, self.widths, self.heights = map(jnp.float64, zip(*(segments)))

    @property
    def currents(self) -> jnp.ndarray:
        return jnp.full(self.starts.shape[0], self.current, dtype=jnp.float64)

    def get_vertices(self) -> jnp.ndarray:
        """
        Calculate the vertices of the rectangular conductor segments.

        Returns:
            List[Point3D]: List of vertices of the rectangular segments.
        """
        return _get_vertices(
            self.starts,
            self.ends,
            self.widths,
            self.heights,
        )


# fmt: off
def _get_vertices(
    starts : jnp.ndarray,
    ends   : jnp.ndarray,
    widths : jnp.ndarray,
    heights: jnp.ndarray,
) -> jnp.ndarray:
# fmt: on
    vectors = ends - starts
    lengths = jnp.linalg.norm(vectors, axis=1)

    # local coordinates
    # fmt: off
    offsets = jnp.array([
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ])[jnp.newaxis, ...] # (1, 8, 3)
    
    halves = jnp.array([
        lengths / 2, 
        widths  / 2, 
        heights / 2]).T[:, jnp.newaxis, :] # (M, 1, 3)
    # fmt: on

    vertices = offsets * halves  # (1, 8, 3) * (M, 1, 3) = (M, 8, 3)

    # rotate vertices
    alpha = jnp.arctan2(vectors[:, 1], vectors[:, 0])
    beta = jnp.arcsin(vectors[:, 2] / lengths)
    cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
    cos_b, sin_b = jnp.cos(beta), jnp.sin(beta)
    zeros = jnp.zeros_like(alpha)
    ones = jnp.ones_like(alpha)

    # Construct CW y-rotation matrix (3, 3, N)
    # fmt: off
    rot_y = jnp.array(
        [
            [ cos_b, zeros, sin_b],
            [ zeros, ones , zeros],
            [-sin_b, zeros, cos_b],
        ]
    )
    # fmt: on

    # Construct CW z-rotation matrix (3, 3, N)
    # fmt: off
    rot_z = jnp.array(
        [
            [ cos_a,  sin_a, zeros],
            [-sin_a,  cos_a, zeros],
            [ zeros,  zeros, ones],
        ]
    )
    # fmt: on
    rot = rot_z.T @ rot_y.T  # (N, 3, 3)
    vertices = jnp.einsum("nij,nvj->nvi", rot, vertices)  # (M, 8, 3)

    # translate vertices
    centers = (starts + ends) / 2
    vertices += centers[:, jnp.newaxis, :]  # (M, 8, 3) + (M, 1, 3) = (M, 8, 3)

    return vertices
