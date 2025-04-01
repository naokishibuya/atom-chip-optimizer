from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


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
        z_offset: Optional[float] = 0.0,
    ):
        self.material = material
        self.current = current
        self.starts, self.ends, self.widths, self.heights = map(np.float64, zip(*(segments)))

        # apply z_offset to all segments
        self.starts[:, 2] += z_offset
        self.ends[:, 2] += z_offset

    @property
    def currents(self) -> np.ndarray:
        return np.full(self.starts.shape[0], self.current, dtype=np.float64)

    def get_vertices(self) -> np.ndarray:
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
    starts : np.ndarray,
    ends   : np.ndarray,
    widths : np.ndarray,
    heights: np.ndarray,
) -> np.ndarray:
# fmt: on
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)

    # local coordinates
    # fmt: off
    offsets = np.array([
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ])[np.newaxis, ...] # (1, 8, 3)
    
    halves = np.array([
        lengths / 2, 
        widths  / 2, 
        heights / 2]).T[:, np.newaxis, :] # (M, 1, 3)
    # fmt: on

    vertices = offsets * halves  # (1, 8, 3) * (M, 1, 3) = (M, 8, 3)

    # rotate vertices
    alpha = np.arctan2(vectors[:, 1], vectors[:, 0])
    beta = np.arcsin(vectors[:, 2] / lengths)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    zeros = np.zeros_like(alpha)
    ones = np.ones_like(alpha)

    # Construct CW y-rotation matrix (3, 3, N)
    # fmt: off
    rot_y = np.array(
        [
            [ cos_b, zeros, sin_b],
            [ zeros, ones , zeros],
            [-sin_b, zeros, cos_b],
        ]
    )
    # fmt: on

    # Construct CW z-rotation matrix (3, 3, N)
    # fmt: off
    rot_z = np.array(
        [
            [ cos_a,  sin_a, zeros],
            [-sin_a,  cos_a, zeros],
            [ zeros,  zeros, ones],
        ]
    )
    # fmt: on
    rot = rot_z.T @ rot_y.T  # (N, 3, 3)
    vertices = np.einsum("nij,nvj->nvi", rot, vertices)  # (M, 8, 3)

    # translate vertices
    centers = (starts + ends) / 2
    vertices += centers[:, np.newaxis, :]  # (M, 8, 3) + (M, 1, 3) = (M, 8, 3)

    return vertices
