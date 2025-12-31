from functools import cached_property
from typing import NamedTuple

import numpy as np
from scipy.spatial import ConvexHull

__all__ = [
    "ROI",
]


class Centroid(NamedTuple):
    y: float
    x: float


def calculate_centroid(rc_vertices: np.ndarray) -> Centroid:
    """
    Calculate the centroid of a polygonal ROI using the shoelace formula.

    If the input is a 1D array, returns it transposed. Otherwise, computes the centroid
    from the vertices of the polygon.

    :param rc_vertices: Vertices of the ROI as a numpy array (N, 2).
    :returns: The centroid as a Centroid object.
    """
    if rc_vertices.ndim == 1:
        return rc_vertices.T
    center_x = 0
    center_y = 0
    sigma_signed_area = 0

    points = rc_vertices.shape[0]
    for pt in range(points):
        if pt < points - 1:
            trapezoid_area = (rc_vertices[pt, 0] * rc_vertices[pt + 1, 1]) - (
                rc_vertices[pt + 1, 0] * rc_vertices[pt, 1]
            )
            sigma_signed_area += trapezoid_area
            center_y += (rc_vertices[pt, 0] + rc_vertices[pt + 1, 0]) * trapezoid_area
            center_x += (rc_vertices[pt, 1] + rc_vertices[pt + 1, 1]) * trapezoid_area
        else:
            trapezoid_area = (rc_vertices[pt, 0] * rc_vertices[0, 1]) - (
                rc_vertices[0, 0] * rc_vertices[pt, 1]
            )
            sigma_signed_area += trapezoid_area
            center_y += (rc_vertices[pt, 0] + rc_vertices[0, 0]) * trapezoid_area
            center_x += (rc_vertices[pt, 1] + rc_vertices[0, 1]) * trapezoid_area

    signed_area = abs(sigma_signed_area) / 2
    center_x /= 6 * signed_area
    center_y /= 6 * signed_area
    center_x = abs(center_x)
    center_y = abs(center_y)
    return Centroid(center_y, center_x)


def calculate_mask(
    pixels: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=bool)
    int_y = np.round(pixels[:, 0]).astype(np.intp)
    int_x = np.round(pixels[:, 1]).astype(np.intp)
    mask[int_y, int_x] = True
    return mask


def flatten_index(shape: tuple[int, ...], indices: np.ndarray) -> int:
    return np.ravel_multi_index([indices[:, dim] for dim in range(len(shape))], shape)


def identify_vertices(
    pixels: np.ndarray,
) -> np.ndarray:
    hull = ConvexHull(pixels)
    return pixels[hull.vertices, ...]


class ROI:
    def __init__(
        self,
        pixels: np.ndarray,
        weight: np.ndarray,
        image_shape: tuple[int, int],
    ):
        self._pixels = pixels
        self._weight = weight
        self._image_shape = image_shape

    def __str__(self) -> str:
        return f"ROI (y:{self.centroid[0]}, x:{self.centroid[1]})"

    def __repr__(self) -> str:
        return (
            f"ROI(y={self.centroid[0]}, x={self.centroid[1]}, "
            f"image_shape={self._image_shape})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ROI):
            return False
        # NOTE: int is used to mitigate floating point precision issues
        return (
            int(self.centroid[0]) == int(other.centroid[0])
            and int(self.centroid[1]) == int(other.centroid[1])
            and self._image_shape == other._image_shape
        )

    def __hash__(self) -> int:
        return hash(
            (
                int(self.centroid[0]),
                int(self.centroid[1]),
                int(self._image_shape[0]),
                int(self._image_shape[1]),
                self.area,
                self.radii[0],
                self.radii[1],
            )
        )

    @cached_property
    def centroid(self) -> Centroid:
        return calculate_centroid(self.vertices)

    @cached_property
    def index(self) -> np.ndarray:
        return flatten_index(self._image_shape, self._pixels)

    @cached_property
    def mask(self) -> np.ndarray:
        return calculate_mask(self._pixels, self._image_shape)

    @cached_property
    def weights(self) -> np.ndarray:
        weights = np.zeros(self._image_shape, dtype=float)
        weights.ravel()[self.index] = self._weight / self._weight.max()
        weights /= np.linalg.norm(weights.ravel(), ord=1)
        return weights

    @cached_property
    def vertices(self) -> np.ndarray:
        return identify_vertices(self._pixels)
