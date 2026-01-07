from functools import cached_property
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

__all__ = [
    "ROI",
]


#: Alias for a flattened index array
FlatIndex = np.ndarray[tuple[Literal[1]], np.intp]

#: Alias for a 2D index of pixels
Pixels = np.ndarray[tuple[int, Literal[2]], np.intp]


class Centroid(NamedTuple):
    """
    Centroid of an ROI in 2D coordinates.

    Args:
        y: Y coordinate.
        x: X coordinate.
    """

    y: float
    x: float


def calculate_centroid(rc_vertices: Pixels) -> Centroid:
    """
    Calculate centroid of a polygonal ROI using the shoelace formula.

    Args:
        rc_vertices: numpy array of shape (N, 2) indexing the vertices of the roi

    Returns:
        Centroid: The polygon centroid.
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


# TODO: Refine me


def calculate_mask(
    index: FlatIndex,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Create a binary mask for given pixel coordinates.

    Args:
        index: A flat indexing array of pixel coordinates.
        image_shape: (height, width) of the image.

    Returns:
        Boolean mask indicating which pixels belong to the ROI.
    """
    mask = np.zeros(image_shape, dtype=bool)
    mask.ravel()[index] = True
    return mask


def flatten_index(shape: tuple[int, ...], indices: Pixels) -> int:
    """
    Convert multi-dimensional coordinates to flat indices.

    Args:
        shape: Shape of the source array.
        indices: A 2D array of pixel coordinates (N, D).

    Returns:
        int: Flattened indices.

    Raises:
        ValueError: If indices are out of bounds for the given shape.
    """
    return np.ravel_multi_index([indices[:, dim] for dim in range(len(shape))], shape)


def identify_vertices(
    pixels: Pixels,
) -> Pixels:
    """
    Compute convex-hull vertices for a set of pixels.

    Args:
        pixels: A numpy array of pixel coordinates (N, 2).

    Returns:
        The pixels corresponding to convex hull vertices.

    Note:
        See the [`Pixels`][] type alias for explicit typing information.
    """
    hull = ConvexHull(pixels)
    return pixels[hull.vertices, ...]


class ROI:
    """
    Region of Interest (ROI) representation.

    Args:
        pixels: Array of pixel coordinates (N, 2).
        weight: Array of per-pixel weights.
        image_shape: the shape of the image (height, width).

    Warning:
        centroid, index, mask, weights, vertices are calculated *once* for performance.
    """

    def __init__(
        self,
        pixels: Pixels,
        weight: np.ndarray,
        image_shape: tuple[int, int],
    ):
        """
        Initializes the ROI object.

        Args:
            pixels: A numpy array of [`Pixels`][] (N, 2) representing the ROI.
            weight: A numpy array of weights corresponding to the pixels.
            image_shape: A tuple representing the shape of the image (height, width).
        """
        self._pixels = pixels
        self._weight = weight
        self._image_shape = image_shape

    def __str__(self) -> str:
        """
        Return a short string describing the ROI.

        Returns:
            str: Short representation including centroid.
        """
        return f"ROI (y:{self.centroid[0]}, x:{self.centroid[1]})"

    def __repr__(self) -> str:
        """
        Return a detailed representation of the ROI.

        Returns:
            str: Detailed representation including centroid and image shape.
        """
        return (
            f"ROI(y={self.centroid[0]}, x={self.centroid[1]}, "
            f"image_shape={self._image_shape})"
        )

    def __eq__(self, other: object) -> bool:
        """
        Compare two ROI objects for equality.

        Args:
            other: Object to compare.

        Returns:
            bool: True if equal, False otherwise.
        """
        if not isinstance(other, ROI):
            return False
        # NOTE: int is used to mitigate floating point precision issues
        return (
            int(self.centroid[0]) == int(other.centroid[0])
            and int(self.centroid[1]) == int(other.centroid[1])
            and self._image_shape == other._image_shape
        )

    def __hash__(self) -> int:
        """
        Compute a hash value for the ROI.

        Returns:
            int: Hash value.
        """
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
        """
        Compute and cache centroid.

        Returns:
            Centroid: ROI centroid.
        """
        return calculate_centroid(self.vertices)

    @cached_property
    def index(self) -> FlatIndex:
        """
        Compute and cache flattened indices for ROI pixels.

        Returns:
            numpy.ndarray: Flattened indices.
        """
        return flatten_index(self._image_shape, self._pixels)

    @cached_property
    def mask(self) -> NDArray[bool]:
        """
        Compute and cache binary mask for the ROI.

        Returns:
            numpy.ndarray: Boolean mask.
        """
        return calculate_mask(self.index, self._image_shape)

    @cached_property
    def weights(self) -> NDArray[float]:
        """
        Compute and cache normalized weights array for the ROI.

        Returns:
            numpy.ndarray: Weights array normalized to L1 norm.
        """
        weights = np.zeros(self._image_shape, dtype=float)
        weights.ravel()[self.index] = self._weight / self._weight.max()
        weights /= np.linalg.norm(weights.ravel(), ord=1)
        return weights

    @cached_property
    def vertices(self) -> Pixels:
        """
        Compute and cache convex hull vertices for the ROI.

        Returns:
            numpy.ndarray: Vertices coordinates.
        """
        return identify_vertices(self._pixels)
