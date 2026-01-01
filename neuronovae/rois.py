from functools import cached_property
from typing import NamedTuple

import numpy as np
from scipy.spatial import ConvexHull

__all__ = [
    "ROI",
]


class Centroid(NamedTuple):
    """
    Represents the centroid of a region of interest (ROI) in a 2D space.

    :param y: The y-coordinate of the centroid.
    :param x: The x-coordinate of the centroid.
    :example:
        centroid = Centroid(y=10.5, x=20.3)
        print(centroid.y, centroid.x)
    :note: This class is immutable and inherits from `NamedTuple`.
    """

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
    """
    Generates a binary mask for the given pixels within the specified image shape.

    :param pixels: A numpy array of pixel coordinates (N, 2) where N is the number of pixels.
    :param image_shape: A tuple representing the shape of the image (height, width).
    :return: A binary mask as a numpy array with the same shape as the image, where True indicates the presence of a pixel.
    :example:
        mask = calculate_mask(np.array([[1, 2], [3, 4]]), (5, 5))
        print(mask)
    :note: The input pixel coordinates are rounded to the nearest integer.
    :attention: Ensure that the pixel coordinates are within the bounds of the image shape to avoid indexing errors.
    """
    mask = np.zeros(image_shape, dtype=bool)
    int_y = np.round(pixels[:, 0]).astype(np.intp)
    int_x = np.round(pixels[:, 1]).astype(np.intp)
    mask[int_y, int_x] = True
    return mask


def flatten_index(shape: tuple[int, ...], indices: np.ndarray) -> int:
    """
    Flattens multi-dimensional indices into a single-dimensional index.

    :param shape: The shape of the multi-dimensional array.
    :param indices: A numpy array of indices with shape (N, D), where N is the number of indices and D is the dimensionality.
    :return: A single-dimensional index corresponding to the input indices.
    :raises ValueError: If the indices are out of bounds for the given shape.
    :example:
        flatten_index((3, 3), np.array([[0, 1], [2, 2]]))
    :note: This function uses numpy's `ravel_multi_index` for the conversion.
    :attention: Ensure that the indices are within the bounds of the provided shape to avoid errors.
    """
    return np.ravel_multi_index([indices[:, dim] for dim in range(len(shape))], shape)


def identify_vertices(
    pixels: np.ndarray,
) -> np.ndarray:
    """
    Identifies the vertices of the convex hull for a given set of pixels.

    :param pixels: A numpy array of pixel coordinates (N, 2) where N is the number of pixels.
    :return: A numpy array of pixel coordinates corresponding to the vertices of the convex hull.
    :example:
        vertices = identify_vertices(np.array([[0, 0], [1, 1], [2, 0]]))
        print(vertices)
    :note: The input pixels should be in a 2D space.
    :attention: Ensure that the input array contains at least three points to form a convex hull.
    """
    hull = ConvexHull(pixels)
    return pixels[hull.vertices, ...]


class ROI:
    """
    Represents a Region of Interest (ROI) in an image.

    :param pixels: A numpy array of pixel coordinates (N, 2) representing the ROI.
    :param weight: A numpy array of weights corresponding to the pixels.
    :param image_shape: A tuple representing the shape of the image (height, width).
    :example:
        roi = ROI(pixels=np.array([[1, 2], [3, 4]]), weight=np.array([0.5, 0.5]), image_shape=(5, 5))
        print(roi)
    :note: The ROI class provides cached properties for centroid, mask, weights, and vertices.
    :attention: Ensure that the pixel coordinates and weights are valid and correspond to the image shape.
    """

    def __init__(
        self,
        pixels: np.ndarray,
        weight: np.ndarray,
        image_shape: tuple[int, int],
    ):
        """
        Initializes the ROI object.

        :param pixels: A numpy array of pixel coordinates (N, 2) representing the ROI.
        :param weight: A numpy array of weights corresponding to the pixels.
        :param image_shape: A tuple representing the shape of the image (height, width).
        """
        self._pixels = pixels
        self._weight = weight
        self._image_shape = image_shape

    def __str__(self) -> str:
        """
        Returns a string representation of the ROI.

        :return: A string in the format "ROI (y:centroid_y, x:centroid_x)".
        """
        return f"ROI (y:{self.centroid[0]}, x:{self.centroid[1]})"

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the ROI.

        :return: A string in the format "ROI(y=centroid_y, x=centroid_x, image_shape=image_shape)".
        """
        return (
            f"ROI(y={self.centroid[0]}, x={self.centroid[1]}, "
            f"image_shape={self._image_shape})"
        )

    def __eq__(self, other: object) -> bool:
        """
        Checks equality between two ROI objects.

        :param other: Another object to compare with.
        :return: True if the objects are equal, False otherwise.
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
        Returns a hash value for the ROI object.

        :return: An integer hash value.
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
        Calculates and caches the centroid of the ROI.

        :return: The centroid as a Centroid object.
        """
        return calculate_centroid(self.vertices)

    @cached_property
    def index(self) -> np.ndarray:
        """
        Calculates and caches the flattened indices of the ROI pixels.

        :return: A numpy array of flattened indices.
        """
        return flatten_index(self._image_shape, self._pixels)

    @cached_property
    def mask(self) -> np.ndarray:
        """
        Generates and caches a binary mask for the ROI.

        :return: A binary mask as a numpy array with the same shape as the image.
        """
        return calculate_mask(self._pixels, self._image_shape)

    @cached_property
    def weights(self) -> np.ndarray:
        """
        Calculates and caches the normalized weights for the ROI.

        :return: A numpy array of normalized weights with the same shape as the image.
        """
        weights = np.zeros(self._image_shape, dtype=float)
        weights.ravel()[self.index] = self._weight / self._weight.max()
        weights /= np.linalg.norm(weights.ravel(), ord=1)
        return weights

    @cached_property
    def vertices(self) -> np.ndarray:
        """
        Identifies and caches the vertices of the convex hull for the ROI.

        :return: A numpy array of pixel coordinates corresponding to the vertices of the convex hull.
        """
        return identify_vertices(self._pixels)
