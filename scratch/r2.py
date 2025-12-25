from collections.abc import Iterator
from functools import cached_property, partial
from itertools import product
from typing import Any, Literal
from weakref import proxy

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist

from scratch.utilities import calculate_centroid, rescale
from scratch.typehints import (
    ArrayLike,
    Centroid,
    Orientation,
    Pixels,
    Position,
    Shape,
    Vertices,
)


def format_pixels(pixels: Pixels, orientation: Orientation = "rc") -> NDArray[int]:
    """
    Formats pixel coordinates based on the given orientation.

    :param pixels: The pixel coordinates, which can be a tuple of integers or a numpy array with any number of rows and exactly 2 columns, with integer elements.
    :param orientation: The orientation of the pixel coordinates. If "xy", the first element of the pixel coordinates is the x-coordinate and the second element is the y-coordinate. If "rc", the first element of the pixel coordinates is the row-coordinate (i.e., "y") and the second element is the column-coordinate.
    :return: A numpy array of processed pixel coordinates.
    :raises ValueError: If the orientation is not "xy" or "rc".
    """
    pixels = np.asarray(pixels)
    pixels = pixels[~np.isnan(pixels).any(axis=1)]
    if orientation == "xy":
        return np.flip(pixels)
    if orientation == "rc":
        return pixels
    raise ValueError("Orientation must be 'xy' or 'rc'")


def calculate_radii(
    rc_centroid: Centroid,
    rc_vertices: Vertices,
    method: Literal["mean", "bound", "unbound", "ellipse"] = "mean",
) -> np.ndarray:
    """
    Calculates the radius of the ROI using one of the following methods:

        1. **mean**
            A symmetrical radius calculated as the average distance between the centroid and the vertices of
            the approximate convex hull.
        2. **bound**
            A symmetrical radius calculated as the minimum distance between the centroid and the vertices of
            the approximate convex hull - 1.
        3. **unbound**
            A symmetrical radius calculated as 1 + the maximal distance between the centroid and the
            vertices of the approximate convex hull.
        4. **ellipse**
            An asymmetrical set of radii whose major-axis radius forms the angle theta with respect to the
            y-axis of the reference image.

    :param rc_centroid: The centroid of the ROI, which can be either a tuple of two floats or a numpy array with shape (2,) and float elements.
    :param rc_vertices: The vertices of the ROI, which is a numpy array with any number of rows and exactly 2 columns, with integer elements.
    :param method: The method to use for calculating the radius. Must be one of "mean", "bound", "unbound", or "ellipse".
    :return: A numpy array containing the calculated radii.
    :raises ValueError: If the method is not one of "mean", "bound", "unbound", or "ellipse".
    """
    center = np.asarray(rc_centroid)
    center = np.reshape(center, (1, 2))
    vertices = np.asarray(rc_vertices)
    # noinspection PyTypeChecker
    radii = cdist(center, vertices)

    if method == "mean":
        return np.repeat(np.mean(radii), 2)
    if method == "bound":
        return np.repeat(np.min(radii) - 1, 2)
    if method == "unbound":
        return np.repeat(np.max(radii) + 1, 2)
    if method == "ellipse":
        return radii
    raise ValueError("Method must be 'mean', 'bound', 'unbound', or 'ellipse'")


def approximate_contours(
    centroid: Centroid, radii: ArrayLike, num_points: int = 100
) -> np.ndarray:
    cy, cx = centroid
    ry, rx = radii
    theta = np.linspace(0, 2 * np.pi, num_points)
    y = cy + ry * np.sin(theta)
    x = cx + rx * np.cos(theta)
    # noinspection PyTypeChecker
    return np.vstack([y, x]).T


def approximate_roi(
    centroid: Centroid,
    radii: ArrayLike,
    reference_shape: Shape = (256, 256),
    plane: int = 0,
) -> "ROI":
    """
    Approximates the ROI based on the given centroid and radii.

    :param centroid: The centroid of the ROI, which can be either a tuple of two floats or a numpy array with shape (2,) and float elements.
    :param radii: The radii of the ROI, which can be either a tuple of two floats or a numpy array with shape (2,) and float elements.
    :param reference_shape: The shape of the reference image from which the ROI was generated.
    :param plane: The plane of the ROI.
    :return: An ROI object representing the approximated ROI.
    """
    # ensure center is numpy array
    centroid = np.asarray(centroid).copy()

    # make sure radii contains both x & y directions
    try:
        assert len(radii) == 2
    except TypeError:
        radii = np.asarray([radii, radii])
    except AssertionError:
        radii = np.asarray([*radii, *radii])

    # generate a rectangle that bounds our mask (upper left, lower right)
    bounding_rect = np.vstack(
        [
            np.ceil(centroid - radii).astype(int),
            np.floor(centroid + radii).astype(int),
        ]
    )

    # constrain to within the reference_shape of the image, if necessary
    bounding_rect[:, 1] = bounding_rect[:, 1].clip(0, reference_shape[-1] - 1)

    # adjust center
    centroid -= bounding_rect[0, :]

    # bounding shape
    bounding = bounding_rect[1, :] - bounding_rect[0, :] + 1
    y_grid, x_grid = np.ogrid[0 : float(bounding[0]), 0 : float(bounding[1])]

    # origin
    y, x = centroid
    r_rad, c_rad = radii

    #         ((x * cos(alpha) + y * sin(alpha)) / x_radius) ** 2 +
    #         ((x * sin(alpha) - y * cos(alpha)) / y_radius) ** 2 = 1

    r, c = (y_grid - y), (x_grid - x)
    distances = (r / r_rad) ** 2 + (c / c_rad) ** 2

    # collect
    yy, xx = np.nonzero(distances < 1)

    # adj bounds
    yy += bounding_rect[0, 0]
    xx += bounding_rect[0, 1]

    # constrain to within the reference_shape of the image, if necessary
    yy.clip(0, reference_shape[0] - 1)
    xx.clip(0, reference_shape[-1] - 1)
    pixels = np.vstack([yy, xx]).T
    # noinspection PyTypeChecker
    return ROI(pixels, reference_shape, plane)


def index_vertices(pixels: Pixels, orientation: Orientation = "rc") -> Vertices:
    """
    Identifies the points of a given polygon which form the vertices of the approximate convex hull. This function
    wraps :class:`scipy.spatial.ConvexHull`, which is ultimately a wrapper for `QHull <https://www.qhull.org>`_.
    It's a fast and easy alternative to actually determining the *true* boundary vertices given the assumption that
    cellular ROIs are convex (i.e., cellular rois ought to be roughly elliptical).

    :param pixels: The pixel coordinates, which can be a tuple of integers or a numpy array with any number of rows and exactly 2 columns, with integer elements.
    :param orientation: The orientation of the pixel coordinates. If "xy", the first element of the pixel coordinates is the x-coordinate and the second element is the y-coordinate. If "rc", the first element of the pixel coordinates is the row-coordinate (i.e., "y") and the second element is the column-coordinate.
    :return: A numpy array containing the indices of the vertices of the approximate convex hull.
    """
    pixels = format_pixels(pixels, orientation)

    # approximate convex hull
    hull = ConvexHull(pixels)

    # return the vertices
    return hull.vertices.astype(np.int16)


def fill_roi(
    pixels: Pixels,
    reference_shape: Shape = (256, 256),
    orientation: Orientation = "rc",
) -> NDArray[int]:
    """
    Fills the ROI by identifying all the pixels within the approximate convex hull of the given pixel coordinates.

    :param pixels: The pixel coordinates, which can be a tuple of integers or a numpy array with any number of rows and exactly 2 columns, with integer elements.
    :param reference_shape: The shape of the reference image from which the ROI was generated.
    :param orientation: The orientation of the pixel coordinates. If "xy", the first element of the pixel coordinates is the x-coordinate and the second element is the y-coordinate. If "rc", the first element of the pixel coordinates is the row-coordinate (i.e., "y") and the second element is the column-coordinate.
    :return: A numpy array containing the coordinates of all the pixels within the ROI.
    """
    pixels = format_pixels(pixels, orientation)
    hull = Delaunay(pixels)
    all_pixels = np.asarray(
        list(product(range(reference_shape[0]), range(reference_shape[1])))
    )
    bool_mask = hull.find_simplex(all_pixels) >= 0
    return all_pixels[bool_mask]


def calculate_position(
    roi: "ROI",
    reference_shape: Shape,
    plane_center: Position,
    bounds: ArrayLike,
) -> Position:
    cy, cx = roi.centroid
    ymin, xmin = (0, 0)
    ymax, xmax = reference_shape
    ymax -= 1  # 0-based indexing
    xmax -= 1  # 0-based indexing

    # noinspection PyTypeChecker
    y = rescale(cy, ymin, ymax, bounds[0], bounds[1]) + plane_center.y
    # noinspection PyTypeChecker
    x = rescale(cx, xmin, xmax, bounds[2], bounds[3]) + plane_center.x

    return Position(plane_center[0], y, x)


def calculate_pixels(
    centroid: Centroid, reference_shape: Shape, plane_center: Pixels, bounds: ArrayLike
):
    cy, cx = centroid
    ymin, xmin = bounds[0], bounds[2]
    ymax, xmax = bounds[1], bounds[3]
    ymax += 1  # 1->0-based indexing
    xmax += 1  # 1->0-based indexing

    # noinspection PyTypeChecker
    y = rescale(cy, ymin, ymax, 0, reference_shape[0]) - plane_center.y
    # noinspection PyTypeChecker
    x = rescale(cx, xmin, xmax, 0, reference_shape[1]) - plane_center.x

    return y, x


class ROI:
    def __init__(
        self,
        pixels: Pixels,
        reference_shape: Shape = (256, 256),
        plane: int = 0,
        orientation: Orientation = "rc",
        position: Position | None = None,
        **properties: Any,
    ):
        """
        An abstract ROI object containing the base characteristics & properties of an ROI. Technically,
        the only abstract method is __name__. Therefore, it isn't *really* abstract, and it is not meant
        to be instanced; it contains the abstract method for protection. Note that the properties
        are only calculated once.

        :warning: Note that these properties are only calculated **once** and then permanently cached for performance.

        :param pixels: The pixel-coordinates of the ROI

        :param reference_shape: The shape of the reference image from which the roi was generated

        :param orientation: The orientation of the pixel-coordinates.
            If "xy", the first element of the pixel-coordinates is the x-coordinate and
            the second element is the y-coordinate. If "rc", the first element of the
            pixel-coordinates is the row-coordinate (i.e., "y") and the second element
            is the column-coordinate.

        :param properties: Optional properties to include
        """
        pixels = format_pixels(pixels, orientation)
        self.x_pixels: Pixels = pixels[:, 1]
        self.y_pixels: Pixels = pixels[:, 0]
        self.plane: int = plane
        self.reference_shape: Shape = reference_shape
        self.properties: dict = properties
        self.position = position

        hull = ConvexHull(pixels)
        self.vert_idx = hull.vertices
        self.area = hull.area
        self.vertices: Vertices = pixels[self.vert_idx]
        self.centroid: Centroid = calculate_centroid(self.vertices)

    def __str__(self) -> str:
        return (
            f"ROI (y:{self.centroid[0]}, x:{self.centroid[1]}) with ~"
            f"{np.max(self.radii):.2f} diameter"
        )

    def __repr__(self) -> str:
        return (
            f"ROI(y={self.centroid[0]}, x={self.centroid[1]}, "
            f"reference_shape={self.reference_shape}, plane={self.plane})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ROI):
            return False
        # int is used to mitigate floating point precision issues
        return (
            int(self.centroid[0]) == int(other.centroid[0])
            and int(self.centroid[1]) == int(other.centroid[1])
            and self.reference_shape == other.reference_shape
            and self.plane == other.plane
        )

    def __hash__(self) -> int:
        return hash(
            (
                int(self.centroid[0]),
                int(self.centroid[1]),
                int(self.reference_shape[0]),
                int(self.reference_shape[1]),
                self.area,
                self.radii[0],
                self.radii[1],
                self.plane,
            )
        )

    @cached_property
    def mask(self) -> np.ndarray:
        """
        Boolean mask for the ROI
        """
        mask = np.zeros(self.reference_shape, dtype=bool)
        int_y = np.round(self.y_pixels).astype(np.int16)
        int_x = np.round(self.x_pixels).astype(np.int16)
        for y, x in zip(int_y, int_x, strict=False):
            mask[y, x] = True
        return mask

    @cached_property
    def filled(self) -> Pixels:
        """
        Filled ROI
        """
        return fill_roi(self.vertices, self.reference_shape, "rc")

    @cached_property
    def radii(self) -> np.ndarray:
        """
        Radii of the ROI
        """
        return calculate_radii(self.centroid, self.vertices)

    @cached_property
    def contours(self) -> np.ndarray:
        """
        Contours of the ROI
        """
        return approximate_contours(self.centroid, self.radii)


class FOV:
    def __init__(
        self,
        rois: list[ROI] | None = None,
        reference_image: Shape = (256, 256),
        fov_size: tuple[float, float] = (412.5, 412.5),
        center: Position = (0, 0, 0),
        plane: int = 0,
    ):
        self.rois = []
        if isinstance(reference_image, np.ndarray) and all(
            shape > 1 for shape in reference_image.shape
        ):
            self.reference_image = reference_image
            self._reference_shape = reference_image.shape
        else:
            self._reference_shape = reference_image
            self.reference_image = np.zeros(reference_image)
        self._fov_size = np.asarray(fov_size).astype(np.float64)
        try:
            assert all([hasattr(center, attr) for attr in ["z", "y", "x"]])
        except AssertionError:
            center = Position(*np.asarray(center).astype(np.float64))
        finally:
            self._center = center
        self._plane = plane
        self.calculate_position = partial(
            calculate_position,
            reference_shape=self._reference_shape,
            plane_center=self._center,
            bounds=(
                -self.fov_size[0] / 2,
                self.fov_size[0] / 2,
                -self.fov_size[1] / 2,
                self.fov_size[1] / 2,
            ),
        )
        rois = rois or []
        if len(rois) > 0:
            for roi in rois:
                self.add(roi)

    @property
    def center(self) -> Position:
        return self._center

    @property
    def centroids(self) -> np.ndarray:
        return np.vstack([roi.centroid for roi in self.rois])

    @cached_property
    def bounds(self) -> np.ndarray:
        return np.asarray(
            [
                self.center[1] - self.fov_size[0] / 2,
                self.center[1] + self.fov_size[0] / 2,
                self.center[2] - self.fov_size[1] / 2,
                self.center[2] + self.fov_size[1] / 2,
            ]
        )

    @property
    def fov_size(self) -> np.ndarray:
        return self._fov_size

    @property
    def num_rois(self) -> int:
        return len(self.rois)

    @property
    def plane(self) -> int:
        return self._plane

    @property
    def positions(self) -> np.ndarray:
        # noinspection PyTypeChecker
        return np.vstack([roi.position for roi in self.rois])

    @property
    def reference_shape(self) -> Shape:
        return self._reference_shape

    @property
    def mask(self) -> np.ndarray:
        return np.sum([roi.mask for roi in self.rois], axis=0, dtype=bool)

    def add(
        self, roi: ROI | np.ndarray, orientation: Orientation = "rc", **properties: Any
    ) -> None:
        """
        Adds a ROI to the FOV. If the ROI is a numpy array, it is converted to a ROI object.

        :param roi: The ROI to add
        :param orientation: The orientation of the pixel-coordinates.
            If "xy", the first element of the pixel-coordinates is the x-coordinate and
            the second element of the pixel-coordinates. If "rc", the first element of the
            pixel-coordinates is the row-coordinate (i.e., "y") and the second element
            is the column-coordinate.
        :param properties: Optional properties to include
        """
        if isinstance(roi, np.ndarray):
            if "plane" not in properties:
                properties["plane"] = self.plane
            # assert properties["plane"] == self.plane
            if "reference_shape" not in properties:
                properties["reference_shape"] = self.reference_shape
            # assert properties["reference_shape"] == self.reference_shape
            roi = ROI(roi, orientation=orientation, **properties)
        roi.properties["fov"] = proxy(self)
        roi.position = self.calculate_position(roi)
        self.rois.append(roi)

    def __iter__(self) -> Iterator["ROI"]:
        return iter(self.rois)

    def __str__(self) -> str:
        return (
            f"FOV (y:{self.center[0]}, x:{self.center[1]}, z:{self.center[2]}) with "
            f"{self.num_rois} ROIs"
        )
