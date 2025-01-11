import numpy as np
from neuronovae.validation import convert
from scipy.spatial import ConvexHull


"""
This module contains all the functionality for ROIs.
"""


"""
////////////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////////////
"""


def calculate_radius() -> None:
    ...


def calculate_centroid() -> None:
    ...


def calculate_mask() -> None:
    ...


@convert("pixels", (list, tuple, np.ndarray), np.ndarray, np.asarray)
def identify_vertices(pixels: np.ndarray[np.integer]) -> np.ndarray:
    return ConvexHull(pixels).vertices



"""
////////////////////////////////////////////////////////////////////////////////////////
// ROI Classes
////////////////////////////////////////////////////////////////////////////////////////
"""


class _BaseROI(metaclass=ABCMeta):
    """
    Base class for all ROIs. This class should not be instantiated directly, as it is
    not a complete implementation of an ROI. Instead, it should be subclassed to create
    a specific ROI type.
    """
    ...


class LiteralROI(_BaseROI):
    ...


class ApproximateROI(_BaseROI):
    ...


class BlendedROI(_BaseROI):
    ...


"""
////////////////////////////////////////////////////////////////////////////////////////
// Handlers
////////////////////////////////////////////////////////////////////////////////////////
"""


class ROIHandler:
    ...
