import numpy as np
from neuronovae.rois import _BaseROI

"""
This module contains the base class for all imaging data, its associated ROIs, and
any other image-related functionality.
"""


class Image:
    """
    Base class for all imaging data, its associated ROIs, and any other image-related
    functionality.
    """
    def __init__(self,
                 image: np.ndarray,
                 rois: dict[int, _BaseROI]):
        self.image = image
        self.rois = rois
        self.background = None
        self.overlay = None
        self.text = None
        self.cutoffs = None

    @property
    def frames(self) -> int:
        """
        The number of frames in the image.
        """
        return self.image.shape[0]

    @property
    def planes(self) -> int:
        """
        The number of planes in the image.
        """
        return self.image.shape[1]

    @property
    def height(self) -> int:
        """
        The number of pixels in the y-dimension of the image.
        """
        return self.image.shape[2]

    @property
    def width(self) -> int:
        """
        The number of pixels in the x-dimension of the image.
        """
        return self.image.shape[3]

    @property
    def total_rois(self) -> int:
        """
        The total number of ROIs in the image.
        """
        return len(self.rois)
