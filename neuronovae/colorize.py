import cv2
import numpy as np
from neuronovae.dataset import Activity, Background, Features

# temp
from neuronovae.scratch_load_cmn import load_cmn

"""
This module contains the implementation for creating blended visuals
"""


def colorize_overlay():
    """
    Colorize an image using a colormap.
    """
    # (0) Test Case

    # (1) Load anything we need
    activity = None
    colormap = None
    features = None
    filenames: set | None = None
    rois = None
    vmin, vmax = None, None

    # (2) Establish background image
    image = np.zeros((cmn.get("dims"), np.uint8))
    background = Background(image=image)


class Colorizer:
    """
    Convenience class for colorizing images
    """
    ...