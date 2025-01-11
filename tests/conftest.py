# flake8: noqa
import sys
from os import devnull
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

"""
////////////////////////////////////////////////////////////////////////////////////////
// CONFIGURATION FOR TESTING
////////////////////////////////////////////////////////////////////////////////////////
"""


TEST_IMAGE_NAME_1D = "dont_panic"


TEST_IMAGE_NAME_2D = "dont_panic"


TEST_IMAGE_NAME_3D = "dont_panic"


TEST_IMAGE_NAME_4D = "dont_panic"


"""
////////////////////////////////////////////////////////////////////////////////////////
// FIXTURES
////////////////////////////////////////////////////////////////////////////////////////
"""


@pytest.fixture(scope="session")
def assets_path() -> Path:
    """
    Returns the path to the test assets folder.

    :return: The path to the assets folder
    """
    return Path(Path(__file__).parent).joinpath("assets")


@pytest.fixture(scope="function")
def test_image_stem(request) -> str:
    """
    Returns the stem of the test image when provided the requested dimension

    :param request: The pytest request object
    :return: The stem of the test image
    """
    dimensions = request.param
    match dimensions:
        case 1:
            return TEST_IMAGE_NAME_1D
        case 2:
            return TEST_IMAGE_NAME_2D
        case 3:
            return TEST_IMAGE_NAME_3D
        case 4:
            return TEST_IMAGE_NAME_4D


@pytest.fixture(scope="function")
def reference_image(request, assets_path):
    """
    Returns a reference image for testing purposes.
    """
    dimensions = request.param
    match dimensions:
        case 1:
            return get_image_1d(assets_path)
        case 2:
            return get_image_2d(assets_path)
        case 3:
            return get_image_3d(assets_path)
        case 4:
            return get_image_4d(assets_path)
        case _:
            raise ValueError("Invalid dimensions")


"""
////////////////////////////////////////////////////////////////////////////////////////
// HELPERS
////////////////////////////////////////////////////////////////////////////////////////
"""


def get_mmap_meta(dimensions: int, assets_path: Path) -> dict[str, Any]:
    """
    Returns the metadata for a memory-mapped image.

    :param dimensions: Dimensions of the test image
    :param assets_path: Assets path
    :returns: A dictionary containing the metadata of the image
    :raises ValueError: If the dimensions are invalid
    """

    def format_meta(image_: np.ndarray) -> dict[str, Any]:
        return {"shape": image_.shape, "dtype": image_.dtype, "order": "C"}

    match dimensions:
        case 1:
            return format_meta(get_image_1d(assets_path))
        case 2:
            return format_meta(get_image_2d(assets_path))
        case 3:
            return format_meta(get_image_3d(assets_path))
        case 4:
            return format_meta(get_image_4d(assets_path))
        case _:
            raise ValueError("Invalid dimensions")


def get_image_1d(assets_path) -> np.ndarray:
    """
    Returns a 1D image for testing purposes.

    :param assets_path: The path to the assets folder
    :returns: A 1D numpy array (X)
    """
    # this is a 1D RGB image, so we need to reshape it to a 1D image
    dont_panic_path = assets_path().joinpath(TEST_IMAGE_NAME_1D + ".npy")
    return np.load(dont_panic_path, allow_pickle=False)


def get_image_2d(assets_path) -> np.ndarray:
    """
    Returns a 2D image for testing purposes.

    :param assets_path: The path to the assets folder
    :returns: A 2D numpy array (Y x X)
    """
    # this is a 2D RGB image, so we need to reshape it to a 2D image
    dont_panic_path = assets_path.joinpath(TEST_IMAGE_NAME_2D + ".npy")
    return np.load(dont_panic_path, allow_pickle=False)


def get_image_3d(assets_path) -> np.ndarray:
    """
    Returns a 3D image for testing purposes.

    :param assets_path: The path to the assets folder
    :returns: A 3D numpy array (Frames x Y x X)
    """
    # this is a 3D RGB image, so we need to reshape it to a 3D image
    dont_panic_path = assets_path.joinpath("dont_panic.npy")
    image_3d = np.load(dont_panic_path, allow_pickle=False)
    return image_3d


def get_image_4d(assets_path) -> np.ndarray:
    """
    Returns a 4D image for testing purposes.

    :param assets_path: The path to the assets folder
    :returns: A 4D numpy array (Frames x Z x Y x X)
    """
    # this is a 4D RGB image, so we need to reshape it to a 4D image
    dont_panic_path = assets_path.joinpath("dont_panic.npy")
    image_4d = np.load(dont_panic_path, allow_pickle=False)
    return image_4d


class BlockPrinting:
    """
    Simple context manager that blocks printing
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._stdout
