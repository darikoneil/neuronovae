from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from neuronovae.constants import Constants
from neuronovae.exceptions import FileFormatError

# noinspection PyProtectedMember
from neuronovae.io_tools import _load_numpy, _load_opencv, _load_tifffile, load_image
from tests.conftest import get_mmap_meta

"""
Test suite for the io_tools module
"""


"""
////////////////////////////////////////////////////////////////////////////////////////
// LOADING IMAGES
////////////////////////////////////////////////////////////////////////////////////////
"""


@pytest.mark.parametrize(
    "file_extension", (extension for extension in Constants.SUPPORTED_FILE_FORMATS)
)
def test_loader_image_loader_inference(assets_path: Path, file_extension: str) -> None:
    """
    Test the image loader inference mechanism.
    """
    ...


@pytest.mark.parametrize("dimensions", (2,), indirect=False)
@pytest.mark.parametrize("reference_image", (2,), indirect=True)
@pytest.mark.parametrize("test_image_stem", (2,), indirect=True)
@pytest.mark.parametrize(
    "file_extension", (extension for extension in Constants.NUMPY_FILE_FORMATS)
)
def test_load_image_numpy(
    dimensions: int,
    reference_image: np.ndarray,
    test_image_stem: str,
    file_extension: str,
    assets_path: Path,
) -> None:
    test_image_path = assets_path.joinpath(f"{test_image_stem}{file_extension}")
    meta = get_mmap_meta(dimensions, assets_path)
    # the meta will be ignored if not mmap
    image = _load_numpy(test_image_path, **meta)
    # make sure to test for integer and float arrays separately
    if np.issubdtype(image.dtype, np.integer):
        np.testing.assert_array_equal(image, reference_image)
    else:
        np.testing.assert_allclose(image, reference_image)


@pytest.mark.parametrize("dimensions", (2,), indirect=False)
@pytest.mark.parametrize("reference_image", (2,), indirect=True)
@pytest.mark.parametrize("test_image_stem", (2,), indirect=True)
@pytest.mark.parametrize(
    "file_extension",
    (
        extension
        for extension in Constants.OPENCV_FILE_FORMATS.intersection(
            Constants.SUPPORTED_2D_EXTENSIONS
        )
    ),
)
def test_load_image_opencv(
    dimensions: int,
    reference_image: np.ndarray,
    test_image_stem: str,
    file_extension: str,
    assets_path: Path,
) -> None:
    test_image_path = assets_path.joinpath(f"{test_image_stem}{file_extension}")
    image = _load_opencv(test_image_path)
    # make sure to test for integer and float arrays separately
    if np.issubdtype(image.dtype, np.integer):
        if file_extension in (".jpg", ".jpeg"):  # because of lossy compression
            assert (
                image.shape == reference_image.shape
            ), f"Image shape {image.shape} does not match reference shape {reference_image.shape}"
            image_sum = np.sum(np.ravel(image)).astype(np.int64)
            reference_sum = np.sum(np.ravel(reference_image)).astype(np.int64)
            diff_ir = image_sum - reference_sum
            total = image_sum + reference_sum
            A = diff_ir / total
            assert (image_sum - reference_sum) / (
                image_sum + reference_sum
            ) < 1e-4, f"Image sum {np.sum(np.ravel(image))} does not match reference sum {np.sum(np.ravel(reference_image))}"
        else:
            np.testing.assert_array_equal(image, reference_image)
    else:
        np.testing.assert_array_almost_equal(image, reference_image)


@pytest.mark.parametrize("dimensions", (2,), indirect=False)
@pytest.mark.parametrize("reference_image", (2,), indirect=True)
@pytest.mark.parametrize("test_image_stem", (2,), indirect=True)
@pytest.mark.parametrize(
    "file_extension", (extension for extension in Constants.TIFFFILE_FILE_FORMATS)
)
def test_load_image_tifffile(
    dimensions: int,
    reference_image: np.ndarray,
    test_image_stem: str,
    file_extension: str,
    assets_path: Path,
) -> None:
    test_image_path = assets_path.joinpath(f"{test_image_stem}{file_extension}")
    image = _load_tifffile(test_image_path)
    # make sure to test for integer and float arrays separately
    if np.issubdtype(image.dtype, np.integer):
        np.testing.assert_array_equal(image, reference_image)
    else:
        np.testing.assert_allclose(image, reference_image)
