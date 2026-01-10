from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neuronovae.loaders import SUPPORTED_EXTENSIONS, load_images

#: Base name of the 2D test image files
IMAGE_2D_STEM = "dont_panic"

#: Supported 2D image file extensions
EXT_2D = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".npy",
    ".ome.tiff",
    ".tif",
    ".tiff",
    ".png",
}

#: Supported 3D image file extensions
EXT_3D = {
    ".npy",
    ".ome.tiff",
    ".tif",
    ".tiff",
}

#: Supported 4D image file extensions
EXT_4D = {
    ".npy",
    ".avi",
    ".gif",
    ".mp4",
    ".mov",
}


@pytest.fixture
def image_case(request: dict[str, Any]) -> tuple[Path, np.ndarray]:
    """
    Fixture to provide test image file paths and reference data based on dimensions.

    Args:
        request: The pytest request object containing parameters.
    Returns:
        A tuple containing the file path and the reference numpy array.
    """
    dimensions = request.param[0]
    match dimensions:
        case 2:
            filename = request.getfixturevalue("assets_path").joinpath(
                IMAGE_2D_STEM + ".npy"
            )
            reference = np.load(filename, allow_pickle=False)
        case _:
            msg = f"Invalid dimensions: {dimensions}"
            raise NotImplementedError(msg)
    # noinspection PyUnboundLocalVariable
    return filename, reference


# WARNING: Pre test extension validation (ensures tests cover all support extensions). DO NOT REMOVE
def test_extension_coverage() -> None:
    """Ensure all supported extensions have corresponding tests."""
    supported_exts = {ext.lower() for ext in SUPPORTED_EXTENSIONS}
    test_exts = EXT_2D | EXT_3D | EXT_4D
    assert supported_exts == test_exts, (
        "Mismatch between supported extensions and test coverage. "
        f"Supported: {supported_exts}, Tested: {test_exts}"
    )


@pytest.mark.parametrize(
    "image_case",
    [(2, ext) for ext in EXT_2D],
    ids=list(EXT_2D),
    indirect=True,
)
def test_load_images_2d(
    image_case: tuple[Path, np.ndarray],
) -> None:
    """
    Test loading of 2D images from various file formats.

    Args:
        image_case: A tuple containing the file path and the reference numpy array.
    """
    (filename, reference) = image_case
    images = load_images(filename)
    np.testing.assert_array_equal(images, reference)


@pytest.mark.skip("No reference 3D images available for testing.")
@pytest.mark.parametrize(
    "image_case",
    [(3, ext) for ext in EXT_3D],
    ids=list(EXT_3D),
    indirect=True,
)
def test_load_images_3d(
    image_case: tuple[Path, np.ndarray],
) -> None:
    """
    Test loading of 3D images from various file formats.

    Args:
        image_case: A tuple containing the file path and the reference numpy array.
    """
    (filename, reference) = image_case
    images = load_images(filename)
    np.testing.assert_array_equal(images, reference)


@pytest.mark.skip("No reference 4D images available for testing.")
@pytest.mark.parametrize(
    "image_case",
    [(4, ext) for ext in EXT_4D],
    ids=list(EXT_4D),
    indirect=True,
)
def test_load_images_4d(
    image_case: tuple[Path, np.ndarray],
) -> None:
    """
    Test loading of 4D images from various file formats.

    Args:
        image_case: A tuple containing the file path and the reference numpy array.
    """
    (filename, reference) = image_case
    images = load_images(filename)
    np.testing.assert_array_equal(images, reference)
