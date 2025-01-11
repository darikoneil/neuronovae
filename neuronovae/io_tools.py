from os import PathLike
from pathlib import Path

import cv2
import numpy as np
from tifffile import TiffFileError, imread, imwrite

from neuronovae.constants import Constants
from neuronovae.exceptions import FileFormatError
from neuronovae.hints import File
from neuronovae.validation import convert

__all__ = [
    "load_image",
]


"""
This module contains functions for file I/O operations.
"""


"""
////////////////////////////////////////////////////////////////////////////////////////
// LOADING IMAGES
////////////////////////////////////////////////////////////////////////////////////////
"""


def _load_numpy(file: Path, **kwargs) -> np.ndarray:
    """
    Loads a numpy file from a file. It is always read in read-only mode, and will be
    copied to memory later if necessary.

    :param file: The path to the numpy file
    :param kwargs: Additional keyword arguments that describe the shape, type, and
        order of memory mapped data (if applicable).
    :returns: The image as a numpy array
    """
    if file.suffix == ".npy":
        return np.load(file, allow_pickle=False, mmap_mode="r")
    elif file.suffix == ".mmap":
        shape = kwargs.get("shape", (-1,))
        dtype = kwargs.get("dtype", np.float64)
        order = kwargs.get("order", "C")
        # noinspection PyTypeChecker
        return np.memmap(file, dtype=dtype, mode="r", shape=shape, order=order)
    else:  # pragma: no cover
        raise RuntimeError(
            f"This should never happen! Why was {file.suffix} passed to _load_numpy!"
        )


def _load_opencv(file: Path) -> np.ndarray:
    """
    Loads an image using the OpenCV package.

    :param file: The path to the image file
    :returns: The image as a numpy array
    """
    return cv2.imread(str(file), cv2.IMREAD_UNCHANGED)


def _load_tifffile(file: Path) -> np.ndarray:
    """
    Loads a tifffile from a file. It is always read in read-only mode, and will be
    copied to memory later if necessary.

    :param file: The path to the tifffile
    :returns: The image as a numpy array
    """
    return imread(file, mode="r")


@convert("file", (str | Path | PathLike), Path)
def load_image(file: File, **kwargs) -> np.ndarray:
    """
    Loads a 2, 3 or 4D image from a file. If the image is 2D, it will be in the form
    of a Y x X numpy array. If the image is 3D, it will be in the form of a Frames x
    Y x X numpy array. If the image is 4D, it will be in the form of Frames x Z x Y x X
    numpy array.

    .. attention::
        This provided file should be in a supported file format
        (see :class:`SupportedFileFormats <neuronovae.constants.Constants>`).

    :param file: The path to the image file
    :param kwargs: Additional keyword arguments that describe the shape, type,
        and order of memory mapped data (if applicable).
    :returns: The image as a numpy array
    :raises FileFormatError: If the file format is not supported
    """
    # Dispatch to the appropriate file reader based on the file extension
    # This is pretty funny, so I'm going to keep this solution.
    match extension := file.suffix:
        case _ if extension in Constants.NUMPY_FILE_FORMATS:
            return _load_numpy(file, **kwargs)
        case _ if extension in Constants.OPENCV_FILE_FORMATS:
            return _load_opencv(file)
        case _ if extension in Constants.TIFFFILE_FILE_FORMATS:
            return _load_tifffile(file)
        case _:
            try:
                return _load_tifffile(file)
            except TiffFileError:
                # do not raise the error, we don't care
                raise FileFormatError(file)  # noqa: TRY200


"""
////////////////////////////////////////////////////////////////////////////////////////
// SAVING IMAGES
////////////////////////////////////////////////////////////////////////////////////////
"""


def _save_numpy_npy(file: Path, image: np.ndarray) -> None:
    """
    Saves an array to a numpy file.

    :param file: The path to the numpy file
    :param image: The array to save
    """
    np.save(file, image, allow_pickle=False)


def _save_numpy_mmap(file: Path, image: np.ndarray) -> None:
    """
    Saves an array to a memory-mapped numpy file.

    :param file: The path to the numpy file
    :param image: The array to save
    """
    # noinspection PyArgumentList
    image_ = np.memmap(file, dtype=image.dtype, mode="w+", shape=image.shape)
    image_[:, ...] = image[:, ...]
    image_.flush()
    image_ = None


@convert("file", (str | Path | PathLike), str)
def _save_opencv_2d(file: str, image: np.ndarray) -> None:
    """
    Saves a 2D image to a file using OpenCV.

    :param file: The path to the file
    :param image: The image to save
    """
    cv2.imwrite(file, image)


def _save_tifffile(file: Path, image: np.ndarray) -> None:
    """
    Saves an array to a tifffile.

    :param file: The path to the tifffile
    :param image: The array to save
    """
    imwrite(file, image)


@convert("file", (str | Path | PathLike), Path)
def save(file: File, image: np.ndarray) -> None:
    """
    Saves an array to a file. The file format is determined by the file extension.

    :param file: The path to the file
    :param image: The image to save
    :raises FileFormatError: If the file format is not supported
    """
    match extension := file.suffix:
        case ".npy":
            _save_numpy_npy(file, image)
        case ".mmap":
            _save_numpy_mmap(file, image)
        case ".gif":
            ...
        case ".mp4":
            ...
        case ".avi":
            ...
        case _ if extension in Constants.TIFFFILE_FILE_FORMATS:
            _save_tifffile(file, image)
        case _ if extension in Constants.OPENCV_FILE_FORMATS.intersection(
            Constants.SUPPORTED_2D_EXTENSIONS
        ):
            _save_opencv_2d(file, image)
        case _:
            raise FileFormatError(file)
