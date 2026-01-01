from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable
from warnings import warn

import cv2
import numpy as np
from tifffile import imread

from neuronovae.issues import FileFormatError, PickleWarning
from neuronovae.rois import ROI

__all__ = ["ROIHandler", "Suite2PHandler", "load_images", "load_rois"]

"""
|=======================================================================================|
|DISPATCHING & REGISTRY OF IMAGING LOADERS
|=======================================================================================|
"""


@runtime_checkable
class ImagingLoader(Protocol):
    """
    Protocol for imaging loader functions.

    :param file: The file path to load the image from.
    :return: The loaded image data.
    """

    def __call__(self, file: Path) -> Any: ...


class _ImagingLoaderRegistry:
    """
    Registry for managing imaging loader functions based on file extensions.

    :note: This class provides methods to register and retrieve loaders for specific file extensions.
    """

    __registry: ClassVar[dict[str, ImagingLoader]] = {}

    @classmethod
    def register(cls, *extension: str) -> ImagingLoader:
        """
        Registers an imaging loader for the specified file extensions.

        :param extension: File extensions to associate with the loader.
        :return: The registered loader function.
        """

        def decorator(func: ImagingLoader) -> ImagingLoader:
            nonlocal extension
            for ext in extension:
                sanitized_ext = cls._sanitize_extension(ext)
                cls.__registry[sanitized_ext] = func
            return func

        return decorator

    @classmethod
    def get_loader(cls, extension: str) -> ImagingLoader | None:
        """
        Retrieves the loader associated with the given file extension.

        :param extension: The file extension to look up.
        :return: The loader function, or None if no loader is registered.
        """
        # NOTE: We're returning None here to propogate the error up to the caller
        sanitized_ext = cls._sanitize_extension(extension)
        return cls.__registry.get(sanitized_ext, None)

    @staticmethod
    def _sanitize_extension(extension: str) -> str:
        """
        Sanitizes a file extension by ensuring it starts with a dot and is lowercase.

        :param extension: The file extension to sanitize.
        :return: The sanitized file extension.
        """
        sanitized = extension.lower()
        if not sanitized.startswith("."):
            sanitized = "." + sanitized
        return sanitized


_NUMPY_EXTENSIONS = {
    ".npy",
}


@_ImagingLoaderRegistry.register(*_NUMPY_EXTENSIONS)
def _numpy_loader(file: Path) -> np.ndarray:
    """
    Loads a numpy file as an array.

    :param file: The path to the numpy file.
    :return: The loaded numpy array.
    """
    return np.load(file, allow_pickle=False, mmap_mode="r")


_OPENCV_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".mp4",
    ".avi",
    ".mov",
    ".bmp",
}


@_ImagingLoaderRegistry.register(*_OPENCV_EXTENSIONS)
def _opencv_loader(path: Path) -> np.ndarray:
    """
    Loads an image file using OpenCV.

    :param path: The path to the image file.
    :return: The loaded image as a numpy array.
    """
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


_TIFFFILE_EXTENSIONS = {".tiff", ".ome.tiff", ".tif"}


@_ImagingLoaderRegistry.register(*_TIFFFILE_EXTENSIONS)
def _tifffile_loader(file: Path) -> np.ndarray:
    """
    Loads a TIFF file using the tifffile library.

    :param file: The path to the TIFF file.
    :return: The loaded image as a numpy array.
    """
    return imread(file, mode="r")


def load_images(file: str | PathLike) -> np.ndarray:
    """
    Loads an image file using the appropriate loader based on its extension.

    :param file: The path to the image file.
    :return: The loaded image as a numpy array.
    :raises ValueError: If the file path is invalid.
    :raises FileFormatError: If no loader is registered for the file extension.
    """
    try:
        file = Path(file)
        ext = file.suffix
    except (AttributeError, TypeError) as exc:
        msg = f"{file} is not a valid file path."
        raise ValueError(msg) from exc

    loader = _ImagingLoaderRegistry().get_loader(ext)
    if loader is None:
        raise FileFormatError(file)

    return loader(file)


"""
|=======================================================================================|
|DISPATCHING & REGISTRY OF ROI HANDLERS
|=======================================================================================|
"""


@runtime_checkable
class ROIHandler(Protocol):
    """
    Protocol for ROI handler functions.

    :param source: The source path to load ROIs from.
    :return: A list of ROI objects.
    """

    def __call__(
        self,
        source: Path,
    ) -> list[ROI]: ...


# FEATURE: Add CaImAnHandler


class Suite2PHandler:
    """
    Handler for loading ROIs from Suite2P output files.

    :note: Suite2P stores ROI information in numpy files containing pickled data structures.
    """

    def __call__(
        self,
        source: Path,
    ) -> list[ROI]:
        """
        Loads ROIs from Suite2P output files.

        :param source: The path to the Suite2P output directory.
        :return: A list of ROI objects.
        :warns PickleWarning: Warns about the use of pickled numpy files.
        """
        # NOTE: As of 12-28-2025, Suite2P still uses numpy files containing pickled data structures
        #  to store ROI information.
        msg = "\nSuite2P stores ROI information in pickled numpy files.\n"
        warn(msg, PickleWarning, stacklevel=3)

        stat = Suite2PHandler._load_stat_file(source)
        image_shape = Suite2PHandler._load_image_shape(source)
        if (neuron_index := self._load_neuron_index(source)) is None:
            neuron_index = np.ones(len(stat), dtype=bool)
        return Suite2PHandler._build_rois(stat, neuron_index, image_shape)

    @staticmethod
    def _build_rois(
        stat: np.ndarray, neuron_index: np.ndarray, image_shape: np.ndarray
    ) -> list[ROI]:
        """
        Builds ROI objects from Suite2P stat data.

        :param stat: The stat data from Suite2P.
        :param neuron_index: A boolean array indicating valid neurons.
        :param image_shape: The shape of the image.
        :return: A list of ROI objects.
        """
        rois = []
        for idx, nrn in enumerate(stat):
            if not neuron_index[idx]:
                continue
            pixels = np.asarray([nrn["ypix"], nrn["xpix"]]).T
            weight = nrn["lam"]
            roi = ROI(pixels, weight, image_shape)
            rois.append(roi)
        return rois

    @staticmethod
    def _load_neuron_index(source: Path) -> np.ndarray | None:
        """
        Loads the neuron index file from Suite2P output.

        :param source: The path to the Suite2P output directory.
        :return: A boolean array indicating valid neurons, or None if the file does not exist.
        :warns UserWarning: Warns if the neuron index file is missing.
        """
        iscell_file = source.joinpath("iscell.npy")
        if not iscell_file.exists():
            msg = f"{iscell_file.name} does not exist. Assuming all ROIs are neurons."
            warn(msg, UserWarning, stacklevel=2)
            return None
        iscell = np.load(iscell_file, allow_pickle=True)
        return iscell[:, 0] == 1

    @staticmethod
    def _load_image_shape(source: Path) -> tuple[int, int]:
        """
        Loads the image shape from Suite2P output.

        :param source: The path to the Suite2P output directory.
        :return: A tuple representing the image shape (height, width).
        :raises FileNotFoundError: If the ops.npy file does not exist.
        """
        ops_file = source.joinpath("ops.npy")
        if not ops_file.exists():
            msg = f"{ops_file.name} does not exist."
            raise FileNotFoundError(msg)
        ops = np.load(ops_file, allow_pickle=True).item()
        return ops["Ly"], ops["Lx"]

    @staticmethod
    def _load_stat_file(source: Path) -> np.ndarray:
        """
        Loads the stat file from Suite2P output.

        :param source: The path to the Suite2P output directory.
        :return: The stat data as a numpy array.
        :raises FileNotFoundError: If the stat.npy file does not exist.
        """
        stat_file = source.joinpath("stat.npy")
        if not stat_file.exists():
            msg = f"{stat_file.name} does not exist."
            raise FileNotFoundError(msg)
        return np.load(stat_file, allow_pickle=True)


def validate_roi_handler(handler: ROIHandler) -> ROIHandler:
    """
    Validates that the provided handler meets the ROIHandler protocol.

    :param handler: The handler to validate.
    :return: The validated handler.
    :raises TypeError: If the handler does not meet the ROIHandler protocol.
    """
    # NOTE: We do this to ensure that an instance of a handler is being used, not the class itself
    #  (1) This allows for handlers that may have internal state in the future
    #  (2) Since functions are instances of function, this check will pass if the user provides a function
    #  instead of a class instance or definition
    if not isinstance(handler, ROIHandler):
        msg = "handler must be a callable that meets the requirements of an ROIHandler."
        raise TypeError(msg)
    if isinstance(handler, type):
        # noinspection PyArgumentList
        return handler()
    return handler


def load_rois(
    source: str | PathLike,
    handler: ROIHandler,
) -> list[ROI]:
    """
    Loads ROIs from a source using the specified handler.

    :param source: The path to the source.
    :param handler: The ROI handler to use.
    :return: A list of ROI objects.
    :raises ValueError: If the source path is invalid.
    """
    try:
        source = Path(source)
    except (AttributeError, TypeError) as exc:
        msg = f"{source} is not a valid file path."
        raise ValueError(msg) from exc
    handler = validate_roi_handler(handler)
    return handler(source)
