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
    Protocol for imaging loader callables.

    Args:
        file: Path to load.

    Returns:
        Any: Loaded image data.
    """

    def __call__(self, file: Path) -> Any: ...


class _ImagingLoaderRegistry:
    """
    Registry that maps file extensions to loader callables.

    Note:
        Use `register` to associate extensions with loader functions and `get_loader` to retrieve them.
    """

    __registry: ClassVar[dict[str, ImagingLoader]] = {}

    @classmethod
    def register(cls, *extension: str) -> ImagingLoader:
        """
        Decorator to register a loader for one or more file extensions.

        Args:
            *extension: File extensions to register for (e.g. ".png", ".tif").

        Returns:
            Callable: Decorator that registers the function.
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
        Retrieve the loader for the given extension.

        Args:
            extension: File extension to look up.

        Returns:
            ImagingLoader or None: Registered loader or None if not found.
        """
        # NOTE: We're returning None here to propogate the error up to the caller
        sanitized_ext = cls._sanitize_extension(extension)
        return cls.__registry.get(sanitized_ext, None)

    @staticmethod
    def _sanitize_extension(extension: str) -> str:
        """
        Ensure extension starts with a dot and is lowercase.

        Args:
            extension: Input file extension.

        Returns:
            str: Sanitized extension.
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
    Load a NumPy (.npy) file.

    Args:
        file: Path to .npy file.

    Returns:
        numpy.ndarray: Loaded array.
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
    Load an image using OpenCV.

    Args:
        path: Path to image file.

    Returns:
        numpy.ndarray: Image array as returned by OpenCV.
    """
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


_TIFFFILE_EXTENSIONS = {".tiff", ".ome.tiff", ".tif"}


@_ImagingLoaderRegistry.register(*_TIFFFILE_EXTENSIONS)
def _tifffile_loader(file: Path) -> np.ndarray:
    """
    Load a TIFF/Ome-TIFF file using tifffile.

    Args:
        file: Path to TIFF file.

    Returns:
        numpy.ndarray: Loaded image array.
    """
    return imread(file, mode="r")


def load_images(file: str | PathLike) -> np.ndarray:
    """
    Load images using the registered loader for the file extension.

    Args:
        file: Path to the image file.

    Returns:
        numpy.ndarray: Loaded image data.

    Raises:
        ValueError: If the provided path is invalid.
        FileFormatError: If no loader is registered for the file extension.
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
    Protocol for ROI handler callables.

    Args:
        source: Path to ROI source.

    Returns:
        list[ROI]: Loaded ROIs.
    """

    def __call__(
        self,
        source: Path,
    ) -> list[ROI]: ...


# FEATURE: Add CaImAnHandler


class Suite2PHandler:
    """
    Handler for loading ROIs from Suite2P outputs.

    Note:
        Suite2P stores ROI metadata in numpy files that may contain pickled objects.
    """

    def __call__(self, source: Path) -> list[ROI]:
        """
        Load ROIs from a Suite2P output directory.

        Args:
            source: Path to Suite2P output directory.

        Returns:
            list[ROI]: List of ROI objects.

        Warns:
            PickleWarning: About pickled numpy files.
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
        Build ROI objects from Suite2P stat structures.

        Args:
            stat: Array-like Suite2P stat entries.
            neuron_index: Boolean array indicating which entries are neurons.
            image_shape: Tuple (height, width) of the image.

        Returns:
            list[ROI]: Constructed ROI objects.
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
        Load iscell.npy if present to identify neurons.

        Args:
            source: Suite2P output directory.

        Returns:
            numpy.ndarray or None: Boolean array marking cells, or None if missing.

        Warns:
            UserWarning: If iscell.npy is missing (assume all ROIs are neurons).
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
        Load image shape from ops.npy.

        Args:
            source: Suite2P output directory.

        Returns:
            tuple[int, int]: (height, width)

        Raises:
            FileNotFoundError: If ops.npy is missing.
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
        Load stat.npy.

        Args:
            source: Suite2P output directory.

        Returns:
            numpy.ndarray: The stat array.

        Raises:
            FileNotFoundError: If stat.npy is missing.
        """
        stat_file = source.joinpath("stat.npy")
        if not stat_file.exists():
            msg = f"{stat_file.name} does not exist."
            raise FileNotFoundError(msg)
        return np.load(stat_file, allow_pickle=True)


def validate_roi_handler(handler: ROIHandler) -> ROIHandler:
    """
    Ensure the handler satisfies the ROIHandler protocol and return an instance.

    Args:
        handler: Callable or class to handle ROIs.

    Returns:
        ROIHandler: An instance that implements the protocol.

    Raises:
        TypeError: If the provided handler is not compatible.
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
    Load ROIs from a given source using the provided handler.

    Args:
        source: Path to ROI source.
        handler: Handler to use for loading.

    Returns:
        list[ROI]: Loaded list of ROI objects.

    Raises:
        ValueError: If the provided source path is invalid.
    """
    try:
        source = Path(source)
    except (AttributeError, TypeError) as exc:
        msg = f"{source} is not a valid file path."
        raise ValueError(msg) from exc
    handler = validate_roi_handler(handler)
    return handler(source)
