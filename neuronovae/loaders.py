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
    def __call__(self, file: Path) -> Any: ...


class _ImagingLoaderRegistry:
    __registry: ClassVar[dict[str, ImagingLoader]] = {}

    @classmethod
    def register(cls, *extension: str) -> ImagingLoader:
        def decorator(func: ImagingLoader) -> ImagingLoader:
            nonlocal extension
            for ext in extension:
                sanitized_ext = cls._sanitize_extension(ext)
                cls.__registry[sanitized_ext] = func
            return func

        return decorator

    @classmethod
    def get_loader(cls, extension: str) -> ImagingLoader | None:
        # NOTE: We're returning None here to propogate the error up to the caller
        sanitized_ext = cls._sanitize_extension(extension)
        return cls.__registry.get(sanitized_ext, None)

    @staticmethod
    def _sanitize_extension(extension: str) -> str:
        sanitized = extension.lower()
        if not sanitized.startswith("."):
            sanitized = "." + sanitized
        return sanitized


_NUMPY_EXTENSIONS = {
    ".npy",
}


@_ImagingLoaderRegistry.register(*_NUMPY_EXTENSIONS)
def _numpy_loader(file: Path) -> np.ndarray:
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
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


_TIFFFILE_EXTENSIONS = {".tiff", ".ome.tiff", ".tif"}


@_ImagingLoaderRegistry.register(*_TIFFFILE_EXTENSIONS)
def _tifffile_loader(file: Path) -> np.ndarray:
    return imread(file, mode="r")


def load_images(file: str | PathLike) -> np.ndarray:
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
    def __call__(
        self,
        source: Path,
    ) -> list[ROI]: ...


class Suite2PHandler:
    def __call__(
        self,
        source: Path,
    ) -> list[ROI]:
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
        iscell_file = source.joinpath("iscell.npy")
        if not iscell_file.exists():
            msg = f"{iscell_file.name} does not exist. Assuming all ROIs are neurons."
            warn(msg, UserWarning, stacklevel=2)
            return None
        iscell = np.load(iscell_file, allow_pickle=True)
        return iscell[:, 0] == 1

    @staticmethod
    def _load_image_shape(source: Path) -> tuple[int, int]:
        ops_file = source.joinpath("ops.npy")
        if not ops_file.exists():
            msg = f"{ops_file.name} does not exist."
            raise FileNotFoundError(msg)
        ops = np.load(ops_file, allow_pickle=True).item()
        return ops["Ly"], ops["Lx"]

    @staticmethod
    def _load_stat_file(source: Path) -> np.ndarray:
        stat_file = source.joinpath("stat.npy")
        if not stat_file.exists():
            msg = f"{stat_file.name} does not exist."
            raise FileNotFoundError(msg)
        return np.load(stat_file, allow_pickle=True)


# TODO: Add CaImAnHandler
def validate_roi_handler(handler: ROIHandler) -> ROIHandler:
    if not isinstance(handler, ROIHandler):
        msg = "handler must be a callable that meets the requirements of an ROIHandler."
        raise TypeError(msg)
    # NOTE: We do this to ensure that an instance of a handler is being used, not the class itself
    #  (1) This allows for handlers that may have internal state in the future
    #  (2) Since functions are instances of function, this check will pass if the user provides a function
    #  instead of a class instance or definition
    if isinstance(handler, type):
        # noinspection PyArgumentList
        return handler()
    return handler


def load_rois(
    source: str | PathLike,
    handler: ROIHandler,
) -> list[ROI]:
    try:
        source = Path(source)
    except (AttributeError, TypeError) as exc:
        msg = f"{source} is not a valid file path."
        raise ValueError(msg) from exc
    handler = validate_roi_handler(handler)
    return handler(source)
