from pathlib import Path
from typing import Protocol, runtime_checkable, Any
import cv2
from tifffile import imread
import numpy as np
from neuronovae.errors import FileFormatError
from os import PathLike

__all__ = [
    "load_images",
]

"""
|=======================================================================================|
|DISPATCHING & REGISTRY OF IMAGING LOADERS
|=======================================================================================|
"""


@runtime_checkable
class ImagingLoader(Protocol):
    def __call__(self, file: Path) -> Any: ...


class _ImagingLoaderRegistry:
    __registry: dict[str, ImagingLoader] = {}

    @classmethod
    def register(cls, *extension: str):
        def decorator(func) -> ImagingLoader:
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
        raise ValueError(f"{file} is not a valid file path.") from exc

    loader = _ImagingLoaderRegistry().get_loader(ext)
    if loader is None:
        raise FileFormatError(file)

    return loader(file)
