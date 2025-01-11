from dataclasses import dataclass

"""
This class operates as a namespace containing (immutable) constants used in the
package--since we don't really have that option in python. I figure it's better to
make sure this constants are well-encapsulated and immutable than having them lose in
the module namespace.
"""


__all__ = [
    "Constants",
]


@dataclass(frozen=True)
class Constants:
    """
    Constants used by Neuronovae.

    :cvar SUPPORTED_4D_EXTENSIONS: Neuronovae supports loading 4D imaging files in the
        following formats.

    :cvar SUPPORTED_3D_EXTENSIONS: Neuronovae supports loading 3D imaging files in the
        following formats

    :cvar SUPPORTED_2D_EXTENSIONS: Neuronovae supports loading 2D imaging files in the
        the following formats.

    :cvar SUPPORTED_FILE_FORMATS: Neuronovae supports loading imaging files in the
        following formats.

    :cvar NUMPY_FILE_FORMATS: Neuronovae supports loading numpy files in the following
        formats.

    :cvar TIFFFILE_FILE_FORMATS: Neuronovae supports uses tifffile to load images in
        the following formats.

    :cvar OPENCV_FILE_FORMATS: Neuronovae uses OpenCV to load images in the following
        formats.

    .. tip:: It's possible some other niche formats are supported. Neuronovae will
        attempt to load unknown file formats with the tifffile package before raising
        an error.
    """

    SUPPORTED_4D_EXTENSIONS = {".npy", ".mmap"}
    SUPPORTED_3D_EXTENSIONS = {
        ".tiff",
        ".ome.tiff",
        ".tif",
        ".gif",
        ".mp4",
        ".avi",
        ".mov",
        ".npy",
        ".mmap",
    }
    SUPPORTED_2D_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".ome.tiff",
        ".tif",
        ".npy",
        ".mmap",
    }
    SUPPORTED_FILE_FORMATS = (
        SUPPORTED_2D_EXTENSIONS | SUPPORTED_3D_EXTENSIONS | SUPPORTED_4D_EXTENSIONS
    )
    NUMPY_FILE_FORMATS = {".npy", ".mmap"}
    TIFFFILE_FILE_FORMATS = {".tiff", ".ome.tiff", ".tif"}
    OPENCV_FILE_FORMATS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".mp4",
        ".avi",
        ".mov",
        ".bmp",
    }
