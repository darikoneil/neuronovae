from neuronovae.constants import Constants
from neuronovae.hints import File

"""
This module contains custom exceptions for neuronovae.
"""

__all__ = [
    "FileFormatError",
    "RGBWarning",
]


class FileFormatError(RuntimeError):
    """
    Exception raised when the input file is not in the expected format.

    :param file: The file that caused the error.
    """

    def __init__(self, file: File):
        self.file = file
        super().__init__(
            f"Unsupported file format: {file.suffix}. "
            f"Expected one of: {Constants.SUPPORTED_FILE_FORMATS}"
        )


class RGBWarning(UserWarning):
    """
    Warning raised when a loaded image is likely RGB, BGR, RGBA, or equivalent but
    neuronovae expects grayscale images.

    :param file: The file that caused the warning.
    :param dimensions: The dimensions of the image.
    """

    def __init__(self, file: File, dimensions: tuple[int, ...]):
        self.file = file
        self.dimensions = dimensions
        super().__init__(
            f"Image {file.name} likely has color channels. "
            f"The final dimension is between 3 and 4 "
            f"({self.dimensions[-1]})."
        )
