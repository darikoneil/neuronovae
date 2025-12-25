from os import PathLike


class FileFormatError(ValueError):
    """
    Exception raised when the input file is not in the expected format.

    :param file: The file that caused the error.
    """

    def __init__(self, file: str | PathLike):
        self.file = file
        super().__init__(f"Unsupported file format: ({file}).")


class RBGFormatError(ValueError):
    """
    Exception raised when the input video frame is not in RBG format.

    :param shape: The shape of the video frame that caused the error.
    """

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        super().__init__(f"Expected data in RBG format, got shape: {shape}.")
