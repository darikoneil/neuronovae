from os import PathLike


class FileFormatError(ValueError):
    """
    Raised when an input file is not in the expected format.

    Args:
        file: The (ideally) path-like object that caused the error.
    """

    def __init__(self, file: str | PathLike):
        self.file = file
        super().__init__(f"Unsupported file format: ({file}).")


class RBGFormatError(ValueError):
    """
    Raised when a presumed video frame is not in RGB format.

    Args:
        shape: The shape of the offending array.
    """

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        super().__init__(f"Expected data in RBG format, got shape: {shape}.")


class PickleWarning(RuntimeWarning):
    """
    Warning about loading pickled files.

    Args:
        msg: Optional custom warning message. A safety note will be appended.
    """

    def __init__(self, msg: str | None = None):
        msg = msg or ""
        msg += "Loading pickled files can be unsafe. Ensure the source is trusted."
        super().__init__(msg)
