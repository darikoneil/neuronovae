from os import PathLike
from pathlib import Path

"""
This module contains some custom aliases for type-hinting
"""

#: A valid file path as a string, Path or PathLike object
File = str | Path | PathLike
