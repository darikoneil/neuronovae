from typing import NamedTuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pydantic import field_validator
from pydantic.config import ConfigDict
from pydantic.dataclasses import dataclass

# FEATURE: Add functionality for more complex instructions

"""
Module for standardizing instructions when coloring images.
"""


class Color(NamedTuple):
    """
    Represents an RGBA color normalized to the 0-1 range.

    :ivar r: Red channel (0.0-1.0).
    :ivar g: Green channel (0.0-1.0).
    :ivar b: Blue channel (0.0-1.0).
    :ivar a: Alpha channel (0.0-1.0), default is 1.0.
    """

    r: float
    g: float
    b: float
    a: float = 1.0

    @classmethod
    def from_hex(cls, hex_str: str) -> "Color":
        """
        Create a Color instance from a hexadecimal color string.

        :param hex_str: Hexadecimal color string (e.g., "#RRGGBB" or "#RRGGBBAA").
        :return: A Color instance representing the given hexadecimal color.
        :raises ValueError: If the hex string is not in the correct format.
        """
        hex_str = hex_str.lstrip("#")
        length = len(hex_str)
        if length == 6:
            r, g, b = (
                int(hex_str[0:2], 16),
                int(hex_str[2:4], 16),
                int(hex_str[4:6], 16),
            )
            a = 255
        elif length == 8:
            r, g, b, a = (
                int(hex_str[0:2], 16),
                int(hex_str[2:4], 16),
                int(hex_str[4:6], 16),
                int(hex_str[6:8], 16),
            )
        else:
            msg = "Hex string must be in the format #RRGGBB or #RRGGBBAA"
            raise ValueError(msg)
        return cls(r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    @classmethod
    def from_rgba(cls, rgba: tuple[float, float, float, ...]) -> "Color":
        """
        Create a Color instance from an RGBA tuple with values in 0-255 range.

        :param rgba: Tuple of RGBA values (0-255).
        :return: A Color instance representing the given RGBA values.
        :raises ValueError: If the tuple does not have 3 or 4 elements.
        """
        if not 3 <= len(rgba) <= 4:
            msg = "RGBA tuple must have 3 or 4 elements."
            raise ValueError(msg)
        return cls(*(channel / 255.0 for channel in rgba))


class ColorMap:
    """
    Represents a colormap for mapping values to colors.

    :ivar colors: Array of Color instances defining the colormap.
    :ivar num_colors: Number of colors in the colormap.
    :ivar _mapping: LinearSegmentedColormap instance for mapping values.
    """

    def __init__(self, colors: tuple[Color, ...]):
        """
        Initialize the ColorMap with a sequence of Color instances.

        :param colors: Tuple of Color instances defining the colormap.
        """
        self.colors = np.asarray([Color(*color) for color in colors])
        self.num_colors = len(colors)
        self._mapping = LinearSegmentedColormap.from_list("custom_colormap", colors)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """
        Map an array of values to their corresponding colors.

        :param values: Array of values to map.
        :return: Array of mapped colors.
        """
        return self._mapping(values)


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class ColorInstruction:
    """
    Represents an instruction for applying a colormap to specific indices.

    :ivar cmap: ColorMap instance to use for mapping.
    :ivar indices: Array of indices to which the colormap is applied.
    """

    cmap: ColorMap
    indices: np.ndarray

    def __call__(self) -> tuple[ColorMap, np.ndarray]:
        """
        Retrieve the colormap and indices.

        :return: Tuple containing the colormap and indices.
        """
        return self.cmap, self.indices

    @field_validator("indices", mode="after")
    @classmethod
    def validate_indices(cls, v: np.ndarray) -> np.ndarray:
        """
        Validate the indices array.

        :param v: Array of indices to validate.
        :return: Validated array of indices.
        :raises ValueError: If the indices are not a 1D array or do not contain integers.
        """
        if v.ndim > 1:
            msg = "Indices must be a 1D array."
            raise ValueError(msg)
        if not np.issubdtype(v.dtype, np.integer):
            msg = "Indices must have integer values."
            raise ValueError(msg)
        return v
