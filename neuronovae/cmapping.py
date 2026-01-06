from collections.abc import Iterable
# from typing import NamedTuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pydantic import field_validator
from pydantic.config import ConfigDict
from pydantic.dataclasses import dataclass
from pydantic import Field

# FEATURE: Add functionality for more complex instructions

"""
Module for standardizing instructions when coloring images.
"""

@dataclass(frozen=True)
class Color:
    """
    Represents an RGBA color normalized to the 0-1 range.

    Attributes:
        r: Red channel
        g: Green channel
        b: Blue channel
        a: Alpha channel

    Note:
        The alpha channel defaults to 1.0 (fully opaque) if not specified.
    """

    r: float = Field(ge=0, le=1.0)
    g: float = Field(ge=0, le=1.0)
    b: float = Field(ge=0, le=1.0)
    a: float = Field(default=1.0, ge=0, le=1.0)

    def __iter__(self) -> Iterable[float]:
        """
        Allow unpacking of Color instance into RGBA components.

        Returns:
            An iterator over the RGBA components.
        """
        return iter((self.r, self.g, self.b, self.a))

    def __len__(self) -> int:
        """
        Return the number of components in the Color instance.

        Returns:
            The number of components (4 for RGBA).
        """
        return 4

    @classmethod
    def from_hex(cls, hex_str: str) -> "Color":
        """
        Create a Color instance from a hexadecimal color string.

        Args:
            hex_str: Hexadecimal color string (e.g., "#RRGGBB" or "#RRGGBBAA").

        Returns:
            A Color instance representing the given hexadecimal color.

        Raises:
            ValueError: If the hex string is not in the correct format.
        """
        hex_str = hex_str.lstrip("#")
        length = len(hex_str)
        if length == 6:
            r, g, b = (
                int(hex_str[0:2], 16),
                int(hex_str[2:4], 16),
                int(hex_str[4:6], 16),
            )
            a = None
            # NOTE: Leave this as None to avoid hardcoding a default alpha value
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
        return cls(
            r / 255.0, g / 255.0, b / 255.0, a / 255.0 if a is not None else None
        )

    @classmethod
    def from_rgba(cls, r: float, g: float, b: float, a: float = 255.0) -> "Color":
        """
        Create a Color instance from an RGB or RGBA tuple with values in 0-255 range.

        Args:
            r: Red channel
            g: Green channel
            b: Blue channel
            a: Alpha channel

        Returns:
            A Color instance representing the given RGB or RGBA values.

        Raises:
            ValueError: If the tuple does not have 3 or 4 elements.
        """
        return cls(*(channel / 255.0 for channel in (r, g, b, a)))


class ColorMap:
    """
    Represents a colormap for mapping values to colors. Wraps matplotlib's
    [`LinearSegmentedColorMap`][matplotlib.colors.LinearSegmentedColormap].

    Attributes:
        colors: Array of Color instances defining each equidistant point in the colormap (inclusive of endpoints).
        num_colors: The number of colors in the colormap.
        _mapping: LinearSegmentedColormap instance for mapping values.
    """

    def __init__(self, colors: Iterable[Color]):
        """
        Each color provided defines an equidistant point in the colormap (inclusive of endpoints).

        Args:
            colors: The colors that define the colormap.
        """
        self.colors: tuple[Color, ...] = tuple([Color(*color) for color in colors])
        self.num_colors: int = len(colors)
        self._mapping: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
            "custom_colormap", colors
        )

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """
        Map an array of values to their corresponding colors.

        Args:
            values: Array of values to map.

        Returns:
            values: Array of mapped colors.
        """
        return self._mapping(values)


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class ColorInstruction:
    """
    Represents an instruction for applying a colormap to specific indices.

    Attributes:
        cmap: ColorMap instance to use for mapping.
        indices: Array of indices to which the colormap is applied.
    """

    cmap: ColorMap
    indices: np.ndarray

    def __call__(self) -> tuple[ColorMap, np.ndarray]:
        """
        Retrieve the colormap and indices.

        Returns:
            The colormap and indices.
        """
        return self.cmap, self.indices

    @field_validator("indices", mode="after")
    @classmethod
    def validate_indices(cls, v: np.ndarray) -> np.ndarray:
        """
        Validate the indices array.

        Args:
            v: Array of indices to validate.

        Returns:
            Validated array of indices.

        Raises:
            ValueError: If the indices are not a 1D array or do not contain integers.
        """
        if v.ndim > 1:
            msg = "Indices must be a 1D array."
            raise ValueError(msg)
        if not np.issubdtype(v.dtype, np.integer):
            msg = "Indices must have integer values."
            raise ValueError(msg)
        return v
