from typing import NamedTuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pydantic import field_validator
from pydantic.config import ConfigDict
from pydantic.dataclasses import dataclass

# FEATURE: Add functionality for more complex instructions


class Color(NamedTuple):
    r: float
    g: float
    b: float
    a: float = 1.0
# DOCME: Color

class ColorMap:
    def __init__(self, colors: tuple[Color, ...]):
        self.colors = np.asarray([Color(*color) for color in colors])
        self.num_colors = len(colors)
        self._mapping = LinearSegmentedColormap.from_list("custom_colormap", colors)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return self._mapping(values)
# DOCME: ColorMap


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class ColorInstruction:
    cmap: ColorMap
    indices: np.ndarray

    def __call__(self) -> tuple[ColorMap, np.ndarray]:
        return self.cmap, self.indices

    @field_validator("indices", mode="after")
    @classmethod
    def validate_indices(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim > 1:
            msg = "Indices must be a 1D array."
            raise ValueError(msg)
        if not np.issubdtype(v, np.integer):
            msg = "Indices must have integer values."
            raise ValueError(msg)
        return v
# DOCME: ColorInstruction