from typing import NamedTuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pydantic.config import ConfigDict
from pydantic.dataclasses import dataclass


# TODO: Add functionality for more complex instructions


class Color(NamedTuple):
    r: float
    g: float
    b: float
    a: float = 1.0


# TODO: Assess performance optimizations (if necessary)
class ColorMap:
    def __init__(self, colors: tuple[Color, ...]):
        self.colors = np.asarray([Color(*color) for color in colors])
        self.num_colors = len(colors)
        self._mapping = LinearSegmentedColormap.from_list("custom_colormap", colors)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return self._mapping(values)


_configuration = ConfigDict(
    validate_assignment=True,
    arbitrary_types_allowed=True,
)


@dataclass(config=_configuration)
class ColorInstruction:
    cmap: ColorMap
    indices: np.ndarray

    def __call__(self) -> tuple[ColorMap, np.ndarray]:
        return self.cmap, self.indices
