from typing import NamedTuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class Color(NamedTuple):
    r: float
    g: float
    b: float
    a: float = 1.0


WHITE: Color = Color(
    1.0,
    1.0,
    1.0,
)

BLACK: Color = Color(
    0.0,
    0.0,
    0.0,
)

RED: Color = Color(
    255 / 255,
    62 / 255,
    65 / 255,
)

ORANGE: Color = Color(
    255 / 255,
    138 / 255,
    67 / 255,
)

YELLOW: Color = Color(
    255 / 255,
    235 / 255,
    127 / 255,
)

GREEN: Color = Color(
    0.0,
    201 / 255,
    167 / 255,
)

BLUE: Color = Color(
    0.0,
    126 / 255,
    167 / 255,
)


# TODO: Assess performance optimizations
class ColorMap:
    def __init__(self, colors: tuple[Color, ...]):
        self.colors = np.asarray([Color(*color) for color in colors])
        self.num_colors = len(colors)
        self._mapping = LinearSegmentedColormap.from_list("custom_colormap", colors)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return self._mapping(values)
