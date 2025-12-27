from functools import partial
from itertools import cycle
from typing import NamedTuple, Protocol, runtime_checkable

from matplotlib.colors import LinearSegmentedColormap


class Color(NamedTuple):
    r: float
    g: float
    b: float
    a: float = 1.0


@runtime_checkable
class ColorGroup(Protocol):
    colors: tuple[Color, ...]

    @classmethod
    def cycle(cls):
        colors = cycle(cls.colors)
        yield partial(cls.mapper, next(colors))

    @staticmethod
    def mapper(color, msg):
        print(f"{msg}: using color {color}")
        return partial(
            LinearSegmentedColormap.from_list, f"{color}_map", [(1.0, 1.0, 1.0), color]
        )

    @property
    def num_colors(self) -> int:
        return len(self.colors)


class Rainbow(ColorGroup):
    RED = Color(255 / 255, 62 / 255, 65 / 255)
    ORANGE = Color(255 / 255, 138 / 255, 67 / 255)
    YELLOW = Color(255 / 255, 235 / 255, 127 / 255)
    GREEN = Color(0.0, 201 / 255, 167 / 255)
    BLUE = Color(0.0, 126 / 255, 167 / 255)

    colors = tuple([RED, ORANGE, YELLOW, GREEN, BLUE])
