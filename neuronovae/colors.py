from typing import Protocol, runtime_checkable, NamedTuple
from matplotlib.colors import LinearSegmentedColormap
from itertools import cycle


class Color(NamedTuple):
    r: float
    g: float
    b: float
    a: float = 1.0


@runtime_checkable
class ColorGroup(Protocol):
    colormaps: list[LinearSegmentedColormap | None]

    def __init__(self, background: np.ndarray) -> None: ...

    @staticmethod
    def build_map(name: str, colors: list[Color]) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(name, colors)

    @classmethod
    def get_iter(cls) -> iter:
        return cycle(cls.colormaps)
