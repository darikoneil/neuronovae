import numpy as np
from typing import Any
from functools import cached_property
from dataclasses import dataclass
import polars as pl


@dataclass(frozen=True)
class Activity:
    raster: np.ndarray
    timestamps: np.ndarray | None = None

    @cached_property
    def components(self) -> int:
        return self.raster.shape[0]

    @cached_property
    def frames(self) -> int:
        return self.raster.shape[-1]


@dataclass(frozen=True)
class Background:
    """
    Background for things to be plotted onto
    """
    image: np.ndarray
    timestamps: np.ndarray | None = None

    @cached_property
    def frames(self) -> int | None:
        return self.image.shape[0] if self.image.ndim >= 3 else None

    @cached_property
    def planes(self) -> int | None:
        return self.image.shape[1] if self.image.ndim == 4 else None

    @cached_property
    def height(self) -> int:
        return self.image.shape[-2]

    @cached_property
    def width(self) -> int:
        return self.image.shape[-1]

    @cached_property
    def min(self) -> float:
        return np.nanmin(self.image)

    @cached_property
    def max(self) -> float:
        return np.nanmax(self.image)


@dataclass(frozen=True)
class Features:

    features: pl.DataFrame | None = None

    timestamps: np.ndarray | None = None
