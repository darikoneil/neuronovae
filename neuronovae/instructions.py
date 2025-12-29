import numpy as np
from pydantic.config import ConfigDict
from pydantic.dataclasses import dataclass

from neuronovae.cmaps import ColorMap

_configuration = ConfigDict(
    validate_assignment=True,
    arbitrary_types_allowed=True,
)


# TODO: Implement
@dataclass(config=_configuration)
class ColorInstruction:
    cmap: ColorMap
    indices: np.ndarray

    def __call__(self) -> tuple[ColorMap, np.ndarray]:
        return self.cmap, self.indices
