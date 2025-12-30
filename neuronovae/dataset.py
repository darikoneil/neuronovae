import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
from pydantic import Field, field_validator, model_validator
from pydantic.config import ConfigDict
from pydantic.dataclasses import dataclass

from neuronovae.cmapping import ColorInstruction
from neuronovae.rois import ROI

_configuration = ConfigDict(
    validate_assignment=True,
    arbitrary_types_allowed=True,
)


@dataclass(config=_configuration)
class Dataset:
    images: np.ndarray = Field(title="Images")
    rois: list[ROI] = Field(title="ROIs", min_length=1)
    instructions: list[ColorInstruction] = Field(
        title="Color Instructions", min_length=1
    )

    @field_validator("images", mode="after")
    @classmethod
    def check_images_ndim(cls, v: np.ndarray) -> np.ndarray:
        if not 1 < v.ndim < 4:
            msg = "Images must be 2D (Y, X) or 3D (Frame, Y, X) numpy arrays."
            raise ValueError(msg)
        return v

    @field_validator("instructions", mode="before")
    @classmethod
    def coerce_instructions_to_list(cls, v: Any) -> list[Any]:
        if isinstance(v, ColorInstruction):
            return [v]
        return v

    @model_validator(mode="after")
    def check_instruction_indices(self) -> "Dataset":
        num_rois = len(self.rois)
        for instruction in self.instructions:
            if np.any((instruction.indices >= num_rois) | (instruction.indices < 0)):
                msg = "Instruction indices outside of ROI bounds."
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_rois_in_bounds(self) -> "Dataset":
        bounds = self.images.shape[-2] * self.images.shape[-1]
        for roi in self.rois:
            if roi.index.min() < 0 or roi.index.max() >= bounds:
                msg = "ROI indices outside of image bounds."
                raise ValueError(msg)
        return self


def validate_dataset(func: Callable) -> Callable:
    @wraps(func)
    def decorator(*args, **kwargs) -> Callable:
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        bound_args.arguments.pop("kwargs", None)
        container = bound_args.arguments
        params = {
            key: container.get(key)
            for key in Dataset.__dataclass_fields__
            if key in container
        }
        valid_args = Dataset(**params)
        return func(**{**bound_args.arguments, **vars(valid_args)})

    return decorator
