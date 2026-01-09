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


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class Dataset:
    """
    Container for images, ROIs, and color instructions.

    Args:
        images: Numpy array of images (2D image or 3D stack).
        rois: List of ROI objects.
        instructions: List of ColorInstruction objects.
        scaling: Tuple of percentiles for baseline/bound (default (10.0, 90.0)).
        chunk_size: Frames per chunk or None.

    Raises:
        ValueError: If validation checks fail.

    Example:
        ```python
        dataset = Dataset(
            images=np.random.rand(10, 256, 256),
            rois=[ROI(...)],
            instructions=[ColorInstruction(...)],
            scaling=(10.0, 90.0),
            chunk_size=5
        )
        ```
    """

    images: np.ndarray = Field(title="Images")
    rois: list[ROI] = Field(title="ROIs", min_length=1)
    instructions: list[ColorInstruction] = Field(
        title="Color Instructions", min_length=1
    )
    scaling: tuple[float, float] = Field(
        title="Scaling Percentiles", default=(10.0, 90.0), min_length=2, max_length=2
    )
    chunk_size: int | None = Field(
        title="Chunk Size",
        default=None,
        ge=1,
        description="Number of frames per chunk.",
    )

    @field_validator("images", mode="after")
    @classmethod
    def check_images_ndim(cls, v: np.ndarray) -> np.ndarray:
        """
        Ensure images are 2D or 3D.

        Args:
            v: Images array.

        Returns:
            numpy.ndarray: The validated images.

        Raises:
            ValueError: If images are not 2D or 3D.
        """
        if not 1 < v.ndim < 4:
            msg = "Images must be 2D (Y, X) or 3D (Frame, Y, X) numpy arrays."
            raise ValueError(msg)
        return v

    @field_validator("instructions", mode="before")
    @classmethod
    def coerce_instructions_to_list(cls, v: Any) -> list[Any]:
        """
        Coerce a single ColorInstruction into a list.

        Args:
            v: Input instructions value.

        Returns:
            list: Instructions as a list.
        """
        if isinstance(v, ColorInstruction):
            return [v]
        return v

    @model_validator(mode="after")
    def check_instruction_indices(self) -> "Dataset":
        """
        Validate that instruction indices are within ROI bounds.

        Returns:
            Dataset: Self if valid.

        Raises:
            ValueError: If any instruction indices are out of bounds.
        """
        num_rois = len(self.rois)
        for instruction in self.instructions:
            if np.any((instruction.indices >= num_rois) | (instruction.indices < 0)):
                msg = "Instruction indices outside of ROI bounds."
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_rois_in_bounds(self) -> "Dataset":
        """
        Validate that ROI pixel indices are within image bounds.

        Returns:
            Dataset: Self if valid.

        Raises:
            ValueError: If any ROI indices are out of image bounds.
        """
        bounds = self.images.shape[-2] * self.images.shape[-1]
        for roi in self.rois:
            if roi.index.min() < 0 or roi.index.max() >= bounds:
                msg = "ROI indices outside of image bounds."
                raise ValueError(msg)
        return self


def validate_dataset(func: Callable) -> Callable:
    """
    Decorator that validates dataset-like arguments using Dataset dataclass.

    Args:
        func: Function to be wrapped.

    Returns:
        Callable: Wrapped function that validates arguments before calling.
    """

    @wraps(func)
    def decorator(*args, **kwargs) -> Callable:
        """
        Internal wrapper that binds arguments, validates via Dataset and calls func.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Any: Result of the original function call.
        """
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
