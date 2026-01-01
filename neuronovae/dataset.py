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
    Represents a dataset containing images, ROIs, and color instructions.

    :param images: A numpy array representing the images (2D or 3D).
    :param rois: A list of ROI objects.
    :param instructions: A list of color instructions for the dataset.
    :param scaling: A tuple representing the scaling percentiles.
    :param chunk_size: The number of frames per chunk, or None if not specified.
    :raises ValueError: If validation checks fail for images, instructions, or ROIs.
    :example:
        dataset = Dataset(
            images=np.random.rand(10, 256, 256),
            rois=[ROI(...)],
            instructions=[ColorInstruction(...)],
            scaling=(10.0, 90.0),
            chunk_size=5
        )
    :note: This class uses Pydantic for validation and supports assignment validation.
    :attention: Ensure that the images, ROIs, and instructions conform to the expected formats.
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
        Validates the number of dimensions of the images.

        :param v: The numpy array of images.
        :return: The validated numpy array.
        :raises ValueError: If the images are not 2D or 3D.
        """
        if not 1 < v.ndim < 4:
            msg = "Images must be 2D (Y, X) or 3D (Frame, Y, X) numpy arrays."
            raise ValueError(msg)
        return v

    @field_validator("instructions", mode="before")
    @classmethod
    def coerce_instructions_to_list(cls, v: Any) -> list[Any]:
        """
        Ensures that the instructions are in a list format.

        :param v: The input instructions.
        :return: The instructions as a list.
        """
        if isinstance(v, ColorInstruction):
            return [v]
        return v

    @model_validator(mode="after")
    def check_instruction_indices(self) -> "Dataset":
        """
        Validates that instruction indices are within the bounds of the ROIs.

        :return: The validated Dataset object.
        :raises ValueError: If any instruction indices are out of bounds.
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
        Validates that ROI indices are within the bounds of the images.

        :return: The validated Dataset object.
        :raises ValueError: If any ROI indices are out of image bounds.
        """
        bounds = self.images.shape[-2] * self.images.shape[-1]
        for roi in self.rois:
            if roi.index.min() < 0 or roi.index.max() >= bounds:
                msg = "ROI indices outside of image bounds."
                raise ValueError(msg)
        return self


def validate_dataset(func: Callable) -> Callable:
    """
    Decorator to validate dataset parameters before passing them to the function.

    :param func: The function to decorate.
    :return: The decorated function.
    :example:
        @validate_dataset
        def process_dataset(images, rois, instructions, scaling, chunk_size):
            ...
    :note: This decorator uses the Dataset class for validation.
    :attention: Ensure that the function parameters match the Dataset fields.
    """

    @wraps(func)
    def decorator(*args, **kwargs) -> Callable:
        """
        Wrapper function for dataset validation.

        :param args: Positional arguments for the function.
        :param kwargs: Keyword arguments for the function.
        :return: The result of the decorated function.
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
