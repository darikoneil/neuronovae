from typing import TYPE_CHECKING

import numpy as np
from boltons.iterutils import chunk_ranges
from tqdm import tqdm

from neuronovae.dataset import validate_dataset

if TYPE_CHECKING:
    from neuronovae.cmapping import ColorInstruction
    from neuronovae.rois import ROI


def rescale(array: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
    """
    Rescales the input array to a new range [new_min, new_max].

    Args:
        array: The input numpy array to be rescaled.
        new_min: The minimum value of the new range.
        new_max: The maximum value of the new range.

    Returns:
        The rescaled numpy array.

    Example:
        >>> rescale(np.array([1, 2, 3]), 0, 1)
        array([0. , 0.5, 1. ])
    """
    old_min = np.nanmin(array)
    old_max = np.nanmax(array)
    return (array - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def normalize(images: np.ndarray) -> np.ndarray:
    """
    Normalizes the input image to the range [0, 1].

    Args:
        images: The input numpy array representing images.

    Returns:
        The normalized numpy array.

    Example:
        >>> normalize(np.array([[0, 50], [100, 150]]))
        array([[0.  , 0.333],
               [0.667, 1.   ]])
    """
    return rescale(images, 0, 1)


def calc_gated_images(
    norm_images: np.ndarray, baseline: np.ndarray, bound: np.ndarray
) -> np.ndarray:
    r"""
    Calculates the baseline-gated intensity of the provided images. This gating function
    determines *when* and *how much* color should be applied, given the baseline intensity
    and an upper bound at which colorization saturates.

    Precisely, the gating function is defined as:
    $$
    g(I) = clip(\frac{I - I_{base}}{I_{bound}}, 0, 1)
    $$

    Args:
        norm_images: The normalized input images.
        baseline: The baseline intensity values at which colorization starts.
        bound: The upper bound intensity values at which colorization saturates.

    Returns:
        The gated images.

    Raises:
        ValueError: If the input arrays have mismatched shapes.

    Example:
        >>> calc_gated_images(
        ...     np.array([[0.2, 0.5], [0.8, 1.0]]),
        ...     np.array([[0.1, 0.3], [0.6, 0.9]]),
        ...     np.array([[0.5, 0.7], [0.9, 1.1]]),
        ... )
        array([[0.2, 0.286], [0.222, 0.091]])
    """
    return norm_images * np.clip((norm_images - baseline) / bound, 0, 1)


@validate_dataset
def colorize(
    images: np.ndarray,
    rois: list["ROI"],
    instructions: list["ColorInstruction"],
    scaling: tuple[float, float] = (10.0, 90.0),
    chunk_size: int | None = None,
) -> np.ndarray:
    """
    Applies colorization to a set of images based on specified regions of interest (ROIs)
    and color instructions. The function normalizes the images, calculates gated intensities,
    and applies color mappings to generate the final colored images.

    Args:
        images: A numpy array of input images to be colorized.
        rois: A list of ROI objects representing regions of interest in the images.
        instructions: A list of ColorInstruction objects specifying how to apply color mappings.
        scaling: A tuple specifying the lower and upper percentiles for baseline and bound intensities.
        chunk_size: The size of chunks to process at a time, or None to process all images at once.

    Returns:
        A numpy array of colorized images with RGB channels.

    Raises:
        ValueError: If the input arrays have mismatched shapes or invalid data.
    """
    norm_images = normalize(images).astype(np.float32)
    colored_images = np.zeros((*images.shape, 3), dtype=np.uint8)
    baseline = np.percentile(norm_images, scaling[0], axis=0)
    bound = np.percentile(norm_images, scaling[1], axis=0)
    chunk_size = chunk_size or len(images)
    chunks = list(chunk_ranges(len(images), chunk_size))
    for chunk in tqdm(chunks, desc="Colorizing", colour="blue"):
        norm_images_chunk = norm_images[slice(*chunk), ...].view()
        gated_images = calc_gated_images(norm_images_chunk, baseline, bound)
        delta = np.zeros((*gated_images.shape, 3), dtype=np.float32)
        for instruction in instructions:
            colormapper, indices = instruction.__call__()
            roi_subset = [rois[idx] for idx in indices]
            weights = np.stack([roi.weights for roi in roi_subset], axis=0)
            masks = np.stack([roi.mask for roi in roi_subset], axis=0)
            intensity = np.einsum("nij,tij->nt", masks, norm_images_chunk)
            colors = colormapper(intensity)[:, :, :3].astype(np.float32)
            k = intensity[:, :, None] * (colors - 1.0)
            for channel in range(3):
                s = np.einsum("nij,nt->tij", weights, k[:, :, channel])
                delta[..., channel] += gated_images * s
        rgb_chunk = norm_images_chunk[..., None] * np.ones(3) + delta
        np.clip(rgb_chunk, 0, 1, out=rgb_chunk)
        colored_images[slice(*chunk), ...] = rescale(rgb_chunk, 0, 255).astype(np.uint8)
    return colored_images
