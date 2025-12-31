from typing import TYPE_CHECKING

import numpy as np
from boltons.iterutils import chunk_ranges
from tqdm import tqdm

from neuronovae.dataset import validate_dataset

if TYPE_CHECKING:
    from neuronovae.cmaps import ColorInstruction
    from neuronovae.rois import ROI


def rescale(array: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
    """
    Rescales the input array to a new range [new_min, new_max].

    :param array: The input numpy array to be rescaled.
    :param new_min: The minimum value of the new range.
    :param new_max: The maximum value of the new range.
    :return: The rescaled numpy array.
    """
    old_min = np.nanmin(array)
    old_max = np.nanmax(array)
    return (array - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def normalize(images: np.ndarray) -> np.ndarray:
    """
    Normalizes the input image to the range [0, 1].

    :param images: The input numpy array representing images.
    :return: The normalized numpy array.
    """
    return rescale(images, 0, 1)


def calc_gated_images(
    norm_images: np.ndarray, baseline: np.ndarray, bound: np.ndarray
) -> np.ndarray:
    return norm_images * np.clip((norm_images - baseline) / bound, 0, 1)


# DOCME: calc_gated_images


@validate_dataset
def colorize(
    images: np.ndarray,
    rois: list["ROI"],
    instructions: list["ColorInstruction"],
    scaling: tuple[float, float] = (10.0, 90.0),
    chunk_size: int | None = None,
) -> np.ndarray:
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
                S = np.einsum("nij,nt->tij", weights, k[:, :, channel])
                delta[..., channel] += gated_images * S
        rgb_chunk = norm_images_chunk[..., None] * np.ones(3) + delta
        np.clip(rgb_chunk, 0, 1, out=rgb_chunk)
        colored_images[slice(*chunk), ...] = rescale(rgb_chunk, 0, 255).astype(np.uint8)
    return colored_images


# DOCME: colorize
