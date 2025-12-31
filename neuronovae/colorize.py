from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from neuronovae.dataset import validate_dataset

if TYPE_CHECKING:
    from neuronovae.cmaps import ColorInstruction
    from neuronovae.rois import ROI


def rescale(array: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
    old_min = np.nanmin(array)
    old_max = np.nanmax(array)
    return (array - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def normalize(image: np.ndarray) -> np.ndarray:
    return rescale(image, 0, 1)


def calc_gated_intensity(intensity: np.ndarray, scale: float = 0.2): ...


@validate_dataset
def colorize(
    images: np.ndarray,
    rois: list["ROI"],
    instructions: list["ColorInstruction"],
) -> np.ndarray:
    normalized_images = normalize(images).astype(np.float32)[..., None] * np.ones(images.ndim)
    for instruction in tqdm(instructions):
        colormapper, indices = instruction.__call__()
        set_rois = [rois[idx] for idx in indices]
        weights = np.stack([roi.weights for roi in set_rois], axis=0)
        intensity = np.einsum("nij,tij->nt", weights, images)
        colors = colormapper(intensity)[:, :, :3].astype(np.float32)
        g = np.clip(normalized_images[:, :, :, 0] / 0.75, 0, 1).astype(np.float32)
        B = g * normalized_images[:, :, :, 0]
        k = intensity[:, :, None] * (colors - 1.0)
        delta_sum = np.zeros_like(normalized_images, dtype=np.float32)
        for c in range(3):
            S = np.einsum("nij,nt->tij", weights, k[:, :, c])
            delta_sum[..., c] = B * S
        normalized_images += delta_sum
    np.clip(normalized_images, 0, 1, out=normalized_images)
    return rescale(normalized_images, 0, 255).astype(np.uint8)
