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


# OPTIMIZE: This function is the bottleneck 85.9% of runtime
def blend(
    foreground: np.ndarray, background: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    return foreground * alpha + background * (1 - alpha)


@validate_dataset
def colorize(
    images: np.ndarray, rois: list["ROI"], instructions: list["ColorInstruction"]
) -> np.ndarray:
    norm_images = normalize(images)
    output_images = []
    colormapper, indices = instructions[0].__call__()
    for frame in tqdm(norm_images, desc="Rendering ROIs", colour="green"):
        background = np.stack([frame for _ in range(3)], axis=-1)
        blank = np.stack([np.zeros_like(frame) for _ in range(3)], axis=-1)
        for index in indices:
            roi = rois[index]
            pixel_index = roi.index
            weights = roi.weights
            intensities = frame.ravel()[pixel_index]
            colored_pixels = colormapper(intensities)[:, :3]
            np.put(blank[:, :, 0], pixel_index, colored_pixels[:, 0])
            np.put(blank[:, :, 1], pixel_index, colored_pixels[:, 1])
            np.put(blank[:, :, 2], pixel_index, colored_pixels[:, 2])
            background = blend(blank, background, weights[:, :, np.newaxis])
        output_images.append(background)
    return rescale(output_images, 0, 255).astype(np.uint8)
    # INCOMPLETE: Currently only supports a single ColorInstruction.


# @validate_dataset
# def colorize1(
#     images: np.ndarray, rois: list["ROI"], instructions: list["ColorInstruction"]
# ) -> np.ndarray:
#     # NOTE: We
#     norm_images = normalize(images)
#     z, y, x = norm_images.shape
#     output_images = []
#     colormapper, indices = instructions[0].__call__()
#
#     # Initialize blank and weight arrays for the entire stack
#     blank = np.zeros((z, y, x, 3), dtype=norm_images.dtype)
#     weights = np.zeros((z, y, x), dtype=norm_images.dtype)
#
#     for index in indices:
#         roi = rois[index]
#         pixel_index = roi.index
#         roi_weights = roi.weights
#         for z_idx in range(z):
#             intensities = norm_images[z_idx].ravel()[pixel_index]
#             colored_pixels = colormapper(intensities)[:, :3]
#             np.put(blank[z_idx, :, :, 0], pixel_index, colored_pixels[:, 0])
#             np.put(blank[z_idx, :, :, 1], pixel_index, colored_pixels[:, 1])
#             np.put(blank[z_idx, :, :, 2], pixel_index, colored_pixels[:, 2])
#             np.put(weights[z_idx], pixel_index, roi_weights)
#
#     # Normalize weights to avoid over-saturation
#     weights = np.clip(weights, 0, 1)[:, :, :, np.newaxis]
#
#     # Blend the blank and background arrays for the entire stack
#     background = np.stack([norm_images for _ in range(3)], axis=-1).transpose(1, 2, 3, 0)
#     blended = blend(blank, background, weights)
#
#     # Rescale and return the output
#     return rescale(blended, 0, 255).astype(np.uint8)
# #
# @validate_dataset
# def colorize2(
#     images: np.ndarray, rois: list["ROI"], instructions: list["ColorInstruction"]
# ) -> np.ndarray:
#     norm_images = normalize(images)
#     output_images = []
#     colormapper, indices = instructions[0].__call__()
#     blanks = np.zeros((*norm_images.shape, 3), dtype=norm_images.dtype)
#     for idx, frame in tqdm(enumerate(norm_images), desc="Rendering ROIs", colour="green"):
#         background = np.stack([frame for _ in range(3)], axis=-1)
#         blank = blanks[idx, ...].view()
#         for index in indices:
#             roi = rois[index]
#             pixel_index = roi.index
#             weights = roi.weights
#             intensities = frame.ravel()[pixel_index]
#             colored_pixels = colormapper(intensities)[:, :3]
#             np.put(blank[:, :, 0], pixel_index, colored_pixels[:, 0])
#             np.put(blank[:, :, 1], pixel_index, colored_pixels[:, 1])
#             np.put(blank[:, :, 2], pixel_index, colored_pixels[:, 2])
#             background = blend(blank, background, weights[:, :, np.newaxis])
#         output_images.append(background)
#     return rescale(output_images, 0, 255).astype(np.uint8)
#     # INCOMPLETE: Currently only supports a single ColorInstruction.


# def colorize_rois_with_intensity_gated_alpha(
#     intensity,        # (F, Y, X), normalized
#     roi_weights,      # (N, Y, X)
#     roi_activity,     # (N, F)
#     cmaps,            # list of N matplotlib colormaps
#     I0=0.0,           # baseline intensity
#     scale=0.2         # controls how fast color turns on
# ):
#     F, Y, X = intensity.shape
#     N = roi_weights.shape[0]
#
#     # Grayscale background
#     gray = intensity[..., None] * np.ones(3)
#
#     # Intensity gate
#     g = np.clip((intensity - I0) / scale, 0, 1)    # (F,Y,X)
#
#     # Broadcast terms
#     I = intensity[None, ..., None]                 # (1,F,Y,X,1)
#     G = g[None, ..., None]                         # (1,F,Y,X,1)
#     w = roi_weights[:, None, ..., None]             # (N,1,Y,X,1)
#     a = roi_activity[:, :, None, None, None]        # (N,F,1,1,1)
#
#     # Colormap lookup (activity → color)
#     colors = np.stack(
#         [cmaps[i](roi_activity[i])[:, :3] for i in range(N)],
#         axis=0
#     )                                               # (N,F,3)
#     C = colors[:, :, None, None, :]                 # (N,F,1,1,3)
#
#     # Chromatic deviation
#     delta = a * w * G * I * (C - 1.0)
#
#     rgb = gray + delta.sum(axis=0)
#     return np.clip(rgb, 0, 1)
#
#
# def colorize_rois_with_cmaps(
#     intensity,       # (F, Y, X)
#     roi_weights,     # (N, Y, X)
#     roi_activity,    # (N, F)
#     cmaps,           # list of N matplotlib colormaps
# ):
#     F, Y, X = intensity.shape
#     N = roi_weights.shape[0]
#
#     # Base grayscale
#     gray = intensity[..., None] * np.ones(3)
#
#     # Expand common terms
#     I = intensity[None, ..., None]          # (1,F,Y,X,1)
#     w = roi_weights[:, None, ..., None]     # (N,1,Y,X,1)
#
#     # Evaluate colormaps at ROI activity
#     # Result: (N, F, 3)
#     colors = np.stack(
#         [cmaps[i](roi_activity[i])[:, :3] for i in range(N)],
#         axis=0
#     )
#
#     # Broadcast to image space
#     C = colors[:, :, None, None, :]         # (N,F,1,1,3)
#
#     # Chromatic deviations
#     delta = w * I * (C - 1.0)
#
#     # Accumulate
#     rgb = gray + delta.sum(axis=0)
#
#     return np.clip(rgb, 0, 1)