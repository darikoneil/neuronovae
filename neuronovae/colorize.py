import numpy as np
from tqdm import tqdm

from neuronovae.dataset import validate_dataset
from neuronovae.instructions import ColorInstruction
from neuronovae.maths import blend, normalize, rescale
from neuronovae.rois import ROI


@validate_dataset
def colorize(
    images: np.ndarray, rois: list[ROI], *instructions: ColorInstruction
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
            background = blend(blank, background, weights[:, :, np.newaxis] / 2)
        output_images.append(background)
    return rescale(output_images, 0, 255).astype(np.uint8)
