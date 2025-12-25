import numpy as np


def flatten_index(shape: tuple[int, ...], indices: np.ndarray) -> int:
    return np.ravel_multi_index([indices[:, dim] for dim in range(len(shape))], shape)


def rescale(array: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
    old_min = np.nanmin(array)
    old_max = np.nanmax(array)
    return (array - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def normalize(image: np.ndarray) -> np.ndarray:
    return rescale(image, 0, 1)


def blend(
    foreground: np.ndarray, background: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    alpha_background = 1.0 - alpha
    return foreground * alpha + background * (1 - alpha)
