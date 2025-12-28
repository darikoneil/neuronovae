import numpy as np

from neuronovae.colors import Color


class ColorMap:
    def __init__(self, colors: tuple[Color, ...]):
        self.colors = np.asarray(colors)
        self.indices = np.linspace(0, 1, len(colors))

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """
        Get the interpolated colors for given values between 0 and 1.
        Supports scalar or vector inputs.
        """
        values = np.clip(values, 0, 1)  # Ensure values are within [0, 1]

        # Find the indices of the surrounding positions
        idx = np.searchsorted(self.indices, values, side="right") - 1
        idx = np.clip(idx, 0, len(self.colors) - 2)  # Avoid out-of-bounds

        # Calculate interpolation weights
        t = (values - self.indices[idx]) / (self.indices[idx + 1] - self.indices[idx])

        # Interpolate between the two colors
        return (1 - t)[..., None] * self.colors[idx] + t[..., None] * self.colors[idx + 1]


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
    return foreground * alpha + background * (1 - alpha)
