from neuronovae.loaders import load_images
import numpy as np
from pathlib import Path
from neuronovae.rois import ROI, flatten_index
import matplotlib
from itertools import cycle
from neuronovae.export import export_video
from tqdm import tqdm
from numpy.random import default_rng
from functools import partial

matplotlib.use("QtAgg")
matplotlib.interactive(True)
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from neuronovae.colorize import rescale, normalize, blend


def load_data() -> tuple[list[ROI], np.ndarray]:
    image_file = Path(
        R"C:\Users\dao25\AppData\Roaming\JetBrains\PyCharm2025.2\scratches\exemplar\s_burst_mini.tif"
    )
    images = load_images(image_file)

    s2p_dir = Path(
        R"C:\Users\dao25\AppData\Roaming\JetBrains\PyCharm2025.2\scratches\exemplar\suite2p\plane0"
    )

    ops = np.load(s2p_dir.joinpath("ops.npy"), allow_pickle=True).item()
    stat = np.load(s2p_dir.joinpath("stat.npy"), allow_pickle=True)
    iscell = np.load(s2p_dir.joinpath("iscell.npy"), allow_pickle=True)

    ref_image = images.mean(axis=0)
    nrn_idx = iscell[:, 0] == 1
    rois = []

    for idx, nrn in enumerate(stat):
        if not nrn_idx[idx]:
            continue
        pixels = np.asarray([nrn["ypix"], nrn["xpix"]]).T
        roi = ROI(pixels, nrn["lam"], ref_image.shape)
        rois.append(roi)
    return rois, images


rois, video = load_data()
selector = partial(default_rng(42).choice, len(rois), size=len(rois), replace=False)
selected_rois = selector()

normalized_video = normalize(video)
ref_image = normalized_video.mean(axis=0)

colormapper = LinearSegmentedColormap.from_list(
    "custom",
    [
        (0, 0, 0),
        (0, 0, 0),
        (0.0, 126 / 255, 167 / 255),
        (0.0, 126 / 255, 167 / 255),
    ],
)

final_frames = []
for frame in tqdm(normalized_video, desc="Rendering ROIs", colour="green"):
    background = np.stack([frame for _ in range(3)], axis=-1)
    blank = np.stack([np.zeros_like(ref_image) for _ in range(3)], axis=-1)
    for index in selected_rois:
        roi = rois[index]
        pixel_index = roi.index
        weights = roi.weights
        intensities = frame.ravel()[pixel_index]
        colored_pixels = colormapper(intensities)[:, :3]
        np.put(blank[:, :, 0], pixel_index, colored_pixels[:, 0])
        np.put(blank[:, :, 1], pixel_index, colored_pixels[:, 1])
        np.put(blank[:, :, 2], pixel_index, colored_pixels[:, 2])
        background = blend(blank, background, weights[:, :, np.newaxis])
    final_frames.append(background)
final_video = rescale(np.asarray(final_frames), 0, 255).astype(np.uint8)
export_file = Path(R"C:\Users\dao25\OneDrive\Desktop\exported_video.mp4")
export_video(export_file, final_video, fps=30)
