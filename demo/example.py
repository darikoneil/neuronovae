from pathlib import Path

import numpy as np
from tqdm import tqdm

from neuronovae.export import export_video
from neuronovae.loaders import Suite2PHandler, load_images, load_rois
from neuronovae.maths import blend, normalize, rescale
from neuronovae.cmaps import ColorMap, RED
from neuronovae.colorize import colorize, ColorInstruction
from neuronovae.dataset import Dataset


if __name__ == "__main__":
    source = Path(R"C:\Users\dao25\PycharmProjects\neuronovae\assets")
    video_source = source.joinpath("exemplar.npy")
    video = load_images(video_source)[:90, :, :]
    rois = load_rois(source, Suite2PHandler)
    cm = ColorMap(
        (
            (0, 0, 0),
            (0, 0, 0),
            RED,
            RED,
        )
    )
    indices = np.arange(len(rois))
    inst = ColorInstruction(cmap=cm, indices=indices)
    dataset = Dataset(images=video, rois=rois, instructions=inst)
    colored_video = colorize(dataset.images, dataset.rois, dataset.instructions)
    export_file = Path(R"C:\Users\dao25\OneDrive\Desktop\exported_video2.mp4")
    export_video(export_file, colored_video, fps=30)
