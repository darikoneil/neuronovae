from pathlib import Path

import matplotlib
import numpy as np

from neuronovae.cmapping import Color, ColorInstruction, ColorMap
from neuronovae.colorize import colorize
from neuronovae.export import export_video
from neuronovae.loaders import Suite2PHandler, load_images, load_rois

matplotlib.use("QtAgg")
matplotlib.interactive(b=True)

BLUE: Color = Color(
    17 / 255,
    159 / 255,
    255 / 255,
)
tag = "video_2"
ext = ".mp4"


if __name__ == "__main__":
    source = Path(R"C:\Users\dao25\PycharmProjects\neuronovae\assets")
    video_source = source.joinpath("exemplar.npy")
    images = load_images(video_source)[:, :, :]
    rois = load_rois(source, Suite2PHandler)
    cm = ColorMap(
        (
            (0, 0, 0, 1),
            BLUE,
        )
    )
    indices = np.arange(len(rois))
    instructions = ColorInstruction(cmap=cm, indices=indices)
    colored_video = colorize(
        images=images,
        rois=rois,
        instructions=instructions,
        scaling=(1.0, 5.0),
    )
    export_file = Path(R"C:\Users\dao25\Desktop").joinpath(f"{tag}{ext}")
    export_video(export_file, colored_video, fps=30)
