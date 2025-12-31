from pathlib import Path

import numpy as np

from neuronovae.cmapping import Color, ColorInstruction, ColorMap
from neuronovae.colorize import colorize
from neuronovae.export import export_video
from neuronovae.loaders import Suite2PHandler, load_images, load_rois
from neuronovae.colorize import colorize
import random
import string
import numpy as np
from matplotlib.colors import Colormap

BLUE: Color = Color(
    0.0,
    126 / 255,
    167 / 255,
)


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
    # intensity = normalize(images)
    # colormapper, indices = instructions.__call__()
    #
    # activity = np.load(Path(R"C:\Users\dao25\PycharmProjects\neuronovae\assets\F.npy"))
    # ic = np.load(Path(R"C:\Users\dao25\PycharmProjects\neuronovae\assets\iscell.npy"))
    # activity = activity[ic[:, 0] == 1, :]

    colored_video = colorize(images=images, rois=rois, instructions=instructions)
    export_file = Path(R"C:\Users\dao25\Desktop\test7.mp4")
    export_video(export_file, colored_video, fps=30)
