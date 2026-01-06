from pathlib import Path
from neuronovae.loaders import load_images, load_rois, Suite2PHandler
from neuronovae.cmapping import Color, ColorMap, ColorInstruction
from neuronovae.export import export_video
from neuronovae.colorize import colorize
import numpy as np


folder = Path.cwd().joinpath("assets")
images = load_images(folder.joinpath("exemplar.npy"))
rois = load_rois(folder, handler=Suite2PHandler)


color_background = Color.from_rgba(0, 0, 0, 0)
color_one = Color.from_rgba(197, 89, 94)
color_map = ColorMap(colors=[color_background, color_one])
instruction = ColorInstruction(cmap=color_map, indices=np.arange(len(rois)))
colored = colorize(images, rois, instruction, scaling=[1, 99.0], chunk_size=50)
export_video(Path(R"C:\Users\doneil\Desktop\example_output.mp4"), colored, fps=30, codec="h264")