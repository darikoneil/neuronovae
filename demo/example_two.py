import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use("QtAgg")
matplotlib.interactive(True)
from matplotlib import pyplot as plt
from neuronovae.cmapping import Color, ColorInstruction, ColorMap
from neuronovae.colorize import colorize
from neuronovae.export import export_video
from neuronovae.loaders import Suite2PHandler, load_images, load_rois
from neuronovae.colorize import normalize, blend, rescale
import random
import string
import numpy as np
from matplotlib.colors import Colormap


source = Path(R"C:\Users\dao25\PycharmProjects\neuronovae\assets")
video_source = source.joinpath("exemplar.npy")
# images = load_images(video_source)
images = np.memmap(
    Path(R"C:\Users\dao25\PycharmProjects\neuronovae\assets\data.bin"),
    mode="r",
    shape=(500, 256, 256),
    dtype=np.int16,
)
rois = load_rois(source, Suite2PHandler)
ic = np.load(Path(R"C:\Users\dao25\PycharmProjects\neuronovae\assets\iscell.npy"))
activity = np.load(Path(R"C:\Users\dao25\PycharmProjects\neuronovae\assets\F.npy"))
activity = activity[ic[:, 0] == 1, :]
roi = rois[0]

I = np.stack([image.ravel()[roi.index] for image in images], axis=0)
II = np.mean(I, axis=-1)
III = I * roi._weight
IIII = np.mean(III, axis=-1)
IV = IIII * 2
stat = np.load(
    Path(R"C:\Users\dao25\PycharmProjects\neuronovae\assets\stat.npy"),
    allow_pickle=True,
)
s = stat[0]
ix = np.arange(len(roi.index))[~s["overlap"]]
V = np.stack([image.ravel()[roi.index] for image in images], axis=0)
VI = images[0, :, :].ravel() * (
    roi.weights.ravel() / np.linalg.norm(roi.weights.ravel(), ord=1)
)
VII = np.sum(VI, axis=-1)

fig, ax = plt.subplots(1, 1)
ax.plot(activity[0, :], color="black")
# ax.plot(II, color="red")
# ax.plot(IIII, color="blue")
# ax.plot(IV, color="green")
ax.plot(VII, "orange")
