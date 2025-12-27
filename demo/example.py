from pathlib import Path
from neuronovae.loaders import load_images, load_rois, Suite2PHandler

source = Path(R"C:\Users\dao25\PycharmProjects\neuronovae\assets")
video_source = source.joinpath("exemplar.npy")
video = load_images(video_source)
rois = load_rois(source, Suite2PHandler())
