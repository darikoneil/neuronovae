import cv2
import numpy as np
from neuronovae.dataset import Activity, Background, Features
from pathlib import Path
#
from neuronovae.scratch_load_cmn import load_cmn
from neuronovae.scratch_make_feature import make_feature

# colorize_overlay

# (0) Test Case
cmn = load_cmn()

# (1) Load anything we need
colormap = None
filenames: set | None = None
rois = None
vmin, vmax = None, None

# (2) Establish background image
image = np.load(Path.cwd().joinpath("tests", "assets", "dont_panic.npy"),
                allow_pickle=False)[:256, :256]
background = Background(image=image)

# (3) Establish ROIs
roi_stuff = {
    "idx": cmn.get("rois"),
    "dropped": cmn.get("drops"),
    "centroids": cmn.get("centroids"),
    "coordinates": cmn.get("coordinates"),
}

# (4) Establish activity
raster = cmn.get("temporal_components")
frame_rate = 30.0
timestamps = np.linspace(0, raster.shape[-1] / frame_rate, raster.shape[-1])
activity = Activity(raster=raster, timestamps=timestamps)

# (5) Establish features
features = Features(features=make_feature(activity.frames, frame_rate*60),
                    timestamps=timestamps)
