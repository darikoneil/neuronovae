from pathlib import Path
import numpy as np
from tqdm import tqdm
from neuronovae.loaders import Suite2PHandler, load_images, load_rois
from neuronovae.maths import ColorMap, normalize, blend, rescale
from neuronovae.export import export_video


if __name__ == "__main__":
    source = Path(R"C:\Users\dao25\PycharmProjects\neuronovae\assets")
    video_source = source.joinpath("exemplar.npy")
    video = load_images(video_source)
    rois = load_rois(source, Suite2PHandler)

    norm_vid = normalize(video)
    colormapper = ColorMap(
        (
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0.0, 126 / 255, 167 / 255),
            (0.0, 126 / 255, 167 / 255),
        )
    )
    final_frames = []
    for frame in tqdm(norm_vid, desc="Rendering ROIs", colour="green"):
        background = np.stack([frame for _ in range(3)], axis=-1)
        blank = np.stack([np.zeros_like(frame) for _ in range(3)], axis=-1)
        for index in np.arange(len(rois)):
            roi = rois[index]
            pixel_index = roi.index
            weights = roi.weights
            intensities = frame.ravel()[pixel_index]
            colored_pixels = colormapper(intensities)[:, :3]
            np.put(blank[:, :, 0], pixel_index, colored_pixels[:, 0])
            np.put(blank[:, :, 1], pixel_index, colored_pixels[:, 1])
            np.put(blank[:, :, 2], pixel_index, colored_pixels[:, 2])
            background = blend(blank, background, weights[:, :, np.newaxis] / 2)
        final_frames.append(background)
    final_video = rescale(np.asarray(final_frames), 0, 255).astype(np.uint8)
    export_file = Path(R"C:\Users\dao25\OneDrive\Desktop\exported_video2.mp4")
    export_video(export_file, final_video, fps=30)