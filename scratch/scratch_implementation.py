import numpy as np
from pathlib import Path
from neuronovae.loaders import Suite2PHandler, load_images, load_rois
from neuronovae.cmapping import ColorInstruction, ColorMap, Color
from neuronovae.rois import ROI
from tqdm import tqdm
from boltons.iterutils import chunk_ranges
from neuronovae.colorize import normalize


RED = Color(255 / 255, 62 / 255, 65 / 255)


source = Path(R"C:\Users\dao25\PycharmProjects\neuronovae\assets")
video_source = source.joinpath("exemplar.npy")
images = load_images(video_source)[:, :, :]
rois = load_rois(source, Suite2PHandler)
cm = ColorMap(
    (
        (0, 0, 0, 1),
        (255 / 255, 62 / 255, 65 / 255, 1),
    )
)
instruction = ColorInstruction(cm, np.arange(len(rois)))
colormapper, indices = instruction.__call__()
roi = rois[0]
cm = ColorMap(
    (
        (0, 0, 0, 1),
        # (0, 0, 0, 1),
        # (0, 0, 0, 1),
        # RED,
        RED,
    )
)
indices = np.arange(len(rois))
instructions = ColorInstruction(cmap=cm, indices=indices)


def colorize_new(
    images: np.ndarray, rois: list[ROI], instructions: list[ColorInstruction]
) -> np.ndarray:
    r"""
    Calculates

    .. math::
        \Delta_i(t,y,x) = a_i(t) * w_i(y,x) * g(I(t,y,x))I(t,y,x)(c_i(t)-1)


        x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
    """
    colormapper = instructions[0].__call__()[0]
    norm_images = normalize(images)
    weights = np.stack(
        [roi.weights / np.linalg.norm(roi.weights.ravel(), ord=1) for roi in rois],
        axis=0,
    )
    intensity = np.stack(
        [
            np.asarray(images * weight[np.newaxis, :, :]).sum(axis=(-2, -1))
            for weight in weights
        ],
        axis=0,
    )
    colors = colormapper(intensity)[:, :, :3]
    g = np.clip((norm_images - norm_images.mean(axis=0)) / 0.2, 0, 1)
    G = g[None, ..., None]
    W = weights[:, None, ..., None]
    A = intensity[:, :, None, None, None]
    I = norm_images[None, ..., None]
    C = colors[:, :, None, None, :]
    delta = A * W * G * I * (C - 1.0)
    rgb = norm_images[..., None] * np.ones(3) + delta.sum(axis=0)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def colorize_new2(
    images: np.ndarray, rois: list[ROI], instructions: list[ColorInstruction]
) -> np.ndarray:
    """
    Optimized colorize that avoids building large N×T×Y×X arrays.
    """
    colormapper = instructions[0].__call__()[0]

    # work in float32 to reduce memory
    images_f = images.astype(np.float32)
    norm_images = normalize(images_f).astype(np.float32)

    # stack ROI weights and normalize (L1)
    weights = np.stack(
        [roi.weights / np.linalg.norm(roi.weights.ravel(), ord=1) for roi in rois],
        axis=0,
    ).astype(np.float32)  # shape (N, Y, X)

    # intensity per ROI per frame: shape (N, T)
    # equivalent to summing images * weight over Y,X
    intensity = np.einsum("nij,tij->nt", weights, images_f)

    # map intensities to colors (N, T, 3)
    colors = colormapper(intensity)[:, :, :3].astype(np.float32)

    # spatial modulation g and base image B = g * I
    g = np.clip(norm_images / 0.2, 0, 1).astype(np.float32)
    B = g * norm_images  # shape (T, Y, X)

    # per-ROI per-frame color factor k = intensity * (color - 1)
    # shape (N, T, 3)
    k = intensity[:, :, None] * (colors - 1.0)

    # accumulate delta per channel without creating full N×T×Y×X array
    delta_sum = np.zeros((*images_f.shape, 3), dtype=np.float32)  # (T, Y, X, 3)
    for c in range(3):
        # S has shape (T, Y, X): sum over ROIs of weights * k[..., c]
        S = np.einsum("nij,nt->tij", weights, k[:, :, c])
        delta_sum[..., c] = B * S

    # base RGB is norm_images expanded to 3 channels
    rgb = norm_images[..., None] + delta_sum

    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


if __name__ == "__main__":
    # chunks = list(chunk_ranges(len(images), 30))
    # colored_video = np.zeros((*images.shape, 3), dtype=np.uint8)
    # for chunk in tqdm(chunks, desc="Processing chunks", colour="red"):
    #     colored_video[chunk[0]:chunk[1], ...] = colorize_new2(
    #         images=images[chunk[0]:chunk[1], :],
    #         rois=rois,
    #         instructions=[instructions],
    #     )
    colored_video = colorize_new2(images, rois, [instructions])
    export_file = Path(R"C:\Users\dao25\Desktop\test_new222.mp4")
    from neuronovae.export import export_video

    export_video(export_file, colored_video, fps=30)
