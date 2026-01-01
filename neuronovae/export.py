from pathlib import Path

import av
import numpy as np
from tqdm import tqdm


# FEATURE: Implement export functions for GIF
def export_gif(path: Path, video: np.ndarray, fps: int = 10) -> None:
    """Placeholder: export video frames to an animated GIF at `path`.

    This is a minimal placeholder so GUI can call the function. The user
    intends to implement the real functionality later.
    """
    # TODO: Implement GIF export
    return


# FEATURE: Implement export functions for images
def export_image(path: Path, video: np.ndarray) -> None:
    """Placeholder: export video frames to a PNG sequence under `path`.

    This is a minimal placeholder so GUI can call the function. The user
    intends to implement the real functionality later.
    """
    # TODO: Implement PNG sequence export
    return


def export_video(path: Path, video: np.ndarray, fps: int = 30) -> None:
    """
    Exports a video to the specified file path in H.264 format.

    :param path: The file path where the video will be saved.
    :param video: A numpy array representing the video frames, with shape (frames, height, width, channels).
    :param fps: The frame rate of the video (default is 30).
    :return: None
    :raises ValueError: If the video array has an invalid shape.
    :warns: Ensure the file path has write permissions.
    :example:
        export_video(Path("output.mp4"), np.random.randint(0, 255, (100, 720, 1280, 3), dtype=np.uint8), fps=24)
    :note: The video frames must be in RGB format.
    :attention: This function requires the `av` library to be installed.
    """
    frames, height, width = video.shape[:3]
    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height

    for frame in tqdm(video, total=frames, desc="Exporting...", colour="green"):
        packet = stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24"))
        container.mux(packet)

    container.close()
