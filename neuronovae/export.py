from pathlib import Path

import av
import numpy as np
from tqdm import tqdm


# FEATURE: Implement export functions for GIF
def export_gif(path: Path, video: np.ndarray, fps: int = 10) -> None:
    """
    Placeholder to export video frames to an animated GIF.

    Args:
        path: Destination path.
        video: Video frames array.
        fps: Frame rate.

    Note:
        This is a minimal placeholder to allow GUI integration.
    """
    # TODO: Implement GIF export
    return


# FEATURE: Implement export functions for images
def export_image(path: Path, video: np.ndarray) -> None:
    """
    Placeholder to export video frames to a PNG sequence.

    Args:
        path: Destination directory.
        video: Video frames array.

    Note:
        This is a minimal placeholder to allow GUI integration.
    """
    # TODO: Implement PNG sequence export
    return


def export_video(path: Path, video: np.ndarray, fps: int = 30) -> None:
    """
    Export a video to disk using H.264 encoding.

    Args:
        path: File path where the video will be written.
        video: Numpy array of frames with shape (frames, height, width, channels).
        fps: Frames per second (default 30).

    Raises:
        ValueError: If the video array shape is invalid.

    Note:
        Frames must be in RGB format. Requires the 'av' library.
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
