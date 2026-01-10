from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

__all__ = [
    "export_gif",
    "export_image",
    "export_video",
]


def _validate_video_array(video: NDArray[np.uint8]) -> None:
    """
    Validate the shape and dtype of the video array.

    Args:
        video: Array of frames with shape (frames, height, width, channels).

    Raises:
        ValueError: If the video array shape or dtype is invalid.
    """
    if (ndim := video.ndim) != 4:
        msg = f"Expected video array with shape (frames, height, width, channels). Shape received: {ndim}"
        raise ValueError(msg)
    if (chan_received := video.shape[3]) != 3:
        msg = f"Expected video frames to have 3 channels (RGB format). Channels received: {chan_received}"
        raise ValueError(msg)
    if not np.issubdtype(video.dtype, np.uint8):
        msg = f"Expected video array to have dtype of uint8. Dtype received: {video.dtype}"
        raise ValueError(msg)


def export_gif(
    path: Path,
    video: NDArray[np.uint8],
    fps: int = 30,
    loop: bool = True,  # noqa: FBT001, FBT002
) -> None:
    """
    Export video frames to a GIF file.

    Args:
        path: File path where the video will be written.
        video: Array of frames with shape (frames, height, width, channels).
        fps: Frames per second.
        loop: Whether the GIF should loop indefinitely.

    Note:
        Frames must be in RGB format (frames x height x width x channels).
    """
    _validate_video_array(video)
    num_frames = video.shape[0]
    gif_frames = [Image.fromarray(frame, mode="RGB") for frame in video]
    duration = num_frames / (1000 * fps)  # duration in milliseconds
    gif_frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration,
        loop=loop,
        disposal=2,
    )


def export_image(path: Path, image: NDArray[np.uint8]) -> None:
    """
    Export a single image to a file.

    Args:
        path: File path where the image will be written.
        image: Array representing the image with shape (height, width, channels).
    """
    path = path.with_suffix(".png")
    img = Image.fromarray(image, mode="RGB")
    img.save(path)


def export_video(
    path: Path, video: NDArray[np.uint8], fps: int = 30, codec: str = "h264"
) -> None:
    """
    Export video frames to a video file.

     Args:
         path: File path where the video will be written.
         video: Array of frames with shape (frames, height, width, channels).
         fps: Frames per second.
         codec: Video codec to use (default "h264"). Refer to [`PyAV`](https://pyav.basswood-io.com/docs/stable/)
             documentation for supported codecs.

     Raises:
         ValueError: If the video array shape is invalid.

     Note:
         Frames must be in RGB format (frames x height x width x channels).
    """
    _validate_video_array(video)
    frames, height, width = video.shape[:3]
    container = av.open(path, mode="w")
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height

    for frame in tqdm(video, total=frames, desc="Exporting...", colour="green"):
        packet = stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24"))
        container.mux(packet)

    container.close()
