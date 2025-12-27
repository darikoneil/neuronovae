from pathlib import Path

import av
import numpy as np
from tqdm import tqdm


def export_video(path: Path, video: np.ndarray, fps: int = 30) -> None:
    frames, height, width = video.shape[:3]
    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height

    for frame in tqdm(video, total=frames, desc="Exporting...", colour="green"):
        packet = stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24"))
        container.mux(packet)

    container.close()
