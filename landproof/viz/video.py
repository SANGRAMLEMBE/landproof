"""H.264 MP4 writer (pipes frames to ffmpeg; uses imageio-ffmpeg's binary if no system ffmpeg)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("ffmpeg not found: install ffmpeg or `pip install imageio-ffmpeg`") from e


class VideoWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], crf: int = 20):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        self.size = size
        self.proc = subprocess.Popen(
            [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}",
             "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)],
            stdin=subprocess.PIPE,
        )

    def write(self, rgb: np.ndarray):
        assert rgb.shape[1] == self.size[0] and rgb.shape[0] == self.size[1], (rgb.shape, self.size)
        self.proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
