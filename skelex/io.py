"""Input discovery and frame reading for images and videos."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}


def kind_of(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return None


def collect_inputs(inputs: list[str]) -> list[tuple[Path, str]]:
    """Expand files/directories into a sorted list of (path, 'image'|'video')."""
    found: list[tuple[Path, str]] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                k = kind_of(f) if f.is_file() else None
                if k:
                    found.append((f, k))
        elif p.is_file():
            k = kind_of(p)
            if k is None:
                raise ValueError(f"Unsupported file type: {p}")
            found.append((p, k))
        else:
            raise FileNotFoundError(p)
    return found


def read_image(path: Path) -> np.ndarray:
    # imdecode handles non-ASCII (e.g. Korean) paths, unlike cv2.imread on Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    buf.tofile(str(path))


class VideoReader:
    """Iterates (frame_index, timestamp_ms, BGR frame)."""

    def __init__(self, path: Path, stride: int = 1):
        self.path = path
        self.stride = max(1, stride)
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[tuple[int, int, np.ndarray]]:
        idx = 0
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                if idx % self.stride == 0:
                    yield idx, int(round(idx * 1000.0 / self.fps)), frame
                idx += 1
        finally:
            self.cap.release()


class VideoWriter:
    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise ValueError(f"Could not open video writer: {path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()
