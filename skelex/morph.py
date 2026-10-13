"""Morphological skeletonization (medial axis / thinning) of shapes in images and videos."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import medial_axis, remove_small_objects, skeletonize, thin

from .io import VideoReader, VideoWriter, read_image, write_image

METHODS = ("zhang", "lee", "medial", "thin")


@dataclass
class MorphConfig:
    method: str = "zhang"
    threshold: int | None = None  # None = Otsu
    invert: bool = False  # set when the object is darker than the background
    blur: int = 3  # Gaussian kernel size before thresholding (0 disables)
    min_size: int = 64  # drop foreground blobs smaller than this many pixels
    prune: int = 0  # remove spur branches up to this many pixels long


def binarize(img: np.ndarray, cfg: MorphConfig) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if cfg.blur and cfg.blur > 1:
        k = cfg.blur | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    mode = cv2.THRESH_BINARY_INV if cfg.invert else cv2.THRESH_BINARY
    if cfg.threshold is None:
        _, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, cfg.threshold, 255, mode)
    mask = mask > 0
    if cfg.min_size > 0:
        mask = _remove_small(mask, cfg.min_size)
    return mask


def _remove_small(mask: np.ndarray, size: int) -> np.ndarray:
    # scikit-image >= 0.26 renamed min_size to max_size.
    if "max_size" in inspect.signature(remove_small_objects).parameters:
        return remove_small_objects(mask, max_size=size - 1)
    return remove_small_objects(mask, min_size=size)


def skeletonize_mask(mask: np.ndarray, cfg: MorphConfig) -> tuple[np.ndarray, np.ndarray | None]:
    """Returns (skeleton bool array, distance map or None).

    The distance map (only for 'medial') gives the local half-thickness along the skeleton.
    """
    dist = None
    if cfg.method == "zhang":
        skel = skeletonize(mask)
    elif cfg.method == "lee":
        skel = skeletonize(mask, method="lee") > 0
    elif cfg.method == "medial":
        skel, dist = medial_axis(mask, return_distance=True)
    elif cfg.method == "thin":
        skel = thin(mask)
    else:
        raise ValueError(f"Unknown method {cfg.method!r}; choose from {METHODS}")
    skel = skel.astype(bool)
    if cfg.prune > 0:
        skel = prune_spurs(skel, cfg.prune)
    return skel, dist


_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


def neighbor_count(skel: np.ndarray) -> np.ndarray:
    return ndimage.convolve(skel.astype(np.uint8), _NEIGHBOR_KERNEL, mode="constant") * skel


def endpoints(skel: np.ndarray) -> np.ndarray:
    return skel & (neighbor_count(skel) == 1)


def branchpoints(skel: np.ndarray) -> np.ndarray:
    return skel & (neighbor_count(skel) >= 3)


def prune_spurs(skel: np.ndarray, length: int) -> np.ndarray:
    """Iteratively strip end pixels, then regrow main branches so only short spurs vanish."""
    skel = skel.copy()
    removed = np.zeros_like(skel)
    for _ in range(length):
        ep = endpoints(skel)
        if not ep.any():
            break
        skel &= ~ep
        removed |= ep
    # Regrow from remaining endpoints back along removed pixels to restore long branches' tips.
    for _ in range(length):
        grow = ndimage.binary_dilation(endpoints(skel), structure=np.ones((3, 3))) & removed
        if not grow.any():
            break
        skel |= grow
        removed &= ~grow
    return skel


def stats(skel: np.ndarray, dist: np.ndarray | None = None) -> dict:
    _, n_components = ndimage.label(skel, structure=np.ones((3, 3)))
    out = {
        "skeleton_pixels": int(skel.sum()),
        "components": int(n_components),
        "endpoints": [[int(x), int(y)] for y, x in zip(*np.nonzero(endpoints(skel)))],
        "branchpoints": [[int(x), int(y)] for y, x in zip(*np.nonzero(branchpoints(skel)))],
    }
    if dist is not None and skel.any():
        widths = dist[skel] * 2
        out["mean_width"] = float(widths.mean())
        out["max_width"] = float(widths.max())
    return out


def render(img: np.ndarray, skel: np.ndarray, thickness: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Returns (white-on-black skeleton image, overlay on the original)."""
    skel_u8 = skel.astype(np.uint8) * 255
    if thickness > 1:
        skel_u8 = cv2.dilate(skel_u8, np.ones((thickness, thickness), np.uint8))
    overlay = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = (overlay * 0.6).astype(np.uint8)
    overlay[skel_u8 > 0] = (0, 0, 255)
    for x, y in zip(*np.nonzero(endpoints(skel))[::-1]):
        cv2.circle(overlay, (int(x), int(y)), 3, (0, 255, 0), -1)
    for x, y in zip(*np.nonzero(branchpoints(skel))[::-1]):
        cv2.circle(overlay, (int(x), int(y)), 3, (255, 0, 0), -1)
    return skel_u8, overlay


def extract(img: np.ndarray, cfg: MorphConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mask = binarize(img, cfg)
    skel, dist = skeletonize_mask(mask, cfg)
    return mask, skel, dist


def process_image(path: Path, out_dir: Path, cfg: MorphConfig, thickness: int = 1) -> dict:
    img = read_image(path)
    mask, skel, dist = extract(img, cfg)
    skel_img, overlay = render(img, skel, thickness)
    stem = path.stem
    write_image(out_dir / f"{stem}_mask.png", mask.astype(np.uint8) * 255)
    write_image(out_dir / f"{stem}_skeleton.png", skel_img)
    write_image(out_dir / f"{stem}_overlay.png", overlay)
    result = {"source": str(path), "method": cfg.method, "width": img.shape[1], "height": img.shape[0],
              **stats(skel, dist)}
    (out_dir / f"{stem}_skeleton.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def process_video(path: Path, out_dir: Path, cfg: MorphConfig, stride: int = 1, thickness: int = 1) -> dict:
    reader = VideoReader(path, stride)
    size = (reader.width, reader.height)
    fps = reader.fps / reader.stride
    stem = path.stem
    skel_writer = VideoWriter(out_dir / f"{stem}_skeleton.mp4", fps, size)
    overlay_writer = VideoWriter(out_dir / f"{stem}_overlay.mp4", fps, size)
    frames = []
    try:
        for idx, ts, frame in reader:
            _, skel, dist = extract(frame, cfg)
            skel_img, overlay = render(frame, skel, thickness)
            skel_writer.write(cv2.cvtColor(skel_img, cv2.COLOR_GRAY2BGR))
            overlay_writer.write(overlay)
            s = stats(skel, dist)
            frames.append({"frame": idx, "timestamp_ms": ts, "skeleton_pixels": s["skeleton_pixels"],
                           "components": s["components"], "endpoints": len(s["endpoints"]),
                           "branchpoints": len(s["branchpoints"])})
    finally:
        skel_writer.close()
        overlay_writer.close()
    result = {"source": str(path), "method": cfg.method, "fps": reader.fps, "stride": reader.stride,
              "width": reader.width, "height": reader.height, "frames": frames}
    (out_dir / f"{stem}_skeleton.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
