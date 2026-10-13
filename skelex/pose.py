"""Human pose skeleton extraction (33 body keypoints) with MediaPipe Pose Landmarker."""

from __future__ import annotations

import csv
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .io import VideoReader, VideoWriter, read_image, write_image

MODEL_URLS = {
    variant: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    f"pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
    for variant in ("lite", "full", "heavy")
}

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def default_model_dir() -> Path:
    return Path(os.environ.get("SKELEX_MODEL_DIR", Path.home() / ".cache" / "skelex"))


def ensure_model(variant: str = "full", model_path: str | None = None) -> Path:
    """Returns a local .task model path, downloading it on first use."""
    if model_path:
        p = Path(model_path)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    if variant not in MODEL_URLS:
        raise ValueError(f"Unknown model variant {variant!r}; choose from {list(MODEL_URLS)}")
    p = default_model_dir() / f"pose_landmarker_{variant}.task"
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pose model ({variant}) -> {p}")
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URLS[variant], tmp)
        tmp.replace(p)
    return p


@dataclass
class PoseConfig:
    model: str = "full"
    model_path: str | None = None
    num_poses: int = 1
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_visibility: float = 0.5  # keypoints below this are not drawn


class PoseExtractor:
    def __init__(self, cfg: PoseConfig, video: bool):
        from mediapipe.tasks.python import BaseOptions, vision

        self.cfg = cfg
        self.video = video
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(cfg.model, cfg.model_path))),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_poses=cfg.num_poses,
            min_pose_detection_confidence=cfg.min_detection_confidence,
            min_pose_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.landmarker.close()

    def detect(self, bgr: np.ndarray, timestamp_ms: int = 0) -> list[list[dict]]:
        """Returns one list of 33 keypoints per detected person.

        x, y are pixel coordinates; z is relative depth (same scale as x, hip-centred);
        world_* are metric 3D coordinates in metres with the hip midpoint as origin.
        """
        import mediapipe as mp

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.video:
            result = self.landmarker.detect_for_video(image, timestamp_ms)
        else:
            result = self.landmarker.detect(image)
        people = []
        worlds = result.pose_world_landmarks or [None] * len(result.pose_landmarks)
        for lms, wlms in zip(result.pose_landmarks, worlds):
            kps = []
            for i, lm in enumerate(lms):
                kp = {
                    "id": i,
                    "name": LANDMARK_NAMES[i],
                    "x": float(lm.x * w),
                    "y": float(lm.y * h),
                    "z": float(lm.z * w),
                    "visibility": float(lm.visibility or 0.0),
                    "presence": float(lm.presence or 0.0),
                }
                if wlms is not None:
                    wl = wlms[i]
                    kp.update(world_x=float(wl.x), world_y=float(wl.y), world_z=float(wl.z))
                kps.append(kp)
            people.append(kps)
        return people


_COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255)]


def draw(img: np.ndarray, people: list[list[dict]], min_visibility: float = 0.5,
         background: bool = True) -> np.ndarray:
    """Draws skeletons on a copy of img (or on black if background is False)."""
    canvas = img.copy() if background else np.zeros_like(img)
    scale = max(1, round(max(img.shape[:2]) / 640))
    for p_idx, kps in enumerate(people):
        color = _COLORS[p_idx % len(_COLORS)]
        ok = [kp["visibility"] >= min_visibility for kp in kps]
        pts = [(int(round(kp["x"])), int(round(kp["y"]))) for kp in kps]
        for a, b in CONNECTIONS:
            if ok[a] and ok[b]:
                cv2.line(canvas, pts[a], pts[b], color, 2 * scale, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            if ok[i]:
                cv2.circle(canvas, pt, 3 * scale, (0, 0, 255), -1, cv2.LINE_AA)
    return canvas


CSV_FIELDS = ["frame", "timestamp_ms", "person", "id", "name", "x", "y", "z", "visibility", "presence",
              "world_x", "world_y", "world_z"]


def _csv_rows(frame: int, ts: int, people: list[list[dict]]):
    for p_idx, kps in enumerate(people):
        for kp in kps:
            yield {"frame": frame, "timestamp_ms": ts, "person": p_idx, **kp}


def process_image(path: Path, out_dir: Path, cfg: PoseConfig, extractor: PoseExtractor) -> dict:
    img = read_image(path)
    people = extractor.detect(img)
    stem = path.stem
    write_image(out_dir / f"{stem}_pose.png", draw(img, people, cfg.min_visibility))
    write_image(out_dir / f"{stem}_pose_skeleton.png", draw(img, people, cfg.min_visibility, background=False))
    result = {"source": str(path), "width": img.shape[1], "height": img.shape[0],
              "landmark_names": LANDMARK_NAMES, "connections": CONNECTIONS, "people": people}
    (out_dir / f"{stem}_pose.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with open(out_dir / f"{stem}_pose.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_rows(0, 0, people))
    return result


def process_video(path: Path, out_dir: Path, cfg: PoseConfig, stride: int = 1, save_video: bool = True) -> dict:
    reader = VideoReader(path, stride)
    stem = path.stem
    size = (reader.width, reader.height)
    fps = reader.fps / reader.stride
    overlay_writer = VideoWriter(out_dir / f"{stem}_pose.mp4", fps, size) if save_video else None
    skel_writer = VideoWriter(out_dir / f"{stem}_pose_skeleton.mp4", fps, size) if save_video else None
    frames = []
    # VIDEO mode keeps a tracker across frames, so each video gets its own landmarker.
    with PoseExtractor(cfg, video=True) as extractor, \
            open(out_dir / f"{stem}_pose.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, CSV_FIELDS)
        writer.writeheader()
        try:
            for idx, ts, frame in reader:
                people = extractor.detect(frame, ts)
                frames.append({"frame": idx, "timestamp_ms": ts, "people": people})
                writer.writerows(_csv_rows(idx, ts, people))
                if save_video:
                    overlay_writer.write(draw(frame, people, cfg.min_visibility))
                    skel_writer.write(draw(frame, people, cfg.min_visibility, background=False))
        finally:
            if save_video:
                overlay_writer.close()
                skel_writer.close()
    result = {"source": str(path), "fps": reader.fps, "stride": reader.stride, "width": reader.width,
              "height": reader.height, "landmark_names": LANDMARK_NAMES, "connections": CONNECTIONS,
              "frames": frames}
    (out_dir / f"{stem}_pose.json").write_text(json.dumps(result, ensure_ascii=False))
    # Compact array for ML pipelines: (frames, people, 33, [x, y, z, visibility]), NaN where missing.
    arr = np.full((len(frames), cfg.num_poses, len(LANDMARK_NAMES), 4), np.nan, dtype=np.float32)
    for t, fr in enumerate(frames):
        for p, kps in enumerate(fr["people"][: cfg.num_poses]):
            arr[t, p] = [[kp["x"], kp["y"], kp["z"], kp["visibility"]] for kp in kps]
    np.save(out_dir / f"{stem}_pose.npy", arr)
    return result
