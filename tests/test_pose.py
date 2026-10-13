import json
import os
import urllib.request

import cv2
import numpy as np
import pytest

from skelex.cli import main

SAMPLE = "https://storage.googleapis.com/mediapipe-assets/pose.jpg"


@pytest.fixture(scope="module")
def person_image(tmp_path_factory):
    path = tmp_path_factory.mktemp("img") / "person.jpg"
    try:
        urllib.request.urlretrieve(SAMPLE, path)
        from skelex.pose import ensure_model
        ensure_model("lite")
    except Exception as e:  # offline
        pytest.skip(f"network unavailable: {e}")
    return path


def test_pose_image(person_image, tmp_path):
    assert main(["pose", str(person_image), "-o", str(tmp_path), "--model", "lite"]) == 0
    data = json.loads((tmp_path / "person_pose.json").read_text())
    assert len(data["people"]) == 1
    assert len(data["people"][0]) == 33
    assert (tmp_path / "person_pose.png").is_file()


def test_pose_video(person_image, tmp_path):
    img = cv2.imread(str(person_image))
    h, w = img.shape[:2]
    vid = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for _ in range(4):
        writer.write(img)
    writer.release()
    out = tmp_path / "out"
    assert main(["pose", str(vid), "-o", str(out), "--model", "lite"]) == 0
    arr = np.load(out / "clip_pose.npy")
    assert arr.shape == (4, 1, 33, 4)
    assert not np.isnan(arr).all()
    assert (out / "clip_pose.mp4").is_file() and (out / "clip_pose.csv").is_file()
