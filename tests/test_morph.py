import cv2
import numpy as np

from skelex.cli import main
from skelex.morph import MorphConfig, branchpoints, endpoints, extract, prune_spurs


def _cross(size=200):
    img = np.zeros((size, size, 3), np.uint8)
    cv2.rectangle(img, (90, 20), (110, 180), (255, 255, 255), -1)
    cv2.rectangle(img, (20, 90), (180, 110), (255, 255, 255), -1)
    return img


def test_cross_skeleton_has_four_ends_and_a_junction():
    for method in ("zhang", "lee", "medial", "thin"):
        _, skel, _ = extract(_cross(), MorphConfig(method=method))
        assert skel.any()
        assert int(endpoints(skel).sum()) >= 4, method
        assert branchpoints(skel).any(), method


def test_invert_handles_dark_object():
    img = 255 - _cross()
    _, skel, _ = extract(img, MorphConfig(invert=True))
    assert 4 <= int(endpoints(skel).sum()) <= 8


def test_prune_removes_short_spur_keeps_long_line():
    skel = np.zeros((50, 50), bool)
    skel[25, 5:45] = True  # main line
    skel[20:25, 25] = True  # 5px spur
    pruned = prune_spurs(skel, 8)
    assert not pruned[20, 25]
    assert pruned[25, 5:45].sum() >= 38


def test_cli_image_and_video(tmp_path):
    img_path = tmp_path / "cross.png"
    cv2.imwrite(str(img_path), _cross())
    vid_path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (200, 200))
    for _ in range(5):
        w.write(_cross())
    w.release()
    out = tmp_path / "out"
    assert main(["morph", str(tmp_path), "-o", str(out), "--method", "medial"]) == 0
    for name in ("cross_skeleton.png", "cross_overlay.png", "cross_skeleton.json",
                 "clip_skeleton.mp4", "clip_overlay.mp4", "clip_skeleton.json"):
        assert (out / name).is_file(), name
