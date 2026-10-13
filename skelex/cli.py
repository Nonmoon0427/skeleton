"""Command line interface: `skelex pose ...` and `skelex morph ...`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .io import collect_inputs


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("inputs", nargs="+", help="image/video files or directories")
    p.add_argument("-o", "--output", default="outputs", help="output directory (default: outputs)")
    p.add_argument("--stride", type=int, default=1, help="process every N-th video frame (default: 1)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skelex", description="Skeleton extraction from images and videos")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    pose = sub.add_parser("pose", help="human pose skeleton (33 keypoints, MediaPipe)")
    _add_common(pose)
    pose.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                      help="model size: lite (fast) / full / heavy (accurate)")
    pose.add_argument("--model-path", help="use a local .task model file instead of downloading")
    pose.add_argument("--num-poses", type=int, default=1, help="max people per frame (default: 1)")
    pose.add_argument("--min-confidence", type=float, default=0.5, help="detection/tracking confidence")
    pose.add_argument("--min-visibility", type=float, default=0.5, help="hide keypoints below this when drawing")
    pose.add_argument("--no-video", action="store_true", help="for videos, save only JSON/CSV/NPY")

    morph = sub.add_parser("morph", help="morphological skeleton (medial axis / thinning) of shapes")
    _add_common(morph)
    morph.add_argument("--method", choices=["zhang", "lee", "medial", "thin"], default="zhang",
                       help="skeletonization algorithm (default: zhang)")
    morph.add_argument("--threshold", type=int, help="fixed binarization threshold 0-255 (default: Otsu)")
    morph.add_argument("--invert", action="store_true", help="object is darker than background")
    morph.add_argument("--blur", type=int, default=3, help="Gaussian blur kernel before thresholding (0=off)")
    morph.add_argument("--min-size", type=int, default=64, help="drop blobs smaller than N pixels")
    morph.add_argument("--prune", type=int, default=0, help="remove spur branches up to N pixels long")
    morph.add_argument("--thickness", type=int, default=1, help="drawn skeleton line thickness")
    return parser


def run_pose(args) -> None:
    from .pose import PoseConfig, PoseExtractor, process_image, process_video

    cfg = PoseConfig(model=args.model, model_path=args.model_path, num_poses=args.num_poses,
                     min_detection_confidence=args.min_confidence, min_presence_confidence=args.min_confidence,
                     min_tracking_confidence=args.min_confidence, min_visibility=args.min_visibility)
    items = collect_inputs(args.inputs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    image_extractor = None
    try:
        for path, kind in items:
            if kind == "image":
                if image_extractor is None:
                    image_extractor = PoseExtractor(cfg, video=False)
                r = process_image(path, out, cfg, image_extractor)
                print(f"[pose] {path}: {len(r['people'])} person(s)")
            else:
                r = process_video(path, out, cfg, args.stride, save_video=not args.no_video)
                hit = sum(1 for f in r["frames"] if f["people"])
                print(f"[pose] {path}: {hit}/{len(r['frames'])} frames with a person")
    finally:
        if image_extractor is not None:
            image_extractor.landmarker.close()


def run_morph(args) -> None:
    from .morph import MorphConfig, process_image, process_video

    cfg = MorphConfig(method=args.method, threshold=args.threshold, invert=args.invert, blur=args.blur,
                      min_size=args.min_size, prune=args.prune)
    items = collect_inputs(args.inputs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for path, kind in items:
        if kind == "image":
            r = process_image(path, out, cfg, args.thickness)
            print(f"[morph] {path}: {r['skeleton_pixels']} px, {len(r['endpoints'])} endpoints, "
                  f"{len(r['branchpoints'])} branchpoints")
        else:
            r = process_video(path, out, cfg, args.stride, args.thickness)
            print(f"[morph] {path}: {len(r['frames'])} frames")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "pose":
            run_pose(args)
        else:
            run_morph(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
