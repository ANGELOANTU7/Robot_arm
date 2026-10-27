"""Turn your recordings into 3D human demonstrations.

    python scripts/process_video.py data/videos/*.mp4                   # -> data/processed/human/
    python scripts/process_video.py data/videos/clip1.mp4 --check       # only check markers and colors

For every clip this writes <name>.npz (the demonstration), <name>_overlay.mp4 (tracking drawn on the video) and
<name>_trajectory.png (3D path, aperture, detected grasp/release). Look at the overlay and plot once per clip:
the yellow path should follow your fingertips and HOLD should cover the time you hold the ball.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from robot_arm.video.calibrate import detect_markers
from robot_arm.video.process import process_video, read_video
from robot_arm.video.tracking import COLOR_RANGES, color_mask


def parse_hsv(text: str) -> list:
    v = [int(x) for x in text.split(",")]
    if len(v) != 6:
        raise argparse.ArgumentTypeError("expected H,S,V,H,S,V (lower then upper, OpenCV ranges)")
    return [(tuple(v[:3]), tuple(v[3:]))]


def check(video: Path, out_dir: Path, ball_color: str, cup_color: str) -> None:
    """Save the first frame with marker detections and color masks highlighted."""
    frames, _ = read_video(video)
    frame = frames[0].copy()
    markers = detect_markers(frame)
    for mid, c in markers.items():
        cv2.polylines(frame, [c.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str(mid), tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    for color, bgr in ((ball_color, (0, 140, 255)), (cup_color, (255, 0, 0))):
        mask = color_mask(frames[0], color) > 0
        frame[mask] = (0.4 * frame[mask] + 0.6 * np.array(bgr)).astype(np.uint8)
    out = out_dir / f"{video.stem}_check.png"
    cv2.imwrite(str(out), frame)
    print(f"{video.name}: markers found {sorted(markers)} (need at least 2, ideally 4). "
          f"Ball pixels in orange, cup pixels in blue: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/processed/human"))
    p.add_argument("--ball-color", default="orange", choices=sorted(COLOR_RANGES))
    p.add_argument("--cup-color", default="blue", choices=sorted(COLOR_RANGES))
    p.add_argument("--ball-hsv", type=parse_hsv, help="custom ball HSV range H,S,V,H,S,V")
    p.add_argument("--cup-hsv", type=parse_hsv, help="custom cup HSV range H,S,V,H,S,V")
    p.add_argument("--ball-diameter-cm", type=float, default=4.0)
    p.add_argument("--cup-diameter-cm", type=float, default=8.0, help="outer diameter at the base")
    p.add_argument("--marker-mm", type=float, default=50.0, help="measured side of a printed marker")
    p.add_argument("--focal-px", type=float, help="camera focal length in pixels (at the processed width)")
    p.add_argument("--hfov-deg", type=float, help="camera horizontal field of view, if focal length unknown")
    p.add_argument("--grasp-time", type=float, help="override: seconds from the hand's first appearance")
    p.add_argument("--release-time", type=float, help="override: seconds from the hand's first appearance")
    p.add_argument("--check", action="store_true", help="only check markers and color masks on frame 0")
    args = p.parse_args()

    if args.ball_hsv:
        COLOR_RANGES["custom_ball"], args.ball_color = args.ball_hsv, "custom_ball"
    if args.cup_hsv:
        COLOR_RANGES["custom_cup"], args.cup_color = args.cup_hsv, "custom_cup"
    args.out.mkdir(parents=True, exist_ok=True)

    for video in args.videos:
        if args.check:
            check(video, args.out, args.ball_color, args.cup_color)
            continue
        try:
            demo, calib = process_video(
                video, args.out, ball_color=args.ball_color, cup_color=args.cup_color,
                ball_radius=args.ball_diameter_cm / 200, cup_radius=args.cup_diameter_cm / 200,
                marker_mm=args.marker_mm, focal_px=args.focal_px, hfov_deg=args.hfov_deg,
                grasp_time=args.grasp_time, release_time=args.release_time)
        except RuntimeError as e:
            print(f"{video.name}: FAILED - {e}")
            continue
        summary = {k: round(v, 3) for k, v in demo.info.items()}
        print(f"{video.name}: grasp {demo.t_grasp:.2f}s, release {demo.t_release:.2f}s, "
              f"ball {np.round(demo.ball, 3).tolist()}, cup {np.round(demo.cup, 3).tolist()}")
        print("   " + json.dumps(summary))
        if not 0.7 < demo.info["scale"] < 1.4:
            print("   WARNING: hand-size scale is unusual; check the camera focal length (--hfov-deg/--focal-px).")
        if demo.info["release_offset_m"] > 0.05:
            print("   WARNING: the hand was >5 cm from the cup at release; check the overlay video.")


if __name__ == "__main__":
    main()
