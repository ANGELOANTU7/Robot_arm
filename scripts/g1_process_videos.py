"""Turn your skill clips for the cup pyramid into 3D human demonstrations (see docs/RECORDING_PYRAMID.md).

    python scripts/g1_process_videos.py data/videos_pyramid/*.mp4          # -> data/processed/human_pyramid/
    python scripts/g1_process_videos.py data/videos_pyramid/toss_01.mp4 --check

Clips are named by skill: place_base_*.mp4, place_top_*.mp4, toss_*.mp4 (or pass --skill). For every clip this
writes <name>.npz (the demonstration), <name>_overlay.mp4 (tracking and the reconstructed palm drawn on the video)
and <name>_trajectory.png (3D palm path, closure, grasp and release).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from robot_arm.video.calibrate import detect_markers
from robot_arm.video.process import read_video
from robot_arm.video.skill_process import SKILLS, SkillColors, process_skill_video
from robot_arm.video.tracking import COLOR_RANGES, color_mask

OVERLAY_BGR = {"cup_a": (255, 0, 0), "cup_b": (0, 255, 0), "top": (0, 255, 255), "ball": (0, 140, 255)}


def check(video: Path, out_dir: Path, colors: SkillColors) -> None:
    """Save the first frame with marker detections and each object's color mask highlighted."""
    frames, _ = read_video(video)
    frame = frames[0].copy()
    markers = detect_markers(frame)
    for mid, c in markers.items():
        cv2.polylines(frame, [c.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str(mid), tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    report = []
    for name, bgr in OVERLAY_BGR.items():
        mask = color_mask(frames[0], getattr(colors, name)) > 0
        frame[mask] = (0.3 * frame[mask] + 0.7 * np.array(bgr)).astype(np.uint8)
        report.append(f"{name} ({getattr(colors, name)}): {int(mask.sum())} px")
    out = out_dir / f"{video.stem}_check.png"
    cv2.imwrite(str(out), frame)
    print(f"{video.name}: markers found {sorted(markers)} (need at least 2, ideally 4); " + ", ".join(report)
          + f". Each object should be one solid patch in {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/processed/human_pyramid"))
    p.add_argument("--skill", choices=SKILLS, help="skill of all given clips (default: from the file name)")
    p.add_argument("--cup-a-color", default="blue", choices=sorted(COLOR_RANGES))
    p.add_argument("--cup-b-color", default="green", choices=sorted(COLOR_RANGES))
    p.add_argument("--top-color", default="yellow", choices=sorted(COLOR_RANGES))
    p.add_argument("--ball-color", default="orange", choices=sorted(COLOR_RANGES))
    p.add_argument("--cup-height-cm", type=float, default=7.5)
    p.add_argument("--cup-diameter-cm", type=float, default=6.2, help="outer diameter at the mouth (the rim)")
    p.add_argument("--ball-diameter-cm", type=float, default=4.0)
    p.add_argument("--ball-rest-height-cm", type=float, default=None,
                   help="height of the ball's centre at the start of toss clips (default: lying on the table)")
    p.add_argument("--marker-mm", type=float, default=50.0, help="measured side of a printed marker")
    p.add_argument("--focal-px", type=float, help="camera focal length in pixels (at the processed width)")
    p.add_argument("--hfov-deg", type=float, help="camera horizontal field of view, if focal length unknown")
    p.add_argument("--check", action="store_true", help="only check markers and color masks on frame 0")
    args = p.parse_args()
    colors = SkillColors(args.cup_a_color, args.cup_b_color, args.top_color, args.ball_color)
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for video in args.videos:
        if args.check:
            check(video, args.out, colors)
            continue
        try:
            demo, calib = process_skill_video(
                video, args.out, args.skill, colors, cup_height=args.cup_height_cm / 100,
                cup_radius=args.cup_diameter_cm / 200, ball_radius=args.ball_diameter_cm / 200,
                ball_rest_height=None if args.ball_rest_height_cm is None else args.ball_rest_height_cm / 100,
                marker_mm=args.marker_mm, focal_px=args.focal_px, hfov_deg=args.hfov_deg)
        except Exception as e:  # keep going with the other clips
            print(f"{video.name}: FAILED - {e}")
            summary[video.name] = {"error": str(e)}
            continue
        row = {"skill": demo.skill, "seconds": round(float(demo.t[-1]), 2),
               "grasp_s": round(demo.t_grasp, 2), "release_s": round(demo.t_release, 2),
               "scale": round(demo.info["scale"], 3), "anchor_err_cm": round(100 * demo.info["grasp_anchor_err"], 1),
               "markers": int(calib["markers_found"])}
        if demo.skill != "toss":
            row["cup_moved_cm"] = round(100 * float(np.linalg.norm(demo.obj_end[:2] - demo.obj_start[:2])), 1)
        else:
            row["release_speed_mps"] = round(float(np.linalg.norm(demo.release_vel)), 2)
            row["release_to_mouth_cm"] = round(100 * float(np.linalg.norm(demo.release_pos - demo.obj_end)), 1)
        summary[video.name] = row
        print(f"{video.name}: {row}")
    if not args.check:
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
