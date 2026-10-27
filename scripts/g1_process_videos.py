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
from robot_arm.video.skills import Props
from robot_arm.video.tracking import COLOR_RANGES, color_mask

OVERLAY_BGR = {"cups": (0, 255, 0), "cap": (255, 255, 0)}


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
          + f". In {out} the cups should be solid green and the cap solid cyan, with nothing else lit up")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/processed/human_pyramid"))
    p.add_argument("--skill", choices=SKILLS, help="skill of all given clips (default: from the file name)")
    p.add_argument("--cup-color", default="red", choices=sorted(COLOR_RANGES))
    p.add_argument("--cap-color", default="blue", choices=sorted(COLOR_RANGES))
    p.add_argument("--cup-height-cm", type=float, default=12.0)
    p.add_argument("--cup-mouth-cm", type=float, default=9.0, help="outer diameter at the mouth (the rim)")
    p.add_argument("--cup-base-cm", type=float, default=5.0, help="outer diameter of the bottom")
    p.add_argument("--lip-mm", type=float, default=5.0, help="height of the white rolled rim at the mouth")
    p.add_argument("--cap-diameter-cm", type=float, default=2.8)
    p.add_argument("--cap-height-cm", type=float, default=1.2)
    p.add_argument("--marker-mm", type=float, default=50.0, help="measured side of a printed marker")
    p.add_argument("--focal-px", type=float, help="camera focal length in pixels (at the processed width)")
    p.add_argument("--hfov-deg", type=float, help="camera horizontal field of view, if focal length unknown")
    p.add_argument("--check", action="store_true", help="only check markers and color masks on frame 0")
    args = p.parse_args()
    colors = SkillColors(args.cup_color, args.cap_color)
    props = Props(args.cup_height_cm / 100, args.cup_mouth_cm / 200, args.cup_base_cm / 200, args.lip_mm / 1000,
                  args.cap_diameter_cm / 200, args.cap_height_cm / 100)
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for video in args.videos:
        if args.check:
            check(video, args.out, colors)
            continue
        try:
            demo, calib = process_skill_video(video, args.out, args.skill, colors, props, marker_mm=args.marker_mm,
                                              focal_px=args.focal_px, hfov_deg=args.hfov_deg)
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
            d = demo.release_pos - demo.obj_end
            row["release_speed_mps"] = round(float(np.linalg.norm(demo.release_vel)), 2)
            row["release_dist_cm"] = round(100 * float(np.linalg.norm(d[:2])), 1)
            row["release_above_mouth_cm"] = round(100 * float(d[2]), 1)
            if demo.base_axis is not None:  # 0 = released straight along the base cups' line
                cos = abs(np.dot(d[:2], demo.base_axis[:2])) / max(np.linalg.norm(d[:2]), 1e-6)
                row["off_line_deg"] = round(float(np.degrees(np.arccos(np.clip(cos, 0, 1)))), 1)
        summary[video.name] = row
        print(f"{video.name}: {row}")
    if not args.check:
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
