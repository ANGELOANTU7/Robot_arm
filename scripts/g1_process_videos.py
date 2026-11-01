"""Turn your skill clips for the cup pyramid into 3D human demonstrations (see docs/RECORDING_PYRAMID.md).

    python scripts/g1_process_videos.py data/videos_pyramid/*/*.MOV --focal-px 1740   # -> data/processed/human_pyramid/
    python scripts/g1_process_videos.py data/videos_pyramid/part_c/IMG_3783.MOV --check

The skill comes from the clip's folder (part_a: the two base cups, part_b: the top cup, part_c: the throw; see
--folders), from its name (place_base_*, place_top_*, toss_*) or from --skill. For every clip this writes
<name>.npz (the demonstration; a base clip in which both base cups are put down gives <name>_a.npz and
<name>_b.npz), <name>_overlay.mp4 (tracking and the reconstructed palm drawn on the video) and
<name>_trajectory.png (3D palm path, closure, grasp and release).

--focal-px is for the clip's full resolution (an iPhone 16 main camera at 1x filming 1080x1920: about 1740).
Frames are scaled down to --max-width for tracking; hand tracking runs at full resolution.
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
    p.add_argument("--skill", choices=SKILLS, help="skill of all given clips (default: from the folder or name)")
    p.add_argument("--folders", default="part_a=place_base,part_b=place_top,part_c=toss",
                   help="skill of the clips in each folder")
    p.add_argument("--max-width", type=int, default=720, help="track the colors on frames scaled to this width")
    p.add_argument("--hands", type=Path, help="folder of hand landmarks tracked earlier: <folder>_<clip>.npz")
    p.add_argument("--cup-color", default="red", choices=sorted(COLOR_RANGES))
    p.add_argument("--cap-color", default="blue", choices=sorted(COLOR_RANGES))
    p.add_argument("--cup-height-cm", type=float, default=12.0)
    p.add_argument("--cup-mouth-cm", type=float, default=9.0, help="outer diameter at the mouth (the rim)")
    p.add_argument("--cup-base-cm", type=float, default=5.0, help="outer diameter of the bottom")
    p.add_argument("--lip-mm", type=float, default=5.0, help="height of the white rolled rim at the mouth")
    p.add_argument("--cap-diameter-cm", type=float, default=2.8)
    p.add_argument("--cap-height-cm", type=float, default=1.2)
    p.add_argument("--marker-mm", type=float, default=50.0, help="measured side of a printed marker")
    p.add_argument("--focal-px", type=float, help="camera focal length in pixels (at the clip's resolution)")
    p.add_argument("--hfov-deg", type=float, help="camera horizontal field of view, if focal length unknown")
    p.add_argument("--check", action="store_true", help="only check markers and color masks on frame 0")
    p.add_argument("--handheld", action="store_true", help="the phone was held in the hand: warp every frame onto "
                   "a fixed camera using the marker sheet (it must stay in view most of the time)")
    args = p.parse_args()
    colors = SkillColors(args.cup_color, args.cap_color)
    props = Props(args.cup_height_cm / 100, args.cup_mouth_cm / 200, args.cup_base_cm / 200, args.lip_mm / 1000,
                  args.cap_diameter_cm / 200, args.cap_height_cm / 100)
    args.out.mkdir(parents=True, exist_ok=True)

    folders = dict(item.split("=") for item in args.folders.split(",")) if args.folders else {}
    summary = {}
    for video in args.videos:
        if args.check:
            check(video, args.out, colors)
            continue
        skill = args.skill or folders.get(video.parent.name)
        hands = None
        if args.hands is not None and (args.hands / f"{video.parent.name}_{video.stem}.npz").exists():
            z = np.load(args.hands / f"{video.parent.name}_{video.stem}.npz")
            hands = {"hand_px": z["hand_px"], "hand_world": z["hand_world"]}
        try:
            demos, calib = process_skill_video(video, args.out, skill, colors, props, marker_mm=args.marker_mm,
                                               focal_px=args.focal_px, hfov_deg=args.hfov_deg, hands=hands,
                                               max_width=args.max_width, handheld=args.handheld)
        except Exception as e:  # keep going with the other clips
            print(f"{video.name}: FAILED - {e}")
            summary[video.name] = {"error": str(e)}
            continue
        for n, demo in enumerate(demos):
            row = demo_row(demo, calib)
            name = video.name if len(demos) == 1 else f"{video.name} ({'AB'[n]})"
            summary[name] = row
            print(f"{name}: {row}")
    if not args.check:
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))


def demo_row(demo, calib) -> dict:
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
        row["throws"] = int(demo.info.get("throws", 1))
        if "thumbs_up_start" in demo.info:
            row["thumbs_up_s"] = round(demo.info["thumbs_up_end"] - demo.info["thumbs_up_start"], 2)
    return row


if __name__ == "__main__":
    main()
