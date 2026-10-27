"""Skill clip (video file) -> SkillTrack -> SkillDemo, plus debug visualisations.

Each cup has its own solid color so the cups stay apart in the image even when they touch: base cup A, base
cup B and the top cup. The toss clips also track the ball.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from robot_arm.video.calibrate import Camera, calibrate
from robot_arm.video.process import HAND_EDGES, read_video
from robot_arm.video.skills import SkillDemo, SkillTrack, reconstruct_skill
from robot_arm.video.tracking import HandTracker, detect_blob

SKILLS = ("place_base", "place_top", "toss")


@dataclass
class SkillColors:
    cup_a: str = "blue"
    cup_b: str = "green"
    top: str = "yellow"
    ball: str = "orange"


def skill_from_name(path: str | Path) -> str:
    name = Path(path).stem.lower()
    for skill in SKILLS:
        if name.startswith(skill):
            return skill
    raise ValueError(f"Cannot tell the skill of {Path(path).name}: name clips place_base_*, place_top_* or toss_* "
                     "(or pass --skill).")


def track_skill(frames: list[np.ndarray], fps: float, skill: str, colors: SkillColors) -> SkillTrack:
    """Hand landmarks and object blobs per frame. Object columns: [ball (toss only)], cup A, cup B, top cup.

    Cups are reported by the lowest pixel of their blob, the ball by its centre."""
    T = len(frames)
    cups = [colors.cup_a, colors.cup_b, colors.top]
    n = len(cups) + (skill == "toss")
    hand_px = np.full((T, 21, 2), np.nan)
    hand_world = np.full((T, 21, 3), np.nan)
    obj_px = np.full((T, n, 2), np.nan)
    obj_r = np.full((T, n), np.nan)
    tracker = HandTracker()
    last_ball, missing = None, 0
    for i, frame in enumerate(frames):
        det = tracker(frame, int(round(i * 1000 / fps)))
        if det is not None:
            hand_px[i], hand_world[i] = det.image_px, det.world_m
        j = 0
        if skill == "toss":
            near = last_ball if missing < int(0.1 * fps) else None  # search everywhere after losing it
            ball = detect_blob(frame, colors.ball, near=near, max_jump=0.12 * frame.shape[1])
            if ball is not None:
                obj_px[i, 0], obj_r[i, 0], last_ball, missing = ball.center, ball.radius, ball.center, 0
            else:
                missing += 1
            j = 1
        for k, color in enumerate(cups):
            cup = detect_blob(frame, color, min_area=200)
            if cup is not None:
                obj_px[i, j + k], obj_r[i, j + k] = cup.bottom, cup.radius
    tracker.close()
    return SkillTrack(np.arange(T) / fps, hand_px, hand_world, obj_px, obj_r)


def process_skill_video(path: str | Path, out_dir: str | Path, skill: str | None = None,
                        colors: SkillColors = SkillColors(), cup_height: float = 0.075, cup_radius: float = 0.031,
                        ball_radius: float = 0.02, ball_rest_height: float | None = None, marker_mm: float = 50.0,
                        focal_px: float | None = None, hfov_deg: float | None = None,
                        debug: bool = True) -> tuple[SkillDemo, dict]:
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skill = skill or skill_from_name(path)
    frames, fps = read_video(path)
    camera, calib_info = calibrate(frames[: int(fps)], marker_mm=marker_mm, focal_px=focal_px, hfov_deg=hfov_deg)
    track = track_skill(frames, fps, skill, colors)
    # Place clips: the cup that moved. Toss clips: the ball (column 0); the top cup is the last column.
    demo = reconstruct_skill(track, camera, skill, cup_height=cup_height, cup_radius=cup_radius,
                             ball_radius=ball_radius, ball_rest_height=ball_rest_height,
                             object_index=0 if skill == "toss" else None)
    demo.info.update({f"calib_{k}": float(v) for k, v in calib_info.items() if isinstance(v, (int, float))})
    demo.save(out_dir / f"{path.stem}.npz")
    np.savez(out_dir / f"{path.stem}_camera.npz", **camera.as_dict())
    if debug:
        save_skill_overlay(frames, track, demo, camera, out_dir / f"{path.stem}_overlay.mp4", fps)
        save_skill_plot(demo, out_dir / f"{path.stem}_trajectory.png")
    return demo, calib_info


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def save_skill_overlay(frames, track: SkillTrack, demo: SkillDemo, camera: Camera, path: Path, fps: float) -> None:
    """Video with landmarks, object detections, table axes, and the reconstructed palm path and palm frame."""
    palm_px = camera.project(demo.palm)
    axes = camera.project(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]))
    start_px, end_px = camera.project(demo.obj_start)[0], camera.project(demo.obj_end)[0]
    out = []
    for i, frame in enumerate(frames):
        img = frame.copy()
        for k, color in zip(range(1, 4), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(img, _pt(axes[0]), _pt(axes[k]), color, 2)
        cv2.drawMarker(img, _pt(start_px), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(img, _pt(end_px), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        for k in range(track.objects_px.shape[1]):
            if not np.isnan(track.objects_px[i, k, 0]):
                cv2.circle(img, _pt(track.objects_px[i, k]), 5, (255, 0, 255), -1)
        if not np.isnan(track.hand_px[i]).any():
            pts = track.hand_px[i]
            for a, b in HAND_EDGES:
                cv2.line(img, _pt(pts[a]), _pt(pts[b]), (200, 200, 200), 1)
        if i < len(demo.t):
            cv2.polylines(img, [palm_px[: i + 1].astype(np.int32)], False, (0, 255, 255), 2)
            frame_px = camera.project(demo.palm[i] + 0.06 * demo.palm_rot[i].T)
            for k, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # fingers, thumb side, normal
                cv2.line(img, _pt(palm_px[i]), _pt(frame_px[k]), color, 2)
            held = demo.t_grasp <= demo.t[i] < demo.t_release
            cv2.putText(img, f"{demo.skill}  t={demo.t[i]:.2f}s  closure={demo.closure[i]:.2f}  "
                             f"{'HOLD' if held else ''}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, out, fps=fps, macro_block_size=1)


def save_skill_plot(demo: SkillDemo, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    held = (demo.t >= demo.t_grasp) & (demo.t < demo.t_release)
    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*demo.palm.T, color="#1f77b4", label="palm centre")
    ax.plot(*demo.palm[held].T, color="#d62728", lw=3, label="holding")
    ax.scatter(*demo.obj_start, color="gray", s=60, label="object start")
    ax.scatter(*demo.obj_end, color="gold", s=60, label="cup mouth" if demo.skill == "toss" else "object end")
    if demo.skill == "toss":
        dt = np.linspace(0, 0.5, 30)[:, None]
        flight = demo.release_pos + demo.release_vel * dt + 0.5 * np.array([0, 0, -9.81]) * dt ** 2
        flight = flight[flight[:, 2] > 0]
        ax.plot(*flight.T, color="orange", ls="--", label="ball flight (fit)")
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)"), ax.set_zlabel("z (m)")
    ax.legend(loc="upper left", fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(demo.t, demo.closure, label="hand closure (0 open, 1 closed)")
    ax2.plot(demo.t, demo.palm[:, 2] * 5, label="palm height (x5, m)")
    for tt, name in ((demo.t_grasp, "grasp"), (demo.t_release, "release")):
        ax2.axvline(tt, color="gray", ls="--")
        ax2.text(tt, ax2.get_ylim()[1] * 0.95, name, rotation=90, va="top")
    ax2.set_xlabel("time (s)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
