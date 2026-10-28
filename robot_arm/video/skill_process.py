"""Skill clip (video file) -> SkillTrack -> SkillDemo, plus debug visualisations.

All cups have the same color (red party cups), so the cup that moves is told apart by what changes: the still
frames at the start and at the end (hand out of view) give the red pixels that are there in both (cups that stay
put), the ones only there at the start (where the moved cup was) and the ones only there at the end (where it
went). In between, the moved cup is the red that is not where a static cup is. The toss clips track the cap.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from robot_arm.video.calibrate import Camera, calibrate
from robot_arm.video.cups import cup_centre, long_axis, two_cup_centres
from robot_arm.video.process import HAND_EDGES, read_video
from robot_arm.video.skills import Props, SkillDemo, SkillTrack, reconstruct_skill
from robot_arm.video.tracking import HandTracker, color_mask, detect_blob

SKILLS = ("place_base", "place_top", "toss")


@dataclass
class SkillColors:
    cups: str = "red"
    cap: str = "blue"


def skill_from_name(path: str | Path) -> str:
    name = Path(path).stem.lower()
    for skill in SKILLS:
        if name.startswith(skill):
            return skill
    raise ValueError(f"Cannot tell the skill of {Path(path).name}: name clips place_base_*, place_top_* or toss_* "
                     "(or pass --skill).")


def largest_component(mask: np.ndarray, min_area: int = 200) -> np.ndarray | None:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n < 2 or stats[1:, cv2.CC_STAT_AREA].max() < min_area:
        return None
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def still_mask(frames: list[np.ndarray], color: str) -> np.ndarray:
    """Pixels of the color in most of the given (still) frames."""
    votes = sum((color_mask(f, color) > 0).astype(np.int32) for f in frames)
    return votes * 2 > len(frames)


def track_skill(frames: list[np.ndarray], fps: float, skill: str, colors: SkillColors, camera: Camera,
                props: Props = Props(), hands: dict | None = None) -> SkillTrack:
    """Hand landmarks per frame (or `hands`, cached: hand_px, hand_world), the moved cup's blob centre (place) or
    the cap's centre (toss), and the marks from the still frames at the start and end (see SkillTrack)."""
    T = len(frames)
    still = max(3, int(0.3 * fps))
    first, last = still_mask(frames[:still], colors.cups), still_mask(frames[-still:], colors.cups)
    marks, in_cup_px, static = {}, None, None
    kernel = np.ones((7, 7), np.uint8)
    on_table = lambda c: None if c is None else np.array([c[0], c[1], 0.0])  # noqa: E731
    if skill == "toss":
        pyramid = largest_component(first)
        if pyramid is None:
            raise RuntimeError("No cups found in the first frames; check the cup color (--check).")
        marks["axis"] = long_axis(pyramid, camera)
        ys, xs = np.nonzero(pyramid)
        in_cup_px = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))[:, 0, :].astype(float)
    else:
        static = cv2.dilate((first & last).astype(np.uint8), kernel) > 0
        start = largest_component(cv2.morphologyEx((first & ~last).astype(np.uint8), cv2.MORPH_OPEN, kernel))
        end = largest_component(cv2.morphologyEx((last & ~first).astype(np.uint8), cv2.MORPH_OPEN, kernel))
        # All cups start mouth-down: their outline is the mouth on the table.
        if start is not None:
            marks["start"] = on_table(cup_centre(start, camera, props.lip, props.mouth_radius))
        if skill == "place_base" and end is not None:
            marks["end"] = on_table(cup_centre(end, camera, props.lip, props.mouth_radius))
        if skill == "place_top":
            moved = cv2.dilate(start.astype(np.uint8), kernel) > 0 if start is not None else np.zeros_like(first)
            bases = largest_component(first & ~moved)
            pair = two_cup_centres(bases, camera, props.lip, props.mouth_radius) if bases is not None else None
            if pair is not None:
                a, b = (on_table(c) for c in pair)
                marks["base_a"], marks["base_b"] = a, b
                mid = (a + b) / 2
                marks["end"] = mid + [0, 0, props.top_bottom_z]  # the top cup's base on the pyramid
                marks["mouth"] = mid + [0, 0, props.top_bottom_z + props.cup_height]
                marks["axis"] = (b - a) / np.linalg.norm(b - a)

    # The cap's expected size in the image, to tell it from other blue things (a mat, a pen).
    ends = camera.project(np.array([[0.0, 0, 0], [0.01, 0, 0]]))
    px_per_m = float(np.linalg.norm(ends[1] - ends[0]) * 100)
    cap_area = np.pi * (props.cap_radius * px_per_m) ** 2
    hand_px = np.full((T, 21, 2), np.nan)
    hand_world = np.full((T, 21, 3), np.nan)
    if hands is not None:
        hand_px[:], hand_world[:] = hands["hand_px"][:T], hands["hand_world"][:T]
    obj_px = np.full((T, 2), np.nan)
    obj_r = np.full(T, np.nan)
    tracker = HandTracker() if hands is None else None
    last_cap, missing = None, 0
    for i, frame in enumerate(frames):
        if tracker is not None:
            det = tracker(frame, int(round(i * 1000 / fps)))
            if det is not None:
                hand_px[i], hand_world[i] = det.image_px, det.world_m
        if skill == "toss":
            near = last_cap if missing < int(0.1 * fps) else None  # search everywhere after losing it
            cap = detect_blob(frame, colors.cap, min_area=0.2 * cap_area, max_area=4 * cap_area, near=near,
                              max_jump=0.12 * frame.shape[1], compact=True)
            if cap is not None:
                obj_px[i], obj_r[i], last_cap, missing = cap.center, cap.radius, cap.center, 0
            else:
                missing += 1
        else:
            moving = ((color_mask(frame, colors.cups) > 0) & ~static).astype(np.uint8) * 255
            cup = detect_blob(frame, colors.cups, min_area=300, mask=moving)
            if cup is not None:
                obj_px[i], obj_r[i] = cup.center, cup.radius
    if tracker is not None:
        tracker.close()
    return SkillTrack(np.arange(T) / fps, hand_px, hand_world, obj_px, obj_r, marks, in_cup_px)


def process_skill_video(path: str | Path, out_dir: str | Path, skill: str | None = None,
                        colors: SkillColors = SkillColors(), props: Props = Props(), marker_mm: float = 50.0,
                        focal_px: float | None = None, hfov_deg: float | None = None, debug: bool = True,
                        hands: dict | None = None) -> tuple[SkillDemo, dict]:
    """`hands`: hand landmarks tracked earlier for this clip (hand_px, hand_world), to skip MediaPipe."""
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skill = skill or skill_from_name(path)
    frames, fps = read_video(path)
    camera, calib_info = calibrate(frames[: int(fps)], marker_mm=marker_mm, focal_px=focal_px, hfov_deg=hfov_deg)
    track = track_skill(frames, fps, skill, colors, camera, props, hands)
    demo = reconstruct_skill(track, camera, skill, props)
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
    """Video with landmarks, the tracked object, the marks, table axes, and the reconstructed palm path and
    palm frame."""
    palm_px = camera.project(demo.palm)
    axes = camera.project(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]))
    start_px, end_px = camera.project(demo.obj_start)[0], camera.project(demo.obj_end)[0]
    marks_px = {k: camera.project(np.asarray(v))[0] for k, v in track.marks.items() if k != "axis"}
    flight_px = None
    if demo.skill == "toss":
        dt = np.linspace(0, 0.5, 40)[:, None]
        flight = demo.release_pos + demo.release_vel * dt + 0.5 * np.array([0, 0, -9.81]) * dt ** 2
        flight_px = camera.project(flight[flight[:, 2] > demo.obj_end[2] - 0.02]).astype(np.int32)
    out = []
    for i, frame in enumerate(frames):
        img = frame.copy()
        for k, color in zip(range(1, 4), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(img, _pt(axes[0]), _pt(axes[k]), color, 2)
        cv2.drawMarker(img, _pt(start_px), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(img, _pt(end_px), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        for name, p in marks_px.items():
            cv2.circle(img, _pt(p), 5, (255, 255, 0), -1)
            cv2.putText(img, name, _pt(p + [6, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if track.in_cup_px is not None:
            cv2.polylines(img, [track.in_cup_px.astype(np.int32)], True, (255, 255, 0), 1)
        if flight_px is not None and demo.t[min(i, len(demo.t) - 1)] >= demo.t_release:
            cv2.polylines(img, [flight_px], False, (0, 140, 255), 2)
        if not np.isnan(track.obj_px[i, 0]):
            cv2.circle(img, _pt(track.obj_px[i]), 5, (255, 0, 255), -1)
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
    ax.scatter(*demo.obj_end, color="gold", s=60, label="top cup mouth" if demo.skill == "toss" else "cup end")
    if demo.skill == "toss":
        dt = np.linspace(0, 0.5, 30)[:, None]
        flight = demo.release_pos + demo.release_vel * dt + 0.5 * np.array([0, 0, -9.81]) * dt ** 2
        flight = flight[flight[:, 2] > demo.obj_end[2] - 0.02]
        ax.plot(*flight.T, color="orange", ls="--", label="cap flight (fit)")
        if demo.base_axis is not None:
            a = demo.obj_end + 0.15 * np.array([[-1.0], [1.0]]) * demo.base_axis
            ax.plot(*a.T, color="red", label="base cups' line")
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
