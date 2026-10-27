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


def lowest_px(blob: np.ndarray) -> np.ndarray:
    """Lowest pixel of a blob at its centroid column (a standing cup: its rim point nearest the camera)."""
    ys, xs = np.nonzero(blob)
    cx = xs.mean()
    col = np.abs(xs - cx) < 2.0
    return np.array([cx, ys[col].max() if col.any() else ys.max()], float)


def rim_px(blob: np.ndarray, top_frac: float = 0.25) -> np.ndarray:
    """The top cup's mouth: the centre of the widest row in the top part of the pyramid's blob. A mouth-up cup's
    red wall is widest just under its white lip, at the ends of the rim, level with the mouth's centre."""
    ys, xs = np.nonzero(blob)
    top, bottom = ys.min(), ys.max()
    best, row = -1, top
    for y in range(top, top + max(3, int(top_frac * (bottom - top))) + 1):
        cols = xs[ys == y]
        if len(cols) and cols.max() - cols.min() > best + 1:  # ties go to the higher row
            best, row = cols.max() - cols.min(), y
    cols = xs[ys == row]
    return np.array([(cols.min() + cols.max()) / 2, row], float)


def base_pair_px(blob: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Lowest pixels of two mouth-down cups standing side by side (one blob): the two low points of its bottom
    edge, with a notch between them where the rims meet. None if there aren't two."""
    ys, xs = np.nonzero(blob)
    cols = np.arange(xs.min(), xs.max() + 1)
    bottom = np.full(len(cols), -np.inf)
    np.maximum.at(bottom, xs - xs.min(), ys)
    bottom = np.convolve(bottom, np.ones(5) / 5, mode="same")
    i1 = int(np.argmax(bottom[2:-2])) + 2
    away = np.abs(np.arange(len(cols)) - i1) > 0.25 * len(cols)
    if not away[2:-2].any():
        return None
    i2 = int(np.argmax(np.where(away, bottom, -np.inf)[2:-2])) + 2
    lo, hi = sorted((i1, i2))
    near = bottom[max(2, i2 - 4):i2 + 5]
    if bottom[i2] < near.max() - 0.5 or bottom[lo:hi + 1].min() > min(bottom[i1], bottom[i2]) - 3:
        return None  # i2 is only the slope down to i1, or there is no notch between them
    pts = []
    for i in sorted((i1, i2)):
        c = cols[i]
        pts.append(np.array([c, ys[np.abs(xs - c) < 2.0].max()], float))
    return pts[0], pts[1]


def still_mask(frames: list[np.ndarray], color: str) -> np.ndarray:
    """Pixels of the color in most of the given (still) frames."""
    votes = sum((color_mask(f, color) > 0).astype(np.int32) for f in frames)
    return votes * 2 > len(frames)


def track_skill(frames: list[np.ndarray], fps: float, skill: str, colors: SkillColors) -> SkillTrack:
    """Hand landmarks per frame, the moved cup's lowest pixel (place) or the cap's centre (toss), and the marks
    from the still frames at the start and end (see SkillTrack)."""
    T = len(frames)
    still = max(3, int(0.3 * fps))
    first, last = still_mask(frames[:still], colors.cups), still_mask(frames[-still:], colors.cups)
    marks = {}
    kernel = np.ones((7, 7), np.uint8)
    if skill == "toss":
        pyramid = largest_component(first)
        if pyramid is None:
            raise RuntimeError("No cups found in the first frames; check the cup color (--check).")
        marks["rim"] = rim_px(pyramid)
        static = None
    else:
        static = cv2.dilate((first & last).astype(np.uint8), kernel) > 0
        start = largest_component(cv2.morphologyEx((first & ~last).astype(np.uint8), cv2.MORPH_OPEN, kernel))
        end = largest_component(cv2.morphologyEx((last & ~first).astype(np.uint8), cv2.MORPH_OPEN, kernel))
        if start is not None:
            marks["start"] = lowest_px(start)
        if skill == "place_base" and end is not None:
            marks["end"] = lowest_px(end)
        pyramid = largest_component(first & last) if skill == "place_top" else None  # the two base cups
        if skill == "place_top" and end is not None:
            full = largest_component(last)
            if full is not None and (full & end).any():
                marks["rim"] = rim_px(full)
    if pyramid is not None and (pair := base_pair_px(pyramid)) is not None:
        marks["base_a"], marks["base_b"] = pair

    hand_px = np.full((T, 21, 2), np.nan)
    hand_world = np.full((T, 21, 3), np.nan)
    obj_px = np.full((T, 2), np.nan)
    obj_r = np.full(T, np.nan)
    tracker = HandTracker()
    last_cap, missing = None, 0
    for i, frame in enumerate(frames):
        det = tracker(frame, int(round(i * 1000 / fps)))
        if det is not None:
            hand_px[i], hand_world[i] = det.image_px, det.world_m
        if skill == "toss":
            near = last_cap if missing < int(0.1 * fps) else None  # search everywhere after losing it
            cap = detect_blob(frame, colors.cap, near=near, max_jump=0.12 * frame.shape[1], compact=True)
            if cap is not None:
                obj_px[i], obj_r[i], last_cap, missing = cap.center, cap.radius, cap.center, 0
            else:
                missing += 1
        else:
            moving = ((color_mask(frame, colors.cups) > 0) & ~static).astype(np.uint8) * 255
            cup = detect_blob(frame, colors.cups, min_area=300, mask=moving)
            if cup is not None:
                obj_px[i], obj_r[i] = cup.bottom, cup.radius
    tracker.close()
    return SkillTrack(np.arange(T) / fps, hand_px, hand_world, obj_px, obj_r, marks)


def process_skill_video(path: str | Path, out_dir: str | Path, skill: str | None = None,
                        colors: SkillColors = SkillColors(), props: Props = Props(), marker_mm: float = 50.0,
                        focal_px: float | None = None, hfov_deg: float | None = None,
                        debug: bool = True) -> tuple[SkillDemo, dict]:
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skill = skill or skill_from_name(path)
    frames, fps = read_video(path)
    camera, calib_info = calibrate(frames[: int(fps)], marker_mm=marker_mm, focal_px=focal_px, hfov_deg=hfov_deg)
    track = track_skill(frames, fps, skill, colors)
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
        for name, p in track.marks.items():
            cv2.circle(img, _pt(p), 4, (255, 255, 0), -1)
            cv2.putText(img, name, _pt(p + [6, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
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
