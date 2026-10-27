"""Video file -> RawTrack -> HumanDemo, plus debug visualisations."""

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from robot_arm.video.calibrate import Camera, calibrate
from robot_arm.video.reconstruct import HumanDemo, RawTrack, reconstruct
from robot_arm.video.tracking import HandTracker, detect_blob

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def read_video(path: str | Path, max_width: int = 1280) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > max_width:
            s = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames could be read from {path}")
    return frames, fps


def track(frames: list[np.ndarray], fps: float, ball_color: str, cup_color: str) -> RawTrack:
    T = len(frames)
    hand_px = np.full((T, 21, 2), np.nan)
    hand_world = np.full((T, 21, 3), np.nan)
    ball_px = np.full((T, 2), np.nan)
    ball_r = np.full(T, np.nan)
    cup_px = np.full((T, 2), np.nan)
    tracker = HandTracker()
    last_ball = None
    for i, frame in enumerate(frames):
        det = tracker(frame, int(round(i * 1000 / fps)))
        if det is not None:
            hand_px[i], hand_world[i] = det.image_px, det.world_m
        jump = 0.15 * frame.shape[1] if last_ball is not None else None
        ball = detect_blob(frame, ball_color, near=last_ball, max_jump=jump)
        if ball is not None:
            ball_px[i], ball_r[i], last_ball = ball.center, ball.radius, ball.center
        cup = detect_blob(frame, cup_color, min_area=200)
        if cup is not None:
            cup_px[i] = cup.bottom
    tracker.close()
    return RawTrack(np.arange(T) / fps, hand_px, hand_world, ball_px, ball_r, cup_px)


def process_video(path: str | Path, out_dir: str | Path, ball_color: str = "orange", cup_color: str = "blue",
                  ball_radius: float = 0.02, cup_radius: float = 0.04, marker_mm: float = 50.0,
                  focal_px: float | None = None, hfov_deg: float | None = None,
                  grasp_time: float | None = None, release_time: float | None = None,
                  debug: bool = True) -> tuple[HumanDemo, dict]:
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, fps = read_video(path)
    camera, calib_info = calibrate(frames[: int(fps)], marker_mm=marker_mm, focal_px=focal_px, hfov_deg=hfov_deg)
    raw = track(frames, fps, ball_color, cup_color)
    demo = reconstruct(raw, camera, ball_radius=ball_radius, cup_radius=cup_radius,
                       grasp_time=grasp_time, release_time=release_time)
    demo.info.update({f"calib_{k}": float(v) for k, v in calib_info.items() if isinstance(v, (int, float))})
    demo.save(out_dir / f"{path.stem}.npz")
    np.savez(out_dir / f"{path.stem}_camera.npz", **camera.as_dict())
    if debug:
        save_overlay(frames, raw, demo, camera, out_dir / f"{path.stem}_overlay.mp4", fps)
        save_plot(demo, out_dir / f"{path.stem}_trajectory.png")
    return demo, calib_info


def save_overlay(frames, raw: RawTrack, demo: HumanDemo, camera: Camera, path: Path, fps: float) -> None:
    """Video with landmarks, detections, table axes and the reconstructed (corrected) pinch trajectory."""
    first = int(np.argmax(~np.isnan(raw.hand_px).any(axis=(1, 2))))
    traj_px = camera.project(demo.pinch)
    axes = camera.project(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]))
    out = []
    for i, frame in enumerate(frames):
        img = frame.copy()
        o = tuple(int(v) for v in axes[0])
        for k, color in zip(range(1, 4), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(img, o, tuple(int(v) for v in axes[k]), color, 2)
        cv2.circle(img, tuple(int(v) for v in camera.project(demo.ball)[0]), 6, (0, 140, 255), 2)
        cv2.circle(img, tuple(int(v) for v in camera.project(demo.cup)[0]), 6, (255, 80, 0), 2)
        if not np.isnan(raw.hand_px[i]).any():
            pts = raw.hand_px[i].astype(int)
            for a, b in HAND_EDGES:
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), (200, 200, 200), 1)
        j = i - first
        if 0 <= j < len(demo.t):
            cv2.polylines(img, [traj_px[: j + 1].astype(np.int32)], False, (0, 255, 255), 2)
            state = "HOLD" if demo.closed[j] else "open"
            cv2.putText(img, f"t={demo.t[j]:.2f}s  {state}  z={demo.pinch[j, 2]:.3f}m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, out, fps=fps, macro_block_size=1)


def save_plot(demo: HumanDemo, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*demo.pinch.T, color="#1f77b4", label="pinch point")
    held = demo.closed
    ax.plot(*demo.pinch[held].T, color="#d62728", lw=3, label="holding ball")
    ax.scatter(*demo.ball, color="orange", s=60, label="ball")
    ax.scatter(*demo.cup, color="blue", s=60, label="cup")
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)"), ax.set_zlabel("z (m)")
    ax.legend(loc="upper left", fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(demo.t, demo.aperture * 100, label="thumb-index aperture (cm)")
    ax2.plot(demo.t, demo.pinch[:, 2] * 100, label="pinch height (cm)")
    for tt, name in ((demo.t_grasp, "grasp"), (demo.t_release, "release")):
        ax2.axvline(tt, color="gray", ls="--")
        ax2.text(tt, ax2.get_ylim()[1] * 0.95, name, rotation=90, va="top")
    ax2.set_xlabel("time (s)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
