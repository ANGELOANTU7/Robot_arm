"""End-to-end video processing on a rendered synthetic clip (MediaPipe replaced by the known hand)."""

import cv2
import imageio.v2 as imageio
import numpy as np

import robot_arm.video.process as process
from robot_arm.synthetic import synthetic_demo
from robot_arm.video.reconstruct import HumanDemo
from robot_arm.video.tracking import HandDetection
from test_video import make_camera, render_sheet, synth_hand


def render_clip(demo: HumanDemo, cam, cup_radius=0.04, cup_height=0.1):
    background = render_sheet(cam)
    # Cup silhouette: convex hull of points on its outer surface.
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1) * cup_radius
    cup_pts = np.concatenate([demo.cup + ring, demo.cup + ring + [0, 0, cup_height]])
    hull = cv2.convexHull(cam.project(cup_pts).astype(np.int32))
    held = np.nonzero(demo.closed)[0]
    frames = []
    for i in range(len(demo.t)):
        img = background.copy()
        cv2.fillConvexPoly(img, hull, (200, 80, 20))  # blue (BGR)
        if i < held[0] + 2:
            ball = demo.ball
        elif i < held[-1]:
            ball = demo.pinch[i]
        else:
            ball = None  # inside the cup
        if ball is not None:
            r = cam.K[0, 0] * 0.02 / np.linalg.norm(cam.to_cam(ball[None])[0])
            cv2.circle(img, tuple(cam.project(ball)[0].astype(int)), int(r), (0, 120, 255), -1)  # orange
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def test_process_video_end_to_end(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    cam = make_camera()
    truth = synthetic_demo(rng, fps=30.0)
    video = tmp_path / "clip.mp4"
    imageio.mimsave(video, render_clip(truth, cam), fps=30, macro_block_size=1, quality=9)

    class FakeTracker:
        def __init__(self, *a, **k):
            pass

        def __call__(self, frame, timestamp_ms):
            i = int(round(timestamp_ms * 30 / 1000))
            px, world = synth_hand(truth.pinch[i], truth.aperture[i], cam, size_error=1.1)
            return HandDetection(px, world, 1.0, "Right")

        def close(self):
            pass

    monkeypatch.setattr(process, "HandTracker", FakeTracker)
    demo, calib = process.process_video(video, tmp_path / "out", focal_px=cam.K[0, 0])
    assert calib["markers_found"] == 4
    np.testing.assert_allclose(demo.ball, truth.ball, atol=0.01)
    np.testing.assert_allclose(demo.cup[:2], truth.cup[:2], atol=0.015)
    assert abs(demo.t_grasp - truth.t_grasp) < 0.25 and abs(demo.t_release - truth.t_release) < 0.25
    loaded = HumanDemo.load(tmp_path / "out" / "clip.npz")
    np.testing.assert_allclose(loaded.pinch, demo.pinch)
    assert (tmp_path / "out" / "clip_overlay.mp4").exists()
    assert (tmp_path / "out" / "clip_trajectory.png").exists()
