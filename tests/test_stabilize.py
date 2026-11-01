"""Hand-held clips: frames from a moving camera are warped onto the first frame's camera."""

import cv2
import numpy as np

from robot_arm.video.calibrate import detect_markers
from robot_arm.video.stabilize import stabilize
from test_video import F, H, W, make_camera, render_sheet


def test_stabilize_moving_camera():
    rng = np.random.default_rng(0)
    point = np.array([0.20, 0.06, 0.0])  # a spot on the table, off the sheet
    frames, cams = [], []
    pos, look = np.array([0.05, -0.45, 0.5]), np.array([0.0, 0.12, 0.03])
    phase = rng.uniform(0, 2 * np.pi, (2, 3))
    for i in range(24):  # a hand-held phone at 30 fps: it sways a couple of centimetres and turns a few degrees
        t = i / 30
        sway = 0.015 * np.sin(2 * np.pi * 1.5 * t + phase[0])
        turn = 0.03 * np.sin(2 * np.pi * 2.0 * t + phase[1])
        cam = make_camera(pos + sway, look + turn)
        img = render_sheet(cam)
        cv2.circle(img, tuple(np.round(cam.project(point)[0]).astype(int)), 6, (0, 0, 255), -1)
        frames.append(img)
        cams.append(cam)
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
    warped, K_canvas, info = stabilize(frames, K, fps=30.0, pad=0.1)
    assert info["stab_pose_frames"] == 1.0 and info["stab_max_shift_px"] > 5
    # The sheet stands still in the warped frames, and the table spot is where the reference camera sees it.
    centres = np.array([np.mean([c.mean(0) for c in detect_markers(f).values()], 0) for f in warped])
    assert np.ptp(centres, axis=0).max() < 2.5
    ref = cams[info["stab_ref_frame"]]
    shift = np.array([K_canvas[0, 2] - K[0, 2], K_canvas[1, 2] - K[1, 2]])
    expected = ref.project(point)[0] + shift
    for f in warped:
        red = (f[..., 2] > 200) & (f[..., 1] < 60) & (f[..., 0] < 60)
        ys, xs = np.nonzero(red)
        assert np.linalg.norm([xs.mean() - expected[0], ys.mean() - expected[1]]) < 3.0
