"""Video pipeline tests on synthetic cameras and hands with known ground truth."""

import cv2
import numpy as np
import pytest

from robot_arm.synthetic import synthetic_demo
from robot_arm.video.calibrate import SHEET_MM, Camera, calibrate, make_marker_sheet
from robot_arm.video.lift3d import aperture, hand_translation
from robot_arm.video.reconstruct import RawTrack, reconstruct
from robot_arm.video.tracking import INDEX_TIP, THUMB_TIP

W, H, F = 1280, 720, 1000.0


def make_camera(pos=(0.05, -0.45, 0.5), look=(0.0, 0.12, 0.03)) -> Camera:
    pos, look = np.array(pos, float), np.array(look, float)
    fwd = (look - pos) / np.linalg.norm(look - pos)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
    return Camera(K, R, -R @ pos, W, H)


def render_sheet(cam: Camera, dpi: int = 100) -> np.ndarray:
    sheet = make_marker_sheet(dpi=dpi)
    mm_per_px = 25.4 / dpi
    # sheet pixel (px, py, 1) -> table (x, y, 1)
    A = np.array([[mm_per_px / 1000, 0, -SHEET_MM[0] / 2000],
                  [0, -mm_per_px / 1000, SHEET_MM[1] / 2000],
                  [0, 0, 1]])
    Hm = cam.K @ np.column_stack([cam.R[:, 0], cam.R[:, 1], cam.t]) @ A
    img = cv2.warpPerspective(sheet, Hm, (W, H), borderValue=110)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_calibration_recovers_camera():
    cam = make_camera()
    est, info = calibrate([render_sheet(cam)])
    assert info["markers_found"] == 4
    assert abs(est.K[0, 0] - F) / F < 0.05
    np.testing.assert_allclose(est.center_table, cam.center_table, atol=0.02)
    # Back-projection of a table point lands where it should.
    p = np.array([0.08, -0.03, 0.0])
    np.testing.assert_allclose(est.backproject_to_plane(cam.project(p)[0]), p, atol=0.005)


# A rigid 21-point hand (metres, hand frame centred near the pinch point), thumb and index tips set per frame.
_rng = np.random.default_rng(0)
HAND_TEMPLATE = np.concatenate([
    [[0.0, -0.11, 0.06]],  # wrist
    _rng.uniform([-0.05, -0.09, 0.02], [0.05, -0.01, 0.07], (20, 3)),
])


def synth_hand(pinch: np.ndarray, ap: float, cam: Camera, size_error: float):
    pts = pinch + HAND_TEMPLATE.copy()
    pts[THUMB_TIP] = pinch + [-ap / 2, 0, 0]
    pts[INDEX_TIP] = pinch + [ap / 2, 0, 0]
    pc = cam.to_cam(pts)
    uv = pc @ cam.K.T
    image_px = uv[:, :2] / uv[:, 2:3]
    world = (pc - pc.mean(axis=0)) / size_error  # MediaPipe assumes the wrong hand size
    return image_px, world


def test_hand_translation_exact():
    cam = make_camera()
    image_px, world = synth_hand(np.array([0.0, 0.0, 0.1]), 0.06, cam, size_error=1.0)
    pc = cam.to_cam(np.array([0.0, 0.0, 0.1]) + HAND_TEMPLATE)
    T = hand_translation(image_px, world, cam.K)
    pinch_cam = world[[THUMB_TIP, INDEX_TIP]].mean(axis=0) + T
    np.testing.assert_allclose(cam.to_table(pinch_cam[None])[0], [0.0, 0.0, 0.1], atol=1e-6)
    assert abs(aperture(world) - 0.06) < 1e-9
    assert pc.shape == (21, 3)


def synth_raw_track(demo, cam: Camera, size_error: float, cup_radius: float, pixel_noise: float, rng) -> RawTrack:
    T = len(demo.t)
    hand_px, hand_world = np.zeros((T, 21, 2)), np.zeros((T, 21, 3))
    for i in range(T):
        hand_px[i], hand_world[i] = synth_hand(demo.pinch[i], demo.aperture[i], cam, size_error)
    hand_px += rng.normal(0, pixel_noise, hand_px.shape)
    hand_world += rng.normal(0, 0.002, hand_world.shape)
    # The ball sits still, moves with the pinch while held, then disappears into the cup.
    ball_px = np.broadcast_to(cam.project(demo.ball)[0], (T, 2)).copy()
    held = np.nonzero(demo.closed)[0]
    lift = held[0] + int(0.1 * len(held))
    ball_px[lift:held[-1]] = cam.project(demo.pinch[lift:held[-1]])
    ball_px[held[-1]:] = np.nan
    r_px = F * 0.02 / np.linalg.norm(cam.to_cam(demo.ball[None])[0])
    view = demo.cup - cam.center_table
    view[2] = 0
    cup_front = demo.cup - cup_radius * view / np.linalg.norm(view)
    cup_px = np.broadcast_to(cam.project(cup_front)[0], (T, 2)).copy()
    hand_px[5:7] = np.nan  # a couple of dropped detections
    return RawTrack(demo.t, hand_px, hand_world, ball_px, np.full(T, r_px), cup_px)


@pytest.mark.parametrize("size_error", [0.85, 1.0, 1.2])
def test_reconstruct_recovers_demo(size_error):
    rng = np.random.default_rng(3)
    cam = make_camera()
    truth = synthetic_demo(rng)
    raw = synth_raw_track(truth, cam, size_error, cup_radius=0.04, pixel_noise=1.0, rng=rng)
    demo = reconstruct(raw, cam, ball_radius=0.02, cup_radius=0.04)

    np.testing.assert_allclose(demo.ball, truth.ball, atol=0.004)
    np.testing.assert_allclose(demo.cup, truth.cup, atol=0.004)
    assert abs(demo.info["scale"] - size_error) < 0.05
    assert abs(demo.t_grasp - truth.t_grasp) < 0.2
    assert abs(demo.t_release - truth.t_release) < 0.2
    n = min(len(demo.t), len(truth.t))
    err = np.linalg.norm(demo.pinch[:n] - truth.pinch[:n], axis=1)
    assert np.median(err) < 0.015, np.median(err)
