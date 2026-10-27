"""Skill-clip video pipeline for the cup pyramid, on synthetic hands and cameras with known ground truth."""

import cv2
import imageio.v2 as imageio
import numpy as np
import pytest

import robot_arm.video.skill_process as skill_process
from robot_arm.humanoid.synthetic import (BALL_R, CUP_H, CUP_R, object_paths, place_demo, render_track,
                                          side_grasp_rot, toss_demo)
from robot_arm.video.hand_pose import closure, palm_frame
from robot_arm.video.skills import GRAVITY, SkillDemo, reconstruct_skill
from robot_arm.video.synthetic_hand import hand_landmarks
from robot_arm.video.tracking import HandDetection
from test_video import make_camera, render_sheet


def skill_camera():
    """Further back than the pinch clips' camera: the whole reach, the wind-up and the ball's flight are in view."""
    return make_camera(pos=(-0.05, -0.85, 0.55), look=(-0.06, 0.18, 0.12))


def angle_deg(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))


def test_palm_frame_and_closure():
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(50):
        from scipy.spatial.transform import Rotation

        R = Rotation.random(random_state=rng.integers(1 << 30)).as_matrix()
        curl = rng.uniform(0, 1)
        world = hand_landmarks(curl, R) + rng.normal(0, 0.002, (21, 3))
        est = palm_frame(world)
        assert abs(np.linalg.det(est) - 1) < 1e-6
        errs.append(angle_deg(est, R))
    assert np.median(errs) < 8, np.median(errs)
    # A side grasp: palm faces +x, the thumb is up (the frame's y axis, away from the thumb, points down).
    est = palm_frame(hand_landmarks(0.3, side_grasp_rot()))
    assert est[0, 2] > 0.9 and est[2, 1] < -0.9
    curls = np.linspace(0, 1, 6)
    values = [closure(hand_landmarks(c)) for c in curls]
    assert np.all(np.diff(values) >= 0) and values[-1] - values[0] > 0.4


@pytest.mark.parametrize("kind,size_error", [("A", 0.85), ("B", 1.0), ("top", 1.2)])
def test_reconstruct_place(kind, size_error):
    rng = np.random.default_rng(1)
    cam = skill_camera()
    truth, static = place_demo(rng, kind)
    demo = reconstruct_skill(render_track(truth, cam, rng, static, size_error=size_error), cam, truth.skill,
                             cup_height=CUP_H, cup_radius=CUP_R)
    assert demo.skill == truth.skill
    np.testing.assert_allclose(demo.obj_start, truth.obj_start, atol=0.008)
    np.testing.assert_allclose(demo.obj_end, truth.obj_end, atol=0.008)
    assert abs(demo.info["scale"] - size_error) < 0.04
    assert abs(demo.t_grasp - truth.t_grasp) < 0.25 and abs(demo.t_release - truth.t_release) < 0.25
    err = np.linalg.norm(demo.palm - truth.palm, axis=1)
    assert np.median(err) < 0.015, np.median(err)
    assert np.median([angle_deg(a, b) for a, b in zip(demo.palm_rot, truth.palm_rot)]) < 10


@pytest.mark.parametrize("seed,size_error", [(0, 0.85), (1, 1.0), (2, 1.2)])
def test_reconstruct_toss(seed, size_error):
    rng = np.random.default_rng(seed)
    cam = skill_camera()
    truth = toss_demo(rng)
    demo = reconstruct_skill(render_track(truth, cam, rng, size_error=size_error), cam, "toss",
                             cup_height=CUP_H, cup_radius=CUP_R, ball_radius=BALL_R)
    np.testing.assert_allclose(demo.obj_start, truth.obj_start, atol=0.008)
    np.testing.assert_allclose(demo.obj_end, truth.obj_end, atol=0.008)
    assert abs(demo.info["scale"] - size_error) < 0.04
    assert abs(demo.t_grasp - truth.t_grasp) < 0.25 and abs(demo.t_release - truth.t_release) < 0.06
    assert np.median(np.linalg.norm(demo.palm - truth.palm, axis=1)) < 0.015
    # The fitted flight matches the true one (compared at the true release time).
    g = np.array([0, 0, -GRAVITY])
    dt = truth.t_release - demo.t_release
    np.testing.assert_allclose(demo.release_pos + demo.release_vel * dt + 0.5 * g * dt ** 2, truth.release_pos,
                               atol=0.015)
    np.testing.assert_allclose(demo.release_vel + g * dt, truth.release_vel, atol=0.2)


def test_skill_demo_roundtrip(tmp_path):
    demo = toss_demo(np.random.default_rng(0))
    demo.save(tmp_path / "d.npz")
    back = SkillDemo.load(tmp_path / "d.npz")
    assert back.skill == "toss" and back.info == demo.info
    np.testing.assert_allclose(back.palm_rot, demo.palm_rot)
    np.testing.assert_allclose(back.release_pos, demo.release_pos)


CUP_BGR = {"blue": (200, 80, 20), "green": (40, 180, 40), "yellow": (0, 220, 240)}


def _cup_hull(cam, base):
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1) * CUP_R
    pts = np.concatenate([base + ring, base + ring + [0, 0, CUP_H]])
    return cv2.convexHull(cam.project(pts).astype(np.int32))


def render_skill_clip(truth: SkillDemo, cam, cups: dict, moved: str | None):
    """Frames of a clip: the sheet, solid colored cups (`moved` follows the demo) and the orange ball."""
    background = render_sheet(cam)
    path = object_paths(truth)
    r_idx = int(np.searchsorted(truth.t, truth.t_release))
    frames = []
    for i in range(len(truth.t)):
        img = background.copy()
        for color, base in cups.items():
            cv2.fillConvexPoly(img, _cup_hull(cam, path[i] if color == moved else base), CUP_BGR[color])
        if truth.skill == "toss" and not (i > r_idx and path[i, 2] < truth.obj_end[2] - 0.01):
            pc = cam.to_cam(path[i][None])[0]
            d = np.linalg.norm(pc)
            r = cam.K[0, 0] * BALL_R / (d * (pc[2] / d) ** 2)
            cv2.circle(img, tuple(cam.project(path[i])[0].astype(int)), int(round(r)), (0, 120, 255), -1)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


@pytest.mark.parametrize("skill", ["place_base", "toss"])
def test_process_skill_video_end_to_end(tmp_path, monkeypatch, skill):
    rng = np.random.default_rng(4)
    cam = skill_camera()
    if skill == "toss":
        truth, fps = toss_demo(rng), 60
        top = truth.obj_end - [0, 0, CUP_H]
        u = np.array([np.cos(-2.4), np.sin(-2.4), 0.0]) * (CUP_R + 0.004)
        cups, moved = {"blue": top - u - [0, 0, CUP_H], "green": top + u - [0, 0, CUP_H], "yellow": top}, None
    else:
        (truth, static), fps = place_demo(rng, "A"), 30
        cups, moved = {"blue": truth.obj_start, "green": static[0]}, "blue"
    video = tmp_path / f"{skill}_01.mp4"
    imageio.mimsave(video, render_skill_clip(truth, cam, cups, moved), fps=fps, macro_block_size=1, quality=9)

    class FakeTracker:
        def __init__(self, *a, **k):
            pass

        def __call__(self, frame, timestamp_ms):
            i = int(round(timestamp_ms * fps / 1000))
            pc = cam.to_cam(hand_landmarks(truth.closure[i], truth.palm_rot[i], truth.palm[i]))
            uv = pc @ cam.K.T
            return HandDetection(uv[:, :2] / uv[:, 2:3], (pc - pc.mean(axis=0)) / 1.1, 1.0, "Right")

        def close(self):
            pass

    monkeypatch.setattr(skill_process, "HandTracker", FakeTracker)
    demo, calib = skill_process.process_skill_video(video, tmp_path / "out", focal_px=cam.K[0, 0],
                                                    cup_height=CUP_H, cup_radius=CUP_R, ball_radius=BALL_R)
    assert calib["markers_found"] == 4 and demo.skill == skill
    np.testing.assert_allclose(demo.obj_start, truth.obj_start, atol=0.015)
    np.testing.assert_allclose(demo.obj_end, truth.obj_end, atol=0.015)
    assert abs(demo.info["scale"] - 1.1) < 0.06
    assert abs(demo.t_grasp - truth.t_grasp) < 0.3 and abs(demo.t_release - truth.t_release) < 0.3
    if skill == "toss":
        assert np.linalg.norm(demo.release_vel - truth.release_vel) < 0.35
    loaded = SkillDemo.load(tmp_path / "out" / f"{skill}_01.npz")
    np.testing.assert_allclose(loaded.palm, demo.palm)
    assert (tmp_path / "out" / f"{skill}_01_overlay.mp4").exists()
    assert (tmp_path / "out" / f"{skill}_01_trajectory.png").exists()
