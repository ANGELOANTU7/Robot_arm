"""Skill-clip video pipeline for the cup pyramid, on synthetic hands and cameras with known ground truth."""

import cv2
import imageio.v2 as imageio
import numpy as np
import pytest

import robot_arm.video.skill_process as skill_process
from robot_arm.humanoid.synthetic import (H, PROPS, cap_hidden, look_at_camera, object_paths, place_demo,
                                          render_track, side_grasp_rot, toss_demo)
from robot_arm.video.hand_pose import closure, palm_frame
from robot_arm.video.skills import GRAVITY, SkillDemo, reconstruct_skill
from robot_arm.video.synthetic_hand import hand_landmarks
from robot_arm.video.tracking import HandDetection
from test_video import render_sheet


def skill_camera():
    """Further back than the pinch clips' camera: the supply cups, the pyramid, the reach above them and the
    cap's flight are in view."""
    return look_at_camera()


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
    demo = reconstruct_skill(render_track(truth, cam, rng, static, size_error=size_error), cam, truth.skill, PROPS)
    assert demo.skill == truth.skill
    np.testing.assert_allclose(demo.obj_start, truth.obj_start, atol=0.008)
    np.testing.assert_allclose(demo.obj_end, truth.obj_end, atol=0.008)
    assert abs(demo.info["scale"] - size_error) < 0.04
    assert abs(demo.t_grasp - truth.t_grasp) < 0.25 and abs(demo.t_release - truth.t_release) < 0.25
    err = np.linalg.norm(demo.palm - truth.palm, axis=1)
    assert np.median(err) < 0.015, np.median(err)
    assert np.median([angle_deg(a, b) for a, b in zip(demo.palm_rot, truth.palm_rot)]) < 10


@pytest.mark.parametrize("seed,size_error", [(0, 0.85), (1, 1.0), (2, 1.2), (3, 1.0)])
def test_reconstruct_toss(seed, size_error):
    rng = np.random.default_rng(seed)
    cam = skill_camera()
    truth, static = toss_demo(rng)
    demo = reconstruct_skill(render_track(truth, cam, rng, static, size_error=size_error), cam, "toss", PROPS)
    np.testing.assert_allclose(demo.obj_start, truth.obj_start, atol=0.008)
    np.testing.assert_allclose(demo.obj_end, truth.obj_end, atol=0.008)
    assert abs(demo.info["scale"] - size_error) < 0.04
    assert abs(demo.t_grasp - truth.t_grasp) < 0.25 and abs(demo.t_release - truth.t_release) < 0.06
    assert np.median(np.linalg.norm(demo.palm - truth.palm, axis=1)) < 0.015
    # The base cups' line (its sign is arbitrary).
    assert abs(np.dot(demo.base_axis, truth.base_axis)) > 0.99
    # The fitted flight matches the true one (compared at the true release time).
    g = np.array([0, 0, -GRAVITY])
    dt = truth.t_release - demo.t_release
    np.testing.assert_allclose(demo.release_pos + demo.release_vel * dt + 0.5 * g * dt ** 2, truth.release_pos,
                               atol=0.02)
    np.testing.assert_allclose(demo.release_vel + g * dt, truth.release_vel, atol=0.15)


def test_skill_demo_roundtrip(tmp_path):
    demo, _ = toss_demo(np.random.default_rng(0))
    demo.save(tmp_path / "d.npz")
    back = SkillDemo.load(tmp_path / "d.npz")
    assert back.skill == "toss" and back.info == demo.info
    np.testing.assert_allclose(back.palm_rot, demo.palm_rot)
    np.testing.assert_allclose(back.release_pos, demo.release_pos)
    np.testing.assert_allclose(back.base_axis, demo.base_axis)


RED, WHITE, BLUE = (40, 25, 200), (235, 235, 235), (200, 80, 20)  # BGR


def _ring(base, z, r):
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    return np.asarray(base) + np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, z)], 1)


def _fill(img, cam, pts, color):
    cv2.fillConvexPoly(img, cv2.convexHull(cam.project(pts).astype(np.int32)), color)


def draw_cup(img, cam, base, mouth_up):
    """A red party cup standing with its table-side rim centred at `base`: white rolled lip at the mouth, white
    inside."""
    r, lip = PROPS.radius_at, PROPS.lip
    if mouth_up:
        _fill(img, cam, np.concatenate([_ring(base, 0, r(0, True)), _ring(base, H - lip, r(H - lip, True))]), RED)
        _fill(img, cam, _ring(base, H, PROPS.mouth_radius), WHITE)
    else:
        _fill(img, cam, np.concatenate([_ring(base, 0, PROPS.mouth_radius), _ring(base, lip, r(lip, False))]), WHITE)
        _fill(img, cam, np.concatenate([_ring(base, lip, r(lip, False)), _ring(base, H, r(H, False))]), RED)


def render_skill_clip(truth: SkillDemo, cam, static):
    """Frames of a clip: the sheet, the static cups, the moved cup following the demo, and the blue cap."""
    background = render_sheet(cam)
    path = object_paths(truth)
    hidden = cap_hidden(truth, path) if truth.skill == "toss" else None
    cups = [(base, up) for _, base, up in static]
    if truth.skill == "toss":
        cups.append((truth.obj_end - [0, 0, H], True))
    frames = []
    for i in range(len(truth.t)):
        img = background.copy()
        scene = cups if truth.skill == "toss" else cups + [(path[i], truth.skill == "place_top")]
        # Far cups first, cups higher up (the top cup) over the ones below.
        for base, up in sorted(scene, key=lambda c: (c[0][2], -np.linalg.norm(c[0][:2] - cam.center_table[:2]))):
            draw_cup(img, cam, base, up)
        if truth.skill == "toss" and not hidden[i]:
            c = path[i] - [0, 0, PROPS.cap_height / 2]
            _fill(img, cam, np.concatenate([_ring(c, 0, PROPS.cap_radius), _ring(c, PROPS.cap_height,
                                                                                  PROPS.cap_radius)]), BLUE)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


@pytest.mark.parametrize("skill,kind", [("place_base", "A"), ("place_base", "B"), ("place_top", "top"),
                                        ("toss", None)])
def test_process_skill_video_end_to_end(tmp_path, monkeypatch, skill, kind):
    rng = np.random.default_rng(4)
    cam = skill_camera()
    if skill == "toss":
        (truth, static), fps = toss_demo(rng), 60
    else:
        (truth, static), fps = place_demo(rng, kind), 30
    video = tmp_path / f"{skill}_01.mp4"
    imageio.mimsave(video, render_skill_clip(truth, cam, static), fps=fps, macro_block_size=1, quality=9)

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
    demo, calib = skill_process.process_skill_video(video, tmp_path / "out", focal_px=cam.K[0, 0], props=PROPS)
    assert calib["markers_found"] == 4 and demo.skill == skill
    np.testing.assert_allclose(demo.obj_start, truth.obj_start, atol=0.015)
    np.testing.assert_allclose(demo.obj_end, truth.obj_end, atol=0.015)
    assert abs(demo.info["scale"] - 1.1) < 0.06
    assert abs(demo.t_grasp - truth.t_grasp) < 0.3 and abs(demo.t_release - truth.t_release) < 0.3
    if skill == "toss":
        assert abs(np.dot(demo.base_axis, truth.base_axis)) > 0.98
        g = np.array([0, 0, -GRAVITY])
        assert np.linalg.norm(demo.release_vel + g * (truth.t_release - demo.t_release) - truth.release_vel) < 0.25
    loaded = SkillDemo.load(tmp_path / "out" / f"{skill}_01.npz")
    np.testing.assert_allclose(loaded.palm, demo.palm)
    assert (tmp_path / "out" / f"{skill}_01_overlay.mp4").exists()
    assert (tmp_path / "out" / f"{skill}_01_trajectory.png").exists()
