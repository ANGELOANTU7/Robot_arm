"""G1 cup-pyramid simulation: scene, control, success checks and the scripted experts."""

import mujoco
import numpy as np
import pytest

from robot_arm.humanoid.config import G1_CFG
from robot_arm.humanoid.controller import G1HandController, rotation_error
from robot_arm.humanoid.envs import OBS_DIM, G1PyramidEnv
from robot_arm.humanoid.objects import DOWN, UP, PyramidLayout, cap_in_cup, base_spacing, pyramid_standing
from robot_arm.humanoid.scene import G1Ids, build_model
from robot_arm.humanoid.scripted import PlaceExpert, hand_frame


@pytest.fixture(scope="module")
def model_and_ids():
    model = build_model()
    return model, G1Ids(model)


def test_pelvis_is_fixed(model_and_ids):
    model, _ = model_and_ids
    free = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    assert sorted(free) == ["cap_joint", "cup0_joint", "cup1_joint", "cup2_joint"]


def test_ik_reaches_side_grasp_poses(model_and_ids):
    model, ids = model_and_ids
    data = mujoco.MjData(model)
    data.qpos[:] = ids.ready_qpos
    ik = G1HandController(model, ids)
    R = hand_frame(0.0)
    for p in ([0.36, -0.24, 0.81], [0.40, -0.08, 0.90], [0.30, 0.10, 0.95]):
        q = ik.solve(data.qpos, np.array(p), R, q_start=ik.home, iters=300)
        pos, mat = ik.fk(data.qpos, q)
        assert np.linalg.norm(pos - p) < 0.01
        assert np.degrees(np.linalg.norm(rotation_error(mat, R))) < 5


def _set_cup(data, ids, k, pos, quat):
    a = ids.cup_qpos[k]
    data.qpos[a:a + 3] = pos
    data.qpos[a + 3:a + 7] = quat


def test_pyramid_and_ball_checks(model_and_ids):
    model, ids = model_and_ids
    data = mujoco.MjData(model)
    data.qpos[:] = ids.ready_qpos
    layout = PyramidLayout.make(G1_CFG.pyramid_center, G1_CFG.pyramid_yaw)
    for k, (pos, quat) in enumerate(zip(layout.targets(), (DOWN, DOWN, UP))):
        _set_cup(data, ids, k, pos, quat)
    b = ids.cap_qpos
    data.qpos[b:b + 3] = layout.top + [0, 0, 0.03]
    mujoco.mj_forward(model, data)
    assert np.isclose(np.linalg.norm(layout.base_a[:2] - layout.base_b[:2]), base_spacing())
    assert pyramid_standing(data, ids.cup_bodies)
    assert cap_in_cup(data, ids.cap_body, ids.cup_bodies[2], G1_CFG.cap_radius)

    # Knock the top cup off: no longer standing, ball no longer inside.
    _set_cup(data, ids, 2, layout.top + [0.15, 0, -G1_CFG.cup.height], UP)
    mujoco.mj_forward(model, data)
    assert not pyramid_standing(data, ids.cup_bodies)
    assert not cap_in_cup(data, ids.cap_body, ids.cup_bodies[2], G1_CFG.cap_radius)


def test_env_shapes_and_hold_still():
    env = G1PyramidEnv(task="place", stage=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and info["stage"] == 0
    start = env.site_pos
    for _ in range(10):
        obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    assert np.linalg.norm(env.site_pos - start) < 0.005  # a zero action holds the hand still


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_scripted_place_expert(stage):
    env = G1PyramidEnv(task="place", stage=stage)
    expert = PlaceExpert()
    wins = 0
    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        expert.reset()
        for _ in range(env.max_steps):
            obs, _, terminated, truncated, info = env.step(expert(obs))
            if terminated or truncated:
                break
        wins += info["is_success"]
    assert wins >= 2  # the expert succeeds ~90-100% of the time


def test_five_finger_hand_places_a_base_cup():
    from robot_arm.humanoid.config import task_config
    from robot_arm.humanoid.hands import SHADOW
    cfg = task_config("shadow")  # with the hand's own grip height and pyramid spot
    env = G1PyramidEnv(task="place", stage=1, cfg=cfg)
    assert env.ids.hand is SHADOW and len(env.ids.hand_ctrl) == 18 and env.observation_space.shape == (OBS_DIM,)
    expert = PlaceExpert(cfg)
    obs, _ = env.reset(seed=1000)
    for _ in range(env.max_steps):
        obs, _, terminated, truncated, info = env.step(expert(obs))
        if terminated or truncated:
            break
    assert info["is_success"]
