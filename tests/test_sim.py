import mujoco
import numpy as np
import pytest

from robot_arm.config import CFG
from robot_arm.controller import IKController
from robot_arm.env import OBS_DIM, PandaBallCupEnv, sample_layout
from robot_arm.scene import SceneIds, build_model
from robot_arm.scripted import ScriptedPolicy


@pytest.fixture(scope="module")
def env():
    return PandaBallCupEnv()


def test_tcp_between_fingertips():
    model = build_model()
    data = mujoco.MjData(model)
    ids = SceneIds(model)
    mujoco.mj_resetDataKeyframe(model, data, ids.home_key)
    mujoco.mj_forward(model, data)
    pads = [data.geom_xpos[g] for g in ids.finger_geoms
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and model.geom_size[g][0] > 0.008]
    assert len(pads) == 2
    np.testing.assert_allclose(np.mean(pads, axis=0), data.site_xpos[ids.tcp_site], atol=2e-3)


def test_ik_reaches_targets():
    model = build_model()
    data = mujoco.MjData(model)
    ids = SceneIds(model)
    ik = IKController(model, ids, iters=50)
    rng = np.random.default_rng(0)
    mujoco.mj_resetDataKeyframe(model, data, ids.home_key)
    for _ in range(10):
        target = rng.uniform([0.3, -0.35, 0.03], [0.7, 0.35, 0.4])
        q = ik.solve(data, target)
        data.qpos[ids.arm_joints] = q
        mujoco.mj_forward(model, data)
        assert np.linalg.norm(data.site_xpos[ids.tcp_site] - target) < 5e-3
        # Gripper still points straight down.
        assert data.xmat[ids.hand_body].reshape(3, 3)[2, 2] < -0.99


def test_reset_and_step_shapes(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert not info["is_success"]
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (OBS_DIM,)
    assert np.isfinite(reward)


def test_layouts_respect_workspace():
    rng = np.random.default_rng(1)
    for _ in range(100):
        ball, cup = sample_layout(rng)
        for xy in (ball, cup):
            assert CFG.workspace_x[0] <= xy[0] <= CFG.workspace_x[1]
            assert CFG.workspace_y[0] <= xy[1] <= CFG.workspace_y[1]
        assert CFG.min_ball_cup_dist <= np.linalg.norm(ball - cup) <= CFG.max_ball_cup_dist


def test_success_detector(env):
    env.reset_to([0.45, -0.15], [0.5, 0.15])
    assert not env.is_success()
    # Drop the ball into the cup by hand.
    d, ids = env.data, env.ids
    d.qpos[ids.ball_qpos:ids.ball_qpos + 3] = [0.5, 0.15, 0.06]
    d.qvel[ids.ball_dof:ids.ball_dof + 6] = 0
    mujoco.mj_forward(env.model, d)
    for _ in range(10):
        env.step(np.array([0, 0, 0, -1.0]))
    assert env.is_success()


def test_scripted_policy_solves_task(env):
    policy = ScriptedPolicy()
    wins = 0
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        policy.reset()
        for _ in range(env.cfg.max_episode_steps):
            obs, _, terminated, truncated, info = env.step(policy(obs))
            if info["is_success"] or terminated or truncated:
                break
        wins += info["is_success"]
    assert wins >= 9
