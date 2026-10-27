import numpy as np

from robot_arm.config import CFG
from robot_arm.env import PandaBallCupEnv
from robot_arm.retarget import canonical_layout, retarget, rollout, similarity_2d
from robot_arm.synthetic import synthetic_demo


def test_similarity_maps_both_anchors():
    M, c = similarity_2d([0.1, 0.2], [0.3, -0.1], [0.5, -0.2], [0.45, 0.15])
    np.testing.assert_allclose(M @ [0.1, 0.2] + c, [0.5, -0.2], atol=1e-12)
    np.testing.assert_allclose(M @ [0.3, -0.1] + c, [0.45, 0.15], atol=1e-12)
    # A similarity: equal singular values (no shear or anisotropic scaling).
    s = np.linalg.svd(M, compute_uv=False)
    assert abs(s[0] - s[1]) < 1e-12


def test_plan_respects_anchors_and_limits():
    demo = synthetic_demo(np.random.default_rng(7))
    ball, cup = np.array([0.45, -0.2]), np.array([0.55, 0.15])
    start = np.array([0.55, 0.0, 0.5])
    plan = retarget(demo, ball, cup, start)
    np.testing.assert_allclose(plan.waypoints[0], start, atol=1e-9)
    np.testing.assert_allclose(plan.waypoints[plan.grasp_idx], [*ball, CFG.ball_radius], atol=0.01)
    over_cup = plan.waypoints[plan.release_idx]
    assert np.linalg.norm(over_cup[:2] - cup) < 0.01
    assert over_cup[2] >= CFG.cup_height + CFG.ball_radius + 0.03 - 1e-9
    steps = np.linalg.norm(np.diff(plan.waypoints, axis=0), axis=1)
    assert steps.max() < CFG.max_step  # never faster than the arm can go
    assert 0 < plan.grasp_idx < plan.release_idx < len(plan.waypoints)


def test_retargeted_demo_succeeds_in_sim():
    env = PandaBallCupEnv()
    rng = np.random.default_rng(0)
    wins = 0
    for i in range(3):
        demo = synthetic_demo(np.random.default_rng(20 + i))
        obs, actions, success, _ = rollout(env, demo, *canonical_layout(demo), rng, seed=i)
        assert obs.shape[0] == actions.shape[0] and actions.shape[1] == 4
        assert np.abs(actions).max() <= 1.0
        wins += success
    assert wins >= 2
