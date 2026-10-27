"""Retargeting human skill demos onto the G1."""

import numpy as np
import pytest

from robot_arm.humanoid.envs import G1PyramidEnv
from robot_arm.humanoid.retarget import (TOSS_BOUNDS, G1Retargeter, Obstacle, place_plan_for_obs, plan_place,
                                         rollout, throw_style, toss_params)
from robot_arm.humanoid.synthetic import place_demo, toss_demo


def test_place_plan_constraints():
    demo, _ = place_demo(np.random.default_rng(0), "B")
    site0 = np.array([0.24, -0.22, 0.97])
    grasp, place = np.array([0.37, -0.36, 0.81]), np.array([0.36, -0.12, 0.81])
    neighbour = Obstacle(np.array([0.41, -0.07]), 0.835)
    plan = plan_place(demo, site0, grasp, place, [neighbour], Obstacle(grasp[:2], 0.835), hang=0.047)
    wp = plan.waypoints
    assert np.linalg.norm(wp[0] - site0) < 0.02  # starts at the robot's hand
    np.testing.assert_allclose(wp[plan.grasp_idx], grasp, atol=0.01)
    # The approach slides in along the fingers (+x): the last 12 cm are behind the cup at grasp height.
    approach = wp[: plan.grasp_idx + 1]
    close = np.linalg.norm(approach - grasp, axis=1) < 0.12
    assert np.all(approach[close, 0] <= grasp[0] + 0.005)
    assert np.all(np.abs(approach[close, 1:] - grasp[1:]).max(axis=1) < 0.01)
    # While carrying, the held cup's bottom clears the neighbouring cup.
    carry = wp[plan.grasp_idx + 1: plan.over_idx]
    near = np.linalg.norm(carry[:, :2] - neighbour.xy, axis=1) < 0.10
    assert np.all(carry[near, 2] - 0.047 > neighbour.top)
    assert plan.grasp_idx < plan.over_idx <= plan.release_idx < len(wp)
    # Motion speed is within what the arm follows.
    assert np.max(np.linalg.norm(np.diff(wp, axis=0), axis=1)) <= 0.25 * 0.05 + 1e-6


def test_throw_style_matches_the_throw():
    from robot_arm.humanoid.synthetic import PROPS, look_at_camera, render_track
    from robot_arm.video.skills import reconstruct_skill

    rng = np.random.default_rng(1)
    truth, scene = toss_demo(rng)
    # The synthetic person releases 5-30 cm from the mouth along the base cups (within 0.25 rad), 4-12 cm above
    # it, with a 0.22-0.34 s flight: the truth's style is exact.
    style = throw_style(truth)
    d = truth.release_pos - truth.obj_end
    assert np.isclose(style["release_dist"], np.linalg.norm(d[:2])) and np.isclose(style["release_height"], d[2])
    assert 0.05 <= style["release_dist"] <= 0.30 and abs(style["release_bearing"]) <= 0.25
    assert 0.22 <= style["flight_time"] <= 0.34
    # Measured through the video pipeline it is close (a release found a frame early or late is a point a couple
    # of centimetres away on the same arc).
    cam = look_at_camera()
    demo = reconstruct_skill(render_track(truth, cam, rng, scene), cam, "toss", PROPS)
    measured = throw_style(demo)
    for k, tol in (("release_dist", 0.02), ("release_height", 0.02), ("release_bearing", 0.1),
                   ("flight_time", 0.03), ("windup", 0.02)):
        assert abs(measured[k] - style[k]) < tol, (k, measured[k], style[k])
    params = toss_params(measured)
    for k, (lo, hi) in TOSS_BOUNDS.items():
        assert lo <= params[k] <= hi


@pytest.mark.parametrize("stage,kind", [(0, "A"), (2, "top")])
def test_retargeted_place_succeeds(stage, kind):
    env = G1PyramidEnv(task="place", stage=stage)
    wins = 0
    for i in range(3):
        demo, _ = place_demo(np.random.default_rng(i), kind)
        *_, done, info, _ = rollout(env, G1Retargeter({stage: demo}, None), np.random.default_rng(i), seed=500 + i)
        wins += info["is_success"]
    assert wins >= 2


def test_place_plan_from_obs_follows_env_layout():
    env = G1PyramidEnv(task="place", stage=1)
    obs, _ = env.reset(seed=3)
    demo, _ = place_demo(np.random.default_rng(3), "B")
    plan = place_plan_for_obs(demo, obs)
    from robot_arm.humanoid.objects import cup_grasp_point
    from robot_arm.humanoid.objects import cup_axis

    grasp = cup_grasp_point(env.cup_pos(1), cup_axis(env.data, env.ids.cup_bodies[1]), env.cfg.grasp_height_frac)
    np.testing.assert_allclose(plan.waypoints[plan.grasp_idx], grasp, atol=0.01)
