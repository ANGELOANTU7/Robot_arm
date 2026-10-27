"""Retarget human skill demonstrations (SkillDemo) onto the G1.

Placing a cup. The human's hand path is mapped with two anchors, as for the Panda:
- Near the grasp it is expressed relative to the cup it picks up. Near the release it is expressed relative to
  where the cup was put down. During the carry the two blend.
- Each anchor map turns the human's heading into the robot's grasp heading about the vertical (gravity stays
  down) and scales distances by an arm-length ratio. The point that is mapped is the one in front of the
  human's palm where the cup's axis is during a side grasp; it corresponds to the G1's grasp site.
- The robot's own constraints are then applied:
  - the last part of the approach slides in along the fingers;
  - the path stays high enough over the other cups (also while carrying a cup);
  - the open hand first lifts off the placed cup;
  - the motion is re-timed only where it is faster than the arm can follow.
- A closed-loop tracking expert follows the path. Its grasp, the fine alignment over the target, lowering,
  and release are triggered by the state, as the Panda work showed is necessary for clean labels.

Tossing the ball. The G1 cannot copy a palm-down pick from the table (its ball sits on a pedestal and is
scooped up palm-up), so the human contributes the throw's style:
- how high and how far from the cup the ball is released, and from which side;
- how long the ball flies (the arc);
- how long the wind-up is.
The values are clipped to the region where the G1's throw works (TOSS_BOUNDS). The release velocity is
ballistic for the robot's own release point, and the aim is calibrated per style in simulation
(`calibrate_aim`): the human's measured velocity is too noisy and too human-specific to aim with.
"""

from dataclasses import dataclass, field

import numpy as np

from robot_arm.humanoid.config import G1_CFG, G1TaskConfig
from robot_arm.humanoid.envs import OBS_BALL, OBS_SITE, OBS_STAGE, OBS_TARGET, G1PyramidEnv, obs_cup
from robot_arm.humanoid.objects import DOWN, ORIENTATIONS, UP, cup_grasp_point
from robot_arm.humanoid.scripted import OPEN_STEPS, PlaceExpert, TossExpert, hand_frame, to_action
from robot_arm.synthetic import min_jerk
from robot_arm.video.skills import GRAVITY, SkillDemo

ARM_SCALE = 0.8  # G1 arm / human arm
PALM_TO_AXIS = 0.05  # human palm centre -> the held cup's axis, along the palm normal
# Robot hand axes (x along the fingers, y out of the palm, z thumb side) in human palm coordinates
# (x along the fingers, y away from the thumb, z out of the palm).
HUMAN_TO_ROBOT_HAND = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def heading(R: np.ndarray) -> float:
    """Horizontal heading of a hand frame's finger axis."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _mj(n: int) -> np.ndarray:
    """n samples of a min-jerk ramp from 0 to 1 (n >= 2)."""
    return min_jerk(np.zeros(1), np.ones(1), n)[:, 0]


def _index(t: np.ndarray, time: float) -> int:
    return int(np.clip(np.searchsorted(t, time), 0, len(t) - 1))


# ---------------------------------------------------------------------------------------------- placing


@dataclass
class PlacePlan:
    waypoints: np.ndarray  # (N, 3) grasp-site targets, one per control step
    grasp_idx: int  # arrive at the cup
    over_idx: int  # the carried cup is about over its target: align and lower from here
    release_idx: int
    yaw: float  # robot hand heading
    human_times: np.ndarray = field(default_factory=lambda: np.zeros(0))  # human time of each waypoint


@dataclass
class Obstacle:
    xy: np.ndarray
    top: float  # height of its highest point


def place_obstacles(obs: np.ndarray, stage: int, cfg: G1TaskConfig = G1_CFG) -> list[Obstacle]:
    """Every cup but the one being placed (its own footprint is handled by the approach)."""
    out = []
    for k in range(3):
        if k == stage:
            continue
        pos, axis = obs_cup(obs, k)
        top = pos[2] if axis[2] < 0 else pos[2] + cfg.cup.height
        out.append(Obstacle(pos[:2].copy(), float(top)))
    return out


def plan_place(demo: SkillDemo, site0: np.ndarray, grasp_r: np.ndarray, place_r: np.ndarray,
               obstacles: list[Obstacle], held_cup: Obstacle, hang: float, cfg: G1TaskConfig = G1_CFG,
               yaw: float = 0.0, scale: float = ARM_SCALE, pre_dist: float = 0.16, max_speed: float = 0.25,
               clearance: float = 0.07) -> PlacePlan:
    """Grasp-site path for the robot from a human place demo.

    grasp_r / place_r: robot grasp-site position at the grasp and when the cup stands on its target. hang: how
    far the held cup reaches below the grasp site. held_cup: the cup to pick (an obstacle until the grasp)."""
    t = demo.t
    g_h = demo.palm + PALM_TO_AXIS * demo.palm_rot[:, :, 2]
    i_g, i_r = _index(t, demo.t_grasp), _index(t, demo.t_release)
    i_r = max(i_r, i_g + 2)
    rob_x = np.array([np.cos(yaw), np.sin(yaw), 0.0])

    def anchor_map(i, target):
        Q = rot_z(yaw - heading(demo.palm_rot[i] @ HUMAN_TO_ROBOT_HAND))
        return target + scale * (g_h - g_h[i]) @ Q.T

    P_g, P_p = anchor_map(i_g, grasp_r), anchor_map(i_r, place_r)
    w = np.clip((t - (t[i_g] + 0.25 * (t[i_r] - t[i_g]))) / (0.5 * (t[i_r] - t[i_g])), 0, 1)
    w = 10 * w ** 3 - 15 * w ** 4 + 6 * w ** 5
    path = (1 - w)[:, None] * P_g + w[:, None] * P_p

    # Slide in along the fingers for the last `pre_dist`: bend the earlier path so it arrives behind the cup.
    pre = grasp_r - pre_dist * rob_x
    dist = np.linalg.norm(path[:i_g] - grasp_r, axis=1)
    far = np.nonzero(dist >= pre_dist)[0]
    i_pre = int(far[-1]) if len(far) else 0
    i_b = max(0, i_pre - int(0.6 / np.median(np.diff(t))))
    shift = pre - path[i_pre]
    if i_pre > i_b:
        path[i_b:i_pre + 1] += shift * _mj(i_pre - i_b + 1)[:, None]
    else:
        path[:i_pre + 1] += shift
    path[i_pre:i_g + 1] = pre + (grasp_r - pre) * _mj(i_g - i_pre + 1)[:, None]

    # After the release the open hand lifts off the cup before moving sideways; it ends at least 8 cm higher.
    rel = path[i_r:] - path[i_r]
    lift = np.maximum(rel[:, 2], 0) + 0.08 * _mj(len(rel)) * (1 - np.clip(rel[:, 2] / 0.08, 0, 1))
    side = rel[:, :2] * np.clip(lift / 0.06, 0, 1)[:, None]
    path[i_r:] = path[i_r] + np.column_stack([side, lift])

    # Begin at the robot's hand: rise first if it is low (e.g. right after letting go of a cup), then fade the
    # offset between the robot's and the mapped start out before the approach.
    T = cfg.table_height
    start = np.array([site0[0], site0[1], max(site0[2], T + 0.22)])
    i_fade = max(2, i_b)
    path[:i_fade] += (start - path[0]) * (1 - _mj(i_fade))[:, None]

    # Clear the other cups (and the cup to pick, before the approach); while carrying, the held cup too.
    z_req = np.full(len(path), -np.inf)
    for i in range(len(path)):
        if i_pre <= i <= i_g or i >= i_r:
            continue
        holding = i_g < i < i_r
        below = hang if holding else 0.0
        reach = 0.10 if holding else 0.12  # held: the cups' footprints plus the fingers around the held one
        for ob in obstacles + ([] if holding else [held_cup]):
            if np.linalg.norm(path[i, :2] - ob.xy) < reach:
                z_req[i] = max(z_req[i], ob.top + below + (0.03 if holding else clearance))
    if np.isfinite(z_req).any():
        k = max(1, int(0.25 / np.median(np.diff(t))))
        padded = np.pad(z_req, k, constant_values=-np.inf)
        z_req = np.array([padded[i:i + 2 * k + 1].max() for i in range(len(path))])  # widen the zones
        z_req[i_pre:i_g + 1] = -np.inf  # but never lift the slide onto the cup
        path[:, 2] = np.maximum(path[:, 2], z_req)
    path[:i_pre, 2] = np.maximum(path[:i_pre, 2], pre[2])  # never dive below the approach height early on
    path[:, 2] = np.maximum(path[:, 2], T + 0.03)

    # Re-time: the human's timing where it is slow enough, stretched where the arm cannot follow.
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    t_robot = np.concatenate([[0.0], np.cumsum(np.maximum(np.diff(t), seg / max_speed))])
    steps = np.arange(0.0, t_robot[-1] + 1e-9, cfg.control_dt)
    wp = np.column_stack([np.interp(steps, t_robot, path[:, k]) for k in range(3)])
    human_t = np.interp(steps, t_robot, t)
    grasp_idx = int(np.searchsorted(steps, t_robot[i_g]))
    release_idx = int(np.searchsorted(steps, t_robot[i_r]))
    horiz = np.linalg.norm(wp[grasp_idx:release_idx + 1, :2] - place_r[:2], axis=1)
    near = np.nonzero(horiz < 0.03)[0]
    over_idx = grasp_idx + (int(near[0]) if len(near) else release_idx - grasp_idx)
    return PlacePlan(wp, min(grasp_idx, len(wp) - 1), min(over_idx, len(wp) - 1), min(release_idx, len(wp) - 1),
                     yaw, human_t)


def place_plan_for_obs(demo: SkillDemo, obs: np.ndarray, cfg: G1TaskConfig = G1_CFG, **kwargs) -> PlacePlan:
    """Plan for the stage shown in an observation (stage 0-2), from the robot's current state."""
    stage = int(np.argmax(obs[OBS_STAGE]))
    pos, axis = obs_cup(obs, stage)
    frac = cfg.top_grasp_height_frac if stage == 2 else cfg.grasp_height_frac
    grasp_r = cup_grasp_point(pos, axis, frac, cfg)
    final_axis = np.array([0, 0, 1.0]) if ORIENTATIONS[stage] is UP else np.array([0, 0, -1.0])
    place_r = cup_grasp_point(obs[OBS_TARGET], final_axis, frac, cfg) + [0, 0, 0.003]
    top = pos[2] if axis[2] < 0 else pos[2] + cfg.cup.height
    hang = frac * cfg.cup.height
    return plan_place(demo, obs[OBS_SITE], grasp_r, place_r, place_obstacles(obs, stage, cfg),
                      Obstacle(pos[:2].copy(), float(top)), hang, cfg, **kwargs)


class PlaceTracker:
    """Follows a PlacePlan in closed loop.

    The plan's clock only advances while the hand keeps up. Contact events are triggered by the state:
    - the final approach and the grasp servo onto the cup's actual grasp point;
    - during the carry the held cup (not the hand) is servoed along the plan, so slip in the grasp is corrected;
    - once the cup is about over its target, it is aligned and lowered straight down;
    - the hand opens when the cup stands on the target, then follows the plan's retreat.
    """

    FINE_PHASES = ("approach", "close", "lower", "open")

    def __init__(self, plan: PlacePlan, cfg: G1TaskConfig = G1_CFG, lag_tol: float = 0.025):
        self.plan, self.cfg, self.lag_tol = plan, cfg, lag_tol
        self.R = hand_frame(plan.yaw)
        self.reset()

    def reset(self):
        self.phase, self.idx, self.timer = "follow", 0, 0
        self.cup0 = None
        self.offset0 = None
        self.release_site = None

    @property
    def fine(self) -> bool:
        """In a phase where exploration noise would only spoil the contact."""
        return self.phase in self.FINE_PHASES

    def _advance(self, site, goal, end):
        if np.linalg.norm(goal - site) < self.lag_tol:
            self.idx = min(self.idx + 1, end)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        cfg, plan = self.cfg, self.plan
        stage = int(np.argmax(obs[OBS_STAGE]))
        site = obs[OBS_SITE]
        cup_pos, cup_axis = obs_cup(obs, stage)
        if self.cup0 is None:
            self.cup0 = cup_pos.copy()
        frac = cfg.top_grasp_height_frac if stage == 2 else cfg.grasp_height_frac
        grasp = cup_grasp_point(cup_pos, cup_axis, frac, cfg)
        target = obs[OBS_TARGET]
        self.timer += 1
        grip = -1.0
        near = lambda p, tol: np.linalg.norm(p - site) < tol  # noqa: E731

        if self.phase == "follow":
            ramp = self.idx / max(plan.grasp_idx, 1)
            goal = plan.waypoints[self.idx] + ramp * (cup_pos - self.cup0)  # follow the cup if it was nudged
            self._advance(site, goal, plan.grasp_idx)
            if self.idx >= plan.grasp_idx:
                self.phase, self.timer = "approach", 0
        if self.phase == "approach":
            goal = grasp
            if near(goal, 0.006) or (near(goal, 0.012) and self.timer > 25):
                self.phase, self.timer = "close", 0
        if self.phase == "close":
            goal, grip = grasp, 1.0
            if self.timer > 8:
                self.phase, self.idx = "carry", plan.grasp_idx
                self.offset0 = site - cup_pos
                self.anchor = site - plan.waypoints[plan.grasp_idx]  # where the grasp really happened
        cup_offset = site - cup_pos
        if self.phase == "carry":
            # Start from where the grasp really happened (fading towards the plan's own target) and correct slip
            # in the grasp.
            fade = 1 - (self.idx - plan.grasp_idx) / max(plan.over_idx - plan.grasp_idx, 1)
            goal, grip = plan.waypoints[self.idx] + fade * self.anchor + (cup_offset - self.offset0), 1.0
            self._advance(site, goal, plan.over_idx)
            if self.idx >= plan.over_idx:
                self.phase = "align"
                self.align_z = max(plan.waypoints[plan.over_idx, 2], site[2])
        if self.phase == "align":
            goal, grip = np.array([*(target[:2] + cup_offset[:2]), self.align_z]), 1.0
            if np.linalg.norm(cup_pos[:2] - target[:2]) < 0.006:
                self.phase = "lower"
        if self.phase == "lower":
            goal, grip = target + cup_offset + [0, 0, 0.002], 1.0
            goal[2] = max(goal[2], site[2] - cfg.max_step)
            if cup_pos[2] - target[2] < 0.004:
                self.phase, self.timer = "open", 0
        if self.phase == "open":
            goal = site
            if self.timer > OPEN_STEPS:  # fully open (see release_rate)
                self.phase, self.idx, self.release_site = "retreat", plan.release_idx, site.copy()
        if self.phase == "retreat":
            goal = self.release_site + plan.waypoints[self.idx] - plan.waypoints[plan.release_idx]
            self._advance(site, goal, len(plan.waypoints) - 1)
        return to_action(obs, goal, self.R, grip, cfg.max_step, cfg)


# ---------------------------------------------------------------------------------------------- tossing

# Where the G1's throw works (from simulated sweeps with the aim calibrated; 53-91% success inside the box):
# human styles are clipped to it. The throw is a precision skill: the ball has 1 cm of clearance in the mouth.
# Releases further right (bearing -2.28 and beyond) need a wind-up the arm can't reach, flatter arcs and
# shorter wind-ups are much less accurate.
TOSS_BOUNDS = {"release_dist": (0.28, 0.32), "release_height": (0.14, 0.18), "release_bearing": (-2.42, -2.37),
               "flight_time": (0.31, 0.34), "windup": (0.16, 0.19)}


def throw_style(demo: SkillDemo, scale: float = ARM_SCALE) -> dict:
    """The human throw's style in the robot's terms (before clipping to TOSS_BOUNDS).

    - Release distance and height (the ball above the mouth) are properties of the throw, so they stay as they
      are. The wind-up is an arm motion, so it is scaled.
    - The bearing is measured from the direction towards the thrower. The person faces -y in the video's table
      frame and the robot faces +x, so a release on the person's right maps to one on the robot's right."""
    mouth, rel = demo.obj_end, demo.release_pos
    d = rel[:2] - mouth[:2]
    bearing_h = np.arctan2(d[1], d[0])
    v_h = np.linalg.norm(demo.release_vel[:2])
    flight = np.linalg.norm(d) / max(v_h, 1e-3)
    i_r = _index(demo.t, demo.t_release)
    # Wind-up: from where the hand was slowest during the 0.6 s before the release to the release.
    fps = 1 / np.median(np.diff(demo.t))
    lo = max(0, i_r - int(0.6 * fps))
    speed = np.linalg.norm(np.gradient(demo.palm, axis=0), axis=1) * fps
    i_w = lo + int(np.argmin(speed[lo:i_r])) if i_r > lo else i_r
    windup = np.linalg.norm(demo.palm[i_r] - demo.palm[i_w])
    return {"release_dist": float(np.linalg.norm(d)), "release_height": float(rel[2] - mouth[2]),
            "release_bearing": float(np.angle(np.exp(1j * (bearing_h + np.pi / 2)))), "flight_time": float(flight),
            "windup": scale * float(windup)}


def toss_params(style: dict, bounds: dict | None = None) -> dict:
    """TossExpert parameters: the style clipped to where the G1's throw works."""
    bounds = TOSS_BOUNDS if bounds is None else bounds
    out = {}
    for k, (lo, hi) in bounds.items():
        v = style[k]
        if k == "release_bearing":
            mid = (lo + hi) / 2
            v = mid + np.angle(np.exp(1j * (v - mid)))  # the equivalent angle nearest the box
        out[k] = float(np.clip(v, lo, hi))
    return out


def _toss_episode(args):
    """One toss episode with the given expert parameters: (success, miss vector at the mouth plane or None,
    outcome)."""
    params, seed = args
    env = G1PyramidEnv(task="toss")
    expert = TossExpert(**params)
    obs, _ = env.reset(seed=seed)
    mouth = obs[OBS_TARGET].copy()
    prev, miss = obs[OBS_BALL].copy(), None
    for _ in range(env.max_steps):
        obs, _, terminated, truncated, info = env.step(expert(obs))
        ball = obs[OBS_BALL]
        if miss is None and env.release_distance is not None and prev[2] >= mouth[2] > ball[2]:
            f = (prev[2] - mouth[2]) / (prev[2] - ball[2])
            miss = (prev + f * (ball - prev) - mouth)[:2]
        prev = ball.copy()
        if terminated or truncated:
            break
    outcome = "ok" if info["is_success"] else info["fail_reason"] or "timeout"
    return bool(info["is_success"]), miss, outcome


def calibrate_aim(params: dict, rounds: int = 3, episodes: int = 16, seed: int = 50_000, pool=None) -> dict:
    """Choose the aim offset of a throw style from simulated throws.

    With aim a the ball comes down at m = a + e, where e is the style's aiming error. The ball only goes in for
    m in a small region around the mouth. The region is not centred on the mouth: a long ball can bounce in off
    the far wall, while a short one hits the near rim. So the throws aim e's mean at the centre of the throws
    that went in, using every round's throws. With fewer than 3 hits so far they aim at the mouth centre.

    Returns the parameters with the new `aim_offset`, plus `aim_history`: the aim and the success rate of each
    round. A style whose balls never come down near the cup keeps its aim; its release point is probably
    unreachable."""
    aim = np.array(params.get("aim_offset", (0.0, 0.04, 0.0)), float)
    history, errors, hits = [], [], []
    for r in range(rounds):
        p = {**params, "aim_offset": tuple(aim)}
        jobs = [(p, seed + r * episodes + i) for i in range(episodes)]
        results = pool.map(_toss_episode, jobs) if pool is not None else list(map(_toss_episode, jobs))
        history.append((tuple(round(float(a), 3) for a in aim), float(np.mean([s for s, _, _ in results]))))
        for success, miss, _ in results:
            if miss is not None and np.linalg.norm(miss) < 0.15:
                errors.append(miss - aim[:2])
                if success:
                    hits.append(miss)
        if len(errors) < episodes // 2:
            break
        centre = np.mean(hits, axis=0) if len(hits) >= 3 else np.zeros(2)
        aim[:2] += 0.8 * (centre - np.mean(errors, axis=0) - aim[:2])
    return {**params, "aim_offset": tuple(float(a) for a in aim), "aim_history": history}


# ---------------------------------------------------------------------------------------------- rollouts


def expert_params(style: dict | None) -> dict | None:
    """TossExpert keyword arguments from a saved throw style (drops the bookkeeping entries)."""
    return None if style is None else {k: v for k, v in style.items() if k not in ("aim_history", "human")}


class G1Retargeter:
    """The retargeted experts for a whole episode: one human demo per place stage and a throw style."""

    def __init__(self, place_demos: dict[int, SkillDemo], toss: dict | None, cfg: G1TaskConfig = G1_CFG):
        self.place_demos, self.cfg = place_demos, cfg
        toss = {k: v for k, v in (toss or {}).items() if k != "aim_history"}
        self.toss = TossExpert(cfg, **toss) if toss else TossExpert(cfg)
        self.reset()

    def reset(self):
        self.stage, self.tracker = None, None
        self.toss.reset()

    @property
    def fine(self) -> bool:
        if self.stage == 3:
            return self.toss.phase in ("slide", "scoop", "settle", "swing", "release")
        return self.tracker is not None and self.tracker.fine

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        stage = int(np.argmax(obs[OBS_STAGE]))
        if stage != self.stage:
            self.stage = stage
            if stage < 3:
                demo = self.place_demos.get(stage)
                self.tracker = PlaceTracker(place_plan_for_obs(demo, obs, self.cfg), self.cfg) if demo else \
                    PlaceExpert(self.cfg)
        if stage == 3:
            return self.toss(obs)
        return self.tracker(obs)


def rollout(env: G1PyramidEnv, policy: G1Retargeter, rng: np.random.Generator, noise: float = 0.0,
            seed: int | None = None, options: dict | None = None, render: bool = False):
    """Run the retargeted experts. DART: noise on the executed motion (not the labels, not the grip, and not
    during contact phases). Returns observations, labels, the stage of every step, per-stage success, frames."""
    obs, _ = env.reset(seed=seed, options=options)
    policy.reset()
    observations, labels, stages, frames = [], [], [], []
    done_stages = set()
    info = {"stage": env.stage, "is_success": False}
    for _ in range(env.max_steps):
        stage = env.stage
        action = policy(obs)
        executed = action.copy()
        if noise > 0 and not policy.fine:
            # The same noise in metres for every stage (an action unit is 5x larger while tossing).
            scale = env.cfg.max_step / env.max_step()
            executed[:3] = np.clip(executed[:3] + rng.normal(0, noise * scale, 3), -1, 1)
            executed[3:6] = np.clip(executed[3:6] + rng.normal(0, 0.5 * noise, 3), -1, 1)
        observations.append(obs)
        labels.append(action)
        stages.append(stage)
        obs, _, terminated, truncated, info = env.step(executed)
        if env.stage != stage or info["is_success"]:
            done_stages.add(stage)
        if render:
            frames.append(env.render())
        if terminated or truncated:
            break
    return (np.array(observations), np.array(labels), np.array(stages), done_stages, info, frames)


class G1ReplayPolicy:
    """Baseline without learning: the retargeted experts, taking the human demos and throw styles in turn."""

    def __init__(self, demos: dict[str, list[SkillDemo]], styles: list[dict], cfg: G1TaskConfig = G1_CFG):
        self.demos, self.styles, self.cfg = demos, styles, cfg
        self.k, self.current = 0, None

    def start_at(self, k: int):
        """Use the demos for episode number k next (evaluation runs each episode in a fresh process)."""
        self.k = k

    def reset(self):
        base, top, k = self.demos["place_base"], self.demos["place_top"], self.k
        self.k += 1
        place = {0: base[k % len(base)], 1: base[(k + 1) % len(base)], 2: top[k % len(top)]}
        style = self.styles[k % len(self.styles)] if self.styles else None
        self.current = G1Retargeter(place, expert_params(style), self.cfg)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.current is None:
            self.reset()
        return self.current(obs)
