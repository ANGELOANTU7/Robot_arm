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
- The hand turns about the vertical the way the human's did while reaching (relative to the grasp heading).
- The top cup is turned over the robot's way once it is picked up: carried mouth-down to above the pyramid,
  rolled half a turn there and put down (see scripted.PlaceExpert). The G1's arm cannot roll a held cup
  everywhere along a human path, and where the cup is let go of decides whether the pyramid stands.

Throwing the cap. The pinch itself is the robot's own (the Dex3 has three fingers, and where they meet the cap
is a matter of millimetres), so the human contributes the throw's style:
- how far from the mouth and how high above it the cap is released, and how far off the base cups' line;
- how long the cap flies (the arc);
- how long the wind-up is.
The values are clipped to the region where the G1's throw works (TOSS_BOUNDS). The release velocity is
ballistic for the robot's own release point, and the aim is calibrated per style in simulation
(`calibrate_aim`): the human's measured velocity is too noisy and too human-specific to aim with.
"""

from dataclasses import dataclass, field

import numpy as np

from robot_arm.humanoid import trylog
from robot_arm.humanoid.config import G1_CFG, G1TaskConfig
from robot_arm.humanoid.hands import get_hand
from robot_arm.humanoid.envs import OBS_CAP, OBS_SITE, OBS_STAGE, OBS_TARGET, TOSS, G1PyramidEnv, obs_cup
from robot_arm.humanoid.objects import (ORIENTATIONS, UP, cup_grasp_point, cup_lowest_z, cup_table_point,
                                        reach_heading)
from robot_arm.humanoid.scripted import (OPEN_STEPS, PlaceExpert, TossExpert, grasp_frame, hand_frame, roll_x,
                                         taper_angle, to_action)
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
    yaw: float  # robot hand heading at the grasp
    human_times: np.ndarray = field(default_factory=lambda: np.zeros(0))  # human time of each waypoint
    place_yaw: float = 0.0  # robot hand heading when putting the cup down
    turn: np.ndarray = field(default_factory=lambda: np.zeros(0))  # per waypoint: heading change while reaching


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
               yaw: float = 0.0, place_yaw: float | None = None, scale: float = ARM_SCALE, pre_dist: float = 0.16,
               max_speed: float = 0.25, clearance: float = 0.07, max_turn: float = 0.6,
               approach: np.ndarray | None = None) -> PlacePlan:
    """Grasp-site path for the robot from a human place demo.

    grasp_r / place_r: robot grasp-site position at the grasp and when the cup stands on its target. hang: how
    far the held cup reaches below the grasp site. held_cup: the cup to pick (an obstacle until the grasp).
    yaw / place_yaw: the robot hand's heading at the grasp and when putting the cup down. approach: the direction
    of the hand's last `pre_dist` onto the cup (default: along its heading, sliding in along the fingers)."""
    t = demo.t
    place_yaw = yaw if place_yaw is None else place_yaw
    g_h = demo.palm + PALM_TO_AXIS * demo.palm_rot[:, :, 2]
    i_g, i_r = _index(t, demo.t_grasp), _index(t, demo.t_release)
    i_r = max(i_r, i_g + 2)
    rob_x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    human_heading = np.unwrap([heading(R @ HUMAN_TO_ROBOT_HAND) for R in demo.palm_rot])

    def anchor_map(i, target, robot_yaw):
        Q = rot_z(robot_yaw - human_heading[i])
        return target + scale * (g_h - g_h[i]) @ Q.T

    P_g, P_p = anchor_map(i_g, grasp_r, yaw), anchor_map(i_r, place_r, place_yaw)
    w = np.clip((t - (t[i_g] + 0.25 * (t[i_r] - t[i_g]))) / (0.5 * (t[i_r] - t[i_g])), 0, 1)
    w = 10 * w ** 3 - 15 * w ** 4 + 6 * w ** 5
    path = (1 - w)[:, None] * P_g + w[:, None] * P_p

    # Slide in along the fingers (or the hand's own approach) for the last `pre_dist`: bend the earlier path so it
    # arrives there.
    pre = grasp_r - pre_dist * (rob_x if approach is None else np.asarray(approach, float))
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
    if i_pre <= 1:
        # The clip starts at the cup (cut out of a longer one) and shows no reach. A straight line from the hand to
        # the approach point would sweep the fingers over the cup's rim: go above that point first, clear of the
        # cup, then down.
        over = np.array([pre[0], pre[1], max(pre[2], held_cup.top + clearance, start[2])])
        path[:i_pre + 1] = pre
        path, t = np.vstack([start, over, path]), np.concatenate([[t[0], t[0]], t])
        human_heading = np.concatenate([human_heading[:1], human_heading[:1], human_heading])
        i_b, i_pre, i_g, i_r = i_b + 2, i_pre + 2, i_g + 2, i_r + 2
    else:
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
        # nor the release point and the lift-off after it (a cup set down next to another would otherwise be let go
        # of high above it, and the retreat, relative to that point, would head down onto the table)
        z_req[i_r:] = -np.inf
        path[:, 2] = np.maximum(path[:, 2], z_req)
    path[:i_pre, 2] = np.maximum(path[:i_pre, 2], pre[2])  # never dive below the approach height early on
    path[:, 2] = np.maximum(path[:, 2], T + 0.03)
    # The hand's turn about the vertical while reaching, relative to its heading at the grasp; none from the
    # slide onto the cup on.
    turn = np.clip(human_heading - human_heading[i_g], -max_turn, max_turn)
    turn[i_pre:] = 0.0
    if i_pre > i_b:
        turn[i_b:i_pre + 1] *= 1 - _mj(i_pre - i_b + 1)

    # Re-time: the human's timing where it is slow enough, stretched where the arm cannot follow.
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    t_robot = np.concatenate([[0.0], np.cumsum(np.maximum(np.diff(t), seg / max_speed))])
    steps = np.arange(0.0, t_robot[-1] + 1e-9, cfg.control_dt)
    wp = np.column_stack([np.interp(steps, t_robot, path[:, k]) for k in range(3)])
    human_t = np.interp(steps, t_robot, t)
    turn = np.interp(steps, t_robot, turn)
    grasp_idx = int(np.searchsorted(steps, t_robot[i_g]))
    release_idx = int(np.searchsorted(steps, t_robot[i_r]))
    horiz = np.linalg.norm(wp[grasp_idx:release_idx + 1, :2] - place_r[:2], axis=1)
    near = np.nonzero(horiz < 0.03)[0]
    over_idx = grasp_idx + (int(near[0]) if len(near) else release_idx - grasp_idx)
    return PlacePlan(wp, min(grasp_idx, len(wp) - 1), min(over_idx, len(wp) - 1), min(release_idx, len(wp) - 1),
                     yaw, human_t, place_yaw, turn)


def place_plan_for_obs(demo: SkillDemo, obs: np.ndarray, cfg: G1TaskConfig = G1_CFG, **kwargs) -> PlacePlan:
    """Plan for the stage shown in an observation (stage 0-2), from the robot's current state."""
    stage = int(np.argmax(obs[OBS_STAGE]))
    pos, axis = obs_cup(obs, stage)
    frac = cfg.top_grasp_height_frac if stage == 2 else cfg.grasp_height_frac
    grasp_r = cup_grasp_point(pos, axis, frac, cfg)
    final_axis = np.array([0, 0, 1.0]) if ORIENTATIONS[stage] is UP else np.array([0, 0, -1.0])
    place_r = cup_grasp_point(obs[OBS_TARGET], final_axis, frac, cfg) + [0, 0, 0.003]
    if stage == 2 and axis[2] < 0:  # held where it will be held once turned over (see PlaceExpert)
        grasp_r = cup_grasp_point(pos, axis, 1 - frac, cfg)
    top = pos[2] if axis[2] < 0 else pos[2] + cfg.cup.height
    hang = frac * cfg.cup.height
    hand = get_hand(cfg.hand)
    kwargs.setdefault("yaw", reach_heading(pos, cfg))
    kwargs.setdefault("place_yaw", hand.expert.get("place_yaw", 0.0))
    kwargs.setdefault("pre_dist", hand.expert.get("pre_dist", 0.16))
    kwargs.setdefault("clearance", hand.clearance)
    kwargs.setdefault("approach", hand_frame(kwargs["yaw"]) @ np.asarray(hand.approach))
    return plan_place(demo, obs[OBS_SITE], grasp_r, place_r, place_obstacles(obs, stage, cfg),
                      Obstacle(pos[:2].copy(), float(top)), hang, cfg, **kwargs)


class PlaceTracker:
    """Follows a PlacePlan in closed loop.

    The plan's clock only advances while the hand keeps up. Contact events are triggered by the state:
    - the final approach and the grasp servo onto the cup's actual grasp point;
    - during the carry the held cup (not the hand) is servoed along the plan, so slip in the grasp is corrected;
    - once the cup is about over its target, it is aligned and lowered straight down;
    - the hand opens when the cup stands on the target (or stops sinking), then follows the plan's retreat.
    The top cup, once held, is turned over and put down by the scripted expert (see the module docstring).
    """

    FINE_PHASES = ("approach", "close", "lower", "open", "back")

    def __init__(self, plan: PlacePlan, cfg: G1TaskConfig = G1_CFG, lag_tol: float = 0.025):
        self.plan, self.cfg, self.lag_tol = plan, cfg, lag_tol
        self.reset()

    def reset(self):
        self.phase, self.idx, self.timer = "follow", 0, 0
        self.cup0 = None
        self.offset0 = None
        self.release_site = None
        self.expert, self.z_hist = None, []

    @property
    def fine(self) -> bool:
        """In a phase where exploration noise would only spoil the contact."""
        if self.expert is not None:
            return self.expert.phase in ("flip", "lower", "open", "back")
        return self.phase in self.FINE_PHASES

    def _advance(self, site, goal, end, ahead: int = 20):
        """Move the plan's clock on while the hand keeps up: to the closest of the next `ahead` waypoints the hand
        is at, at least one. A hand that got ahead of the plan (pushed by noise, or a learner being labelled in
        DAgger) is then not sent back."""
        shift = goal - self.plan.waypoints[self.idx]
        dist = np.linalg.norm(self.plan.waypoints[self.idx:min(self.idx + ahead, end) + 1] + shift - site, axis=1)
        j = int(np.argmin(dist))
        if dist[j] < self.lag_tol:
            self.idx = min(self.idx + max(j, 1), end)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        cfg, plan = self.cfg, self.plan
        stage = int(np.argmax(obs[OBS_STAGE]))
        if self.expert is not None:
            return self.expert(obs)
        site = obs[OBS_SITE]
        cup_pos, cup_axis = obs_cup(obs, stage)
        if self.cup0 is None:
            self.cup0 = cup_pos.copy()
            self.flip = ORIENTATIONS[stage] is UP and cup_axis[2] < 0
        frac = cfg.top_grasp_height_frac if stage == 2 else cfg.grasp_height_frac
        grasp = cup_grasp_point(cup_pos, cup_axis, 1 - frac if self.flip else frac, cfg)
        target = obs[OBS_TARGET]
        if self.flip:
            R_grasp = hand_frame(plan.yaw) @ roll_x(get_hand(cfg.hand).expert.get("flip_taper_sign", -1.0)
                                                    * taper_angle(cfg))
        else:
            R_grasp = grasp_frame(plan.yaw, ORIENTATIONS[stage] is UP, cfg)
        R = R_grasp
        R_place = grasp_frame(plan.place_yaw, ORIENTATIONS[stage] is UP, cfg)
        self.timer += 1
        grip = -1.0
        near = lambda p, tol: np.linalg.norm(p - site) < tol  # noqa: E731

        if self.phase == "follow":
            ramp = self.idx / max(plan.grasp_idx, 1)
            goal = plan.waypoints[self.idx] + ramp * (cup_pos - self.cup0)  # follow the cup if it was nudged
            if len(plan.turn):
                R = rot_z(plan.turn[min(self.idx, len(plan.turn) - 1)]) @ R_grasp  # the human's turn while reaching
            before = self.idx
            self._advance(site, goal, plan.grasp_idx)
            self.stuck = 0 if self.idx > before else getattr(self, "stuck", 0) + 1
            if self.stuck > 8:  # a waypoint the arm can't reach: go on to the next
                self.idx, self.stuck = min(self.idx + 1, plan.grasp_idx), 0
            if self.idx >= plan.grasp_idx:
                self.phase, self.timer = "approach", 0
        if self.phase == "approach":
            goal = grasp
            if near(goal, 0.006) or (near(goal, 0.012) and self.timer > 25):
                self.phase, self.timer = "close", 0
        if self.phase == "close":
            goal, grip = grasp, 1.0
            if self.timer > 8:
                if self.flip:  # held: the robot's own way of turning it over and putting it down
                    self.expert = PlaceExpert(cfg, yaw=plan.yaw, top_place_yaw=plan.place_yaw)
                    self.expert.stage, self.expert.phase = stage, "lift"
                    return self.expert(obs)
                self.phase, self.idx = "carry", plan.grasp_idx
                self.offset0 = site - cup_pos
                self.anchor = site - plan.waypoints[plan.grasp_idx]  # where the grasp really happened
        final_axis = np.array([0, 0, 1.0]) if ORIENTATIONS[stage] is UP else np.array([0, 0, -1.0])
        cup_tp, target_tp = cup_table_point(cup_pos, cup_axis, cfg), cup_table_point(target, final_axis, cfg)
        cup_offset = site - cup_tp
        if self.phase in ("carry", "align", "lower", "open", "retreat"):
            R = R_place  # turning to the placing heading while carrying
        if self.phase == "carry":
            # Start from where the grasp really happened (fading towards the plan's own target) and correct slip
            # in the grasp.
            fade = 1 - (self.idx - plan.grasp_idx) / max(plan.over_idx - plan.grasp_idx, 1)
            goal, grip = plan.waypoints[self.idx] + fade * self.anchor + (cup_offset - self.offset0), 1.0
            self._advance(site, goal, plan.over_idx)
            if self.idx >= plan.over_idx:
                self.phase, self.timer = "align", 0
                self.align_z = max(plan.waypoints[plan.over_idx, 2], site[2])
        if self.phase == "align":
            goal, grip = np.array([*(target_tp[:2] + cup_offset[:2]), self.align_z]), 1.0
            off = np.linalg.norm(cup_tp[:2] - target_tp[:2])
            if off < 0.006 or (self.timer > 20 and off < 0.012):
                self.phase = "lower"
        if self.phase == "lower":
            goal, grip = target_tp + cup_offset + [0, 0, 0.002], 1.0
            goal[2] = max(goal[2], site[2] - cfg.max_step)
            self.z_hist.append(cup_pos[2])
            stalled = len(self.z_hist) > 3 and self.z_hist[-4] - self.z_hist[-1] < 0.002
            above = cup_lowest_z(cup_pos, cup_axis, cfg) - target_tp[2]
            if above < 0.003 or (above < 0.012 and stalled):  # touched down
                self.phase, self.timer = "open", 0
        if self.phase == "open":
            goal = site
            if self.timer > OPEN_STEPS:  # fully open (see release_rate)
                self.phase, self.release_site = "retreat", site.copy()
                self.idx = plan.release_idx
        if self.phase == "retreat":
            goal = self.release_site + plan.waypoints[self.idx] - plan.waypoints[plan.release_idx]
            before = self.idx
            self._advance(site, goal, len(plan.waypoints) - 1)
            self.stuck = 0 if self.idx > before else getattr(self, "stuck", 0) + 1
            if self.idx >= len(plan.waypoints) - 1 or self.stuck > 5:
                # The human's retreat is done, or the arm can't follow it here: rise clear of the cups.
                self.phase = "rise"
        if self.phase == "rise":
            goal = np.array([site[0], site[1], cfg.table_height + 0.30])
        return to_action(obs, goal, R, grip, cfg.max_step, cfg)


# ---------------------------------------------------------------------------------------------- tossing

# Where the G1's throw works (from simulated sweeps on random layouts, before aim calibration: 58-96% success at
# the corners of the box, 24 layouts each): human styles are clipped to it. Releases closer than 10 cm let go
# within the 5 cm minimum or bump the pyramid with the fingers; from 20 cm on, the cap mostly misses, and flights
# of 0.38 s are less accurate. Releasing slightly off the base cups' line on the robot's side (positive bearing)
# works better than the other side. Short flights with a long wind-up throw well on a pre-built pyramid but miss
# after the robot has built it (the arm arrives in another configuration), so the flight time stays near 0.3 s.
TOSS_BOUNDS = {"release_dist": (0.10, 0.16), "release_height": (0.06, 0.12), "release_bearing": (0.0, 0.10),
               "flight_time": (0.26, 0.32), "windup": (0.14, 0.18)}


def throw_style(demo: SkillDemo, scale: float = ARM_SCALE) -> dict:
    """The human throw's style in the robot's terms (before clipping to TOSS_BOUNDS).

    - Release distance (horizontal, from the mouth's centre) and height above the mouth are properties of the
      throw, so they stay as they are; the flight time follows from the distance and the horizontal speed. The
      wind-up is an arm motion, so it is scaled.
    - The bearing is the release point's angle from the base cups' line (0: straight along it), on whichever side
      the person threw from; the robot throws from cup B's side. Without the base cups in view it is taken as 0,
      as the recording guide asks for a throw along them."""
    mouth, rel = demo.obj_end, demo.release_pos
    d = rel[:2] - mouth[:2]
    axis = demo.base_axis
    bearing = 0.0
    if axis is not None:
        a = axis[:2] * (1.0 if np.dot(axis[:2], d) >= 0 else -1.0)  # pointing to the release side
        bearing = float(np.arctan2(a[0] * d[1] - a[1] * d[0], np.dot(a, d)))
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
            "release_bearing": bearing, "flight_time": float(flight), "windup": scale * float(windup)}


def toss_params(style: dict, bounds: dict | None = None) -> dict:
    """TossExpert parameters: the style clipped to where the G1's throw works."""
    bounds = TOSS_BOUNDS if bounds is None else bounds
    return {k: float(np.clip(style[k], lo, hi)) for k, (lo, hi) in bounds.items()}


def _toss_episode(args):
    """The first throw of a toss episode with the given expert parameters: (in the cup, miss vector at the mouth
    plane or None, outcome)."""
    params, seed = args
    env = G1PyramidEnv(task="toss", **trylog.render_kwargs())
    expert = TossExpert(**params)
    obs, _ = env.reset(seed=seed)
    tries = trylog.TryRecorder(env, "aimcal_toss", f"aim calibration {np.round(params.get('aim_offset', (0, 0, 0)), 3)}")
    mouth = obs[OBS_TARGET].copy()
    prev, miss = obs[OBS_CAP].copy(), None
    for t in range(env.max_steps):
        obs, _, terminated, truncated, info = env.step(expert(obs))
        tries.step(t)
        cap = obs[OBS_CAP]
        if miss is None and env.release_distance is not None and prev[2] >= mouth[2] > cap[2]:
            f = (prev[2] - mouth[2]) / (prev[2] - cap[2])
            miss = (prev + f * (cap - prev) - mouth)[:2]
        prev = cap.copy()
        if terminated or truncated or env.stage > TOSS or env.misses:  # the first throw decides
            break
    tries.finish(info if env.stage <= TOSS else {**info, "is_success": env.misses == 0}, seed)
    success = env.stage > TOSS and env.misses == 0
    outcome = "ok" if success else info["fail_reason"] or ("missed" if env.misses else "timeout")
    return bool(success), miss, outcome


def calibrate_aim(params: dict, rounds: int = 3, episodes: int = 16, seed: int = 50_000, pool=None) -> dict:
    """Choose the aim offset of a throw style from simulated throws.

    With aim a the cap comes down at m = a + e, where e is the style's aiming error. The cap only goes in for m
    in a small region around the mouth. The region is not centred on the mouth: a long throw can bounce in off
    the far wall, while a short one hits the near rim. So each round aims e's mean at the centre of the throws
    that went in so far (at the mouth centre with fewer than 3 hits). Every round throws on the same layouts,
    and the aim with the best success rate is kept: with so few throws a round's correction can overshoot.

    Returns the parameters with that `aim_offset`, plus `aim_history`: the aim and the success rate of each
    round. A style whose caps never come down near the cup keeps its aim; its release point is probably
    unreachable."""
    aim = np.array(params.get("aim_offset", (0.0, 0.0, 0.0)), float)
    history, errors, hits = [], [], []
    for _ in range(rounds):
        p = {**params, "aim_offset": tuple(aim)}
        jobs = [(p, seed + i) for i in range(episodes)]
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
    best = max(history, key=lambda h: h[1])  # the earliest of equals
    return {**params, "aim_offset": tuple(float(a) for a in best[0]), "aim_history": history}


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
        if self.stage is not None and self.stage >= TOSS:
            return self.toss.phase in ("back", "rise", "down", "pinch", "settle", "swing", "release", "thumbs_up")
        if isinstance(self.tracker, PlaceExpert):
            return self.tracker.phase in ("approach", "close", "flip", "lower", "open", "back")
        return self.tracker is not None and self.tracker.fine

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        stage = int(np.argmax(obs[OBS_STAGE]))
        if stage != self.stage:
            self.stage = stage
            if stage < TOSS:
                demo = self.place_demos.get(stage)
                self.tracker = PlaceTracker(place_plan_for_obs(demo, obs, self.cfg), self.cfg) if demo else \
                    PlaceExpert(self.cfg)
        if stage >= TOSS:
            return self.toss(obs)
        return self.tracker(obs)


def rollout(env: G1PyramidEnv, policy: G1Retargeter, rng: np.random.Generator, noise: float = 0.0,
            seed: int | None = None, options: dict | None = None, render: bool = False, name: str | None = None,
            stop_stage: int | None = None):
    """Run the retargeted experts. DART: noise on the executed motion (not the labels, not the grip, and not
    during contact phases). Returns observations, labels, the stage of every step, per-stage success, frames.
    With `name` and TRY_VIDEOS set, the try is saved as a video (see trylog.py). With `stop_stage`, the episode
    ends (as a success) once it gets there, e.g. TOSS when only the placing is wanted. `render` may also be a list
    of camera names: each step's frames are then those views, taken with its observation."""
    obs, _ = env.reset(seed=seed, options=options)
    policy.reset()
    tries = trylog.TryRecorder(env, name, f"retargeted demo, noise {noise}") if name and trylog.try_dir() else None
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
        if render and render is not True:  # camera names: the views this observation was taken with
            frames.append([env.render(c) for c in render])
        obs, _, terminated, truncated, info = env.step(executed)
        if env.stage != stage or info["is_success"]:
            done_stages.add(stage)
        if render is True:
            frames.append(env.render())
        if tries is not None:
            tries.step(len(labels) - 1)
        if stop_stage is not None and env.stage >= stop_stage:
            info = {**info, "is_success": True}
            break
        if terminated or truncated:
            break
    if tries is not None:
        tries.finish(info, seed)
    return (np.array(observations), np.array(labels), np.array(stages), done_stages, info, frames)


def place_demos_by_cup(demos: dict[str, list[SkillDemo]]):
    """(cup A demos, cup B demos, top cup demos). Base clips showing both base cups give one demo per cup (cup B's
    knows where A stood); single-cup base demos serve for either."""
    base = demos["place_base"]
    b = [d for d in base if "other_x" in d.info]
    a = [d for d in base if "other_x" not in d.info and str(d.info.get("name", "")).endswith("_a")] or \
        [d for d in base if "other_x" not in d.info] or base
    return a, b or base, demos["place_top"]


class G1ReplayPolicy:
    """Baseline without learning: the retargeted experts, taking the human demos and throw styles in turn."""

    def __init__(self, demos: dict[str, list[SkillDemo]], styles: list[dict], cfg: G1TaskConfig = G1_CFG):
        self.demos, self.styles, self.cfg = demos, styles, cfg
        self.k, self.current = 0, None

    def start_at(self, k: int):
        """Use the demos for episode number k next (evaluation runs each episode in a fresh process)."""
        self.k = k

    def reset(self):
        a, b, top, k = place_demos_by_cup(self.demos) + (self.k,)
        self.k += 1
        place = {0: a[k % len(a)], 1: b[k % len(b)], 2: top[k % len(top)]}
        style = self.styles[k % len(self.styles)] if self.styles else None
        self.current = G1Retargeter(place, expert_params(style), self.cfg)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.current is None:
            self.reset()
        return self.current(obs)
