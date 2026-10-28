"""Gymnasium environments for the G1 cup pyramid.

One class, three tasks:
- "place": put one cup (stage 0: base A, 1: base B, 2: top) from its supply spot onto the pyramid;
  earlier cups are already in place.
- "toss": the pyramid is built; pick the cap off its stand and throw it into the top cup from at least
  `min_throw_distance` away.
- "full": everything from the supply spots. The stage advances automatically when a stage succeeds; the
  observation tells the policy which stage it is in.
"""

import gymnasium as gym
import mujoco
import numpy as np

from robot_arm.humanoid.config import G1_CFG, G1TaskConfig
from robot_arm.humanoid.controller import (G1HandController, hand_targets, pinch_frame, rotation_error,
                                           rotvec_to_mat, site_for_pinch)
from robot_arm.humanoid.objects import (DOWN, ORIENTATIONS, START_ORIENTATIONS, TOP_TILT_DEG, UP, PyramidLayout,
                                        base_pair_ok,
                                        base_spacing, cap_in_cup, cup_axis, cup_grasp_point, cup_in_place,
                                        pyramid_standing, top_cup_resting)
from robot_arm.humanoid.scene import G1Ids, build_model

N_STAGES = 4  # place cup0, place cup1, place cup2, toss
SUCCESS_REWARD = 10.0
# Observation layout.
OBS_SITE = slice(27, 30)
OBS_HAND_ROT = slice(30, 36)
OBS_CUPS = slice(36, 54)  # 3 x (position, mouth axis)
OBS_CAP = slice(54, 57)
OBS_CAP_VEL = slice(57, 60)
OBS_STAGE = slice(60, 64)
OBS_TARGET = slice(64, 67)
OBS_TO_OBJECT = slice(67, 70)  # active object's grasp point - grasp site
OBS_OBJ_TO_TARGET = slice(70, 73)  # target - active object
OBS_GRIP = 73
OBS_CMD_POS = slice(74, 77)  # commanded grasp-site position - actual
OBS_CMD_ROT = slice(77, 80)  # rotation from the actual hand orientation to the commanded one
OBS_DIM = 80
ACT_DIM = 7  # palm translation (3), palm rotation (3), grip (1: -1 open .. 1 closed)


def obs_cup(obs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    c = obs[OBS_CUPS]
    return c[6 * k:6 * k + 3], c[6 * k + 3:6 * k + 6]


class G1PyramidEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / G1_CFG.control_dt)}

    def __init__(self, task: str = "place", stage: int | None = None, cfg: G1TaskConfig = G1_CFG,
                 render_mode: str | None = None, camera: str = "front", width: int = 640, height: int = 480):
        assert task in ("place", "toss", "full")
        self.task, self.fixed_stage, self.cfg = task, stage, cfg
        self.model = build_model(cfg)
        self.data = mujoco.MjData(self.model)
        self.ids = G1Ids(self.model)
        self.controller = G1HandController(self.model, self.ids)
        # Arm posture for the cap pinch (hand turned palm-down over the cap spot): the IK's null space pulls
        # towards it while tossing, so the arm does not wander into its joint limits while turning the hand.
        u = np.array([np.cos(cfg.pyramid_yaw), np.sin(cfg.pyramid_yaw), 0.0])
        R = pinch_frame(-u)
        above_cap = np.array([*cfg.cap_spot, cfg.table_height + 0.12])
        self.pinch_posture = self.controller.solve(self.ids.ready_qpos, site_for_pinch(above_cap, R), R,
                                                   q_start=self.controller.home, iters=400)
        self.render_mode, self.camera = render_mode, camera
        self._renderer = mujoco.Renderer(self.model, height, width) if render_mode else None
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        seconds = {"place": cfg.place_seconds, "toss": cfg.toss_seconds,
                   "full": 3 * cfg.place_seconds + cfg.toss_seconds}[task]
        self.max_steps = int(round(seconds / cfg.control_dt))
        self.ws_lo = np.array([0.05, -0.55, cfg.table_height + 0.01])
        self.ws_hi = np.array([0.65, 0.35, cfg.table_height + 0.55])

    # ------------------------------------------------------------------ state
    @property
    def site_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ids.grasp_site].copy()

    @property
    def hand_mat(self) -> np.ndarray:
        return self.data.xmat[self.ids.hand_body].reshape(3, 3).copy()

    def cup_pos(self, k: int) -> np.ndarray:
        return self.data.xpos[self.ids.cup_bodies[k]].copy()

    @property
    def cap_pos(self) -> np.ndarray:
        return self.data.xpos[self.ids.cap_body].copy()

    def hand_touches(self, geoms: set) -> bool:
        hand = self.ids.hand_geoms
        for c in self.data.contact[: self.data.ncon]:
            if (c.geom1 in hand and c.geom2 in geoms) or (c.geom2 in hand and c.geom1 in geoms):
                return True
        return False

    def touches(self, a: set, b: set) -> bool:
        for c in self.data.contact[: self.data.ncon]:
            if (c.geom1 in a and c.geom2 in b) or (c.geom2 in a and c.geom1 in b):
                return True
        return False

    # ------------------------------------------------------------------ reset
    def _set_free(self, qadr: int, pos, quat):
        self.data.qpos[qadr:qadr + 3] = pos
        self.data.qpos[qadr + 3:qadr + 7] = quat

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if self.task != "toss":
            return self._reset_scene(options)
        # The throw starts where it does in the full task: the robot has just let go of the top cup it put on
        # the base cups, with the hand still around it. The scripted expert places it (a few tries if needed).
        from robot_arm.humanoid.scripted import PlaceExpert

        for _ in range(5):
            obs, _ = self._reset_scene(options, stage=2)
            expert = PlaceExpert(self.cfg)
            for _ in range(int(round(self.cfg.place_seconds / self.cfg.control_dt))):
                obs, _, terminated, _, info = self.step(expert(obs))
                if terminated:
                    break
            if info["is_success"]:
                break
        self.stage = 3
        self.placed[2] = self.cup_pos(2)
        self.steps, self.events, self.fail_reason, self.release_distance = 0, [], None, None
        self._phi = self._potential()
        return self._obs(), self._info(False)

    def _reset_scene(self, options: dict, stage: int | None = None):
        """Layout, cups and cap, the arm at its ready pose. Cups before the start stage stand in place."""
        rng, cfg = self.np_random, self.cfg
        m, d, ids = self.model, self.data, self.ids
        mujoco.mj_resetData(m, d)
        d.qpos[:] = ids.ready_qpos
        d.qpos[ids.arm_qpos[3:]] += rng.normal(0, 0.03, 7)

        j = cfg.layout_jitter
        center = np.asarray(options.get("pyramid_center", cfg.pyramid_center)) + rng.uniform(-j, j, 2)
        yaw = options.get("pyramid_yaw", cfg.pyramid_yaw + rng.uniform(-0.15, 0.15))
        self.layout = PyramidLayout.make(center, yaw, cfg)
        supply = [np.asarray(p) + rng.uniform(-j, j, 2) for p in options.get("supply", cfg.supply_cups)]

        if stage is not None:
            self.stage = stage
        elif self.task == "place":
            self.stage = self.fixed_stage if self.fixed_stage is not None else int(rng.integers(3))
            self.stage = int(options.get("stage", self.stage))
        else:
            self.stage = 0
        prebuilt = self.stage  # cups with index < stage start in their final place
        T = cfg.table_height
        for k in range(3):
            if k < prebuilt:
                pos = self.layout.targets()[k] + np.array([*rng.normal(0, 0.002, 2), 0.0])
                if k == 1:  # the robot sets cup B down 0-4.5 mm further out than its target
                    u = self.layout.base_b[:2] - self.layout.base_a[:2]
                    pos[:2] += rng.uniform(0.0, 0.0045) * u / np.linalg.norm(u)
            else:
                z = T + (cfg.cup.height if START_ORIENTATIONS[k] is DOWN else 0.0)
                pos = np.array([*supply[k], z])
            self._set_free(ids.cup_qpos[k], pos, ORIENTATIONS[k] if k < prebuilt else START_ORIENTATIONS[k])
        cap_xy = np.asarray(options.get("cap", cfg.cap_spot)) + rng.uniform(-j, j, 2)
        self._set_free(ids.cap_qpos, [*cap_xy, T + cfg.cap_half_height + 0.001], UP)

        for i in range(m.nu):
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
        self.pinch_ready = False
        d.ctrl[ids.hand_ctrl] = hand_targets(-1.0)
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d, nstep=100)  # let everything settle

        self.grip = -1.0
        self.hand_state = -1.0
        self.cmd_pos, self.cmd_mat = self.site_pos, self.hand_mat
        self.controller.reset()
        self.steps = 0
        # Where each finished cup was left; it must stay there.
        self.placed = {k: self.cup_pos(k) for k in range(prebuilt)}
        self.cap_was_held = False
        self.release_distance = None
        self.events: list[str] = []
        self.fail_reason = None
        self._phi = self._potential()
        return self._obs(), self._info(False)

    # ------------------------------------------------------------------ goals
    def stage_target(self) -> np.ndarray:
        if self.stage == 0:
            return self.layout.base_a
        if self.stage == 1:
            # Next to cup A wherever it actually is, along the pyramid's direction.
            u = self.layout.base_b[:2] - self.layout.base_a[:2]
            u /= np.linalg.norm(u)
            return np.array([*(self.cup_pos(0)[:2] + base_spacing(self.cfg) * u), self.layout.base_b[2]])
        if self.stage == 2:
            # The top cup goes on the base cups wherever they actually ended up.
            mid = (self.cup_pos(0)[:2] + self.cup_pos(1)[:2]) / 2
            return np.array([*mid, self.layout.top[2]])
        return self.cup_pos(2) + cup_axis(self.data, self.ids.cup_bodies[2]) * self.cfg.cup.height

    def active_grasp_point(self) -> np.ndarray:
        if self.stage == 3:
            return self.cap_pos
        k = self.stage
        frac = self.cfg.top_grasp_height_frac if k == 2 else self.cfg.grasp_height_frac
        return cup_grasp_point(self.cup_pos(k), cup_axis(self.data, self.ids.cup_bodies[k]), frac, self.cfg)

    def active_object(self) -> np.ndarray:
        return self.cap_pos if self.stage == 3 else self.cup_pos(self.stage)

    # ------------------------------------------------------------------ step
    def max_step(self) -> float:
        return self.cfg.toss_max_step if self.stage == 3 else self.cfg.max_step

    def step(self, action):
        a = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float64)), -1.0, 1.0)
        cfg = self.cfg
        # Actions move a stored command pose, so a zero action holds the arm still and a stop is a real stop.
        # A leash keeps the command near the actual hand (e.g. when pushing against the table).
        leash = cfg.leash * (self.max_step() / cfg.max_step)
        site = self.site_pos
        prev_pos, prev_mat = self.cmd_pos.copy(), self.cmd_mat.copy()
        target_pos = self.cmd_pos + a[:3] * self.max_step()
        self.cmd_pos = np.clip(site + np.clip(target_pos - site, -leash, leash), self.ws_lo, self.ws_hi)
        self.cmd_mat = rotvec_to_mat(a[3:6] * cfg.max_rot_step) @ self.cmd_mat
        if np.linalg.norm(rotation_error(self.hand_mat, self.cmd_mat)) > cfg.rot_leash:
            # Pull the commanded orientation back towards the actual one.
            err = rotation_error(self.hand_mat, self.cmd_mat)
            self.cmd_mat = rotvec_to_mat(err * cfg.rot_leash / np.linalg.norm(err)) @ self.hand_mat
        self.grip = float(a[6])
        # Fingers move at a finite speed (a snap-open flings a light cup away); a throw needs a quick release.
        close, open_ = (cfg.toss_grip_rate,) * 2 if self.stage == 3 else (cfg.grip_rate, cfg.release_rate)
        self.hand_state += np.clip(self.grip - self.hand_state, -open_, close)
        # Interpolate the command over the control period so fast motions are tracked smoothly.
        n = cfg.ik_substeps
        rot = rotation_error(prev_mat, self.cmd_mat)
        # The hand takes the pinch shape for the cap only once it is above the pyramid: changing shape right
        # after letting go of the top cup would knock it with the fingers.
        if self.stage == 3 and not self.pinch_ready:
            self.pinch_ready = self.site_pos[2] > cfg.table_height + 2 * cfg.cup.height + 0.05
        pinch = self.stage == 3 and self.pinch_ready
        self.controller.posture = self.pinch_posture if pinch else self.controller.home
        for i in range(1, n + 1):
            f = i / n
            self.controller.apply(self.data, prev_pos + f * (self.cmd_pos - prev_pos),
                                  rotvec_to_mat(f * rot) @ prev_mat, self.hand_state, cfg.control_dt / n,
                                  pinch=pinch)
            mujoco.mj_step(self.model, self.data, nstep=cfg.frame_skip // n)
        self.steps += 1

        self._track_cap()
        success, failed = self._stage_status()
        # Potential-based shaping: the step's reward is the change in staged progress, and the progress counts
        # for nothing once a stage ends (succeeded or failed), so only the success bonus is kept. Rewarding the
        # progress itself every step paid more for holding a placed cup until the time ran out than for letting go.
        phi = 0.0 if success or failed else self._potential()
        reward = (SUCCESS_REWARD if success else 0.0) + phi - self._phi
        self._phi = phi
        stage_done = success
        terminated = failed
        if success:
            self.events.append(f"stage{self.stage}@{self.steps}")
            if self.task == "full" and self.stage < 3:
                self.stage += 1
                self._phi = self._potential()
                stage_done = False
            else:
                terminated = True
        truncated = self.steps >= self.max_steps
        info = self._info(stage_done and terminated)
        return self._obs(), reward, bool(terminated), bool(truncated), info

    def _track_cap(self):
        ids = self.ids
        held = self.hand_touches({ids.cap_geom}) and self.cap_pos[2] > self.cfg.table_height + 0.04
        if held:
            self.cap_was_held = True
        elif self.cap_was_held and self.release_distance is None and not self.hand_touches({ids.cap_geom}):
            mouth = self.layout.top + [0, 0, self.cfg.cup.height]
            self.release_distance = float(np.linalg.norm(self.cap_pos[:2] - mouth[:2]))

    def _others_in_place(self) -> bool:
        """Cups finished in earlier stages are still where they were left. The top cup only has to stay resting
        on the base cups: it may settle a little lower between them after it is let go."""
        for k, pos in self.placed.items():
            if k == 2:
                if not top_cup_resting(self.data, self.ids.cup_bodies, self.cfg):
                    return False
            elif not cup_in_place(self.data, self.ids.cup_bodies[k], pos, False, pos_tol=0.012, tilt_tol_deg=12.0):
                return False
        return True

    def _stage_status(self) -> tuple[bool, bool]:
        """(stage succeeded, episode failed)."""
        d, ids, cfg = self.data, self.ids, self.cfg
        T = cfg.table_height
        if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
            return self._fail("simulation became unstable")
        for k in range(3):
            if self.cup_pos(k)[2] < T - 0.03:
                return self._fail(f"cup{k} fell off the table")
        if self.cap_pos[2] < T - 0.03:
            return self._fail("cap fell off the table")
        if not self._others_in_place():
            return self._fail("knocked a placed cup")
        if self.stage < 3:
            k = self.stage
            body = ids.cup_bodies[k]
            # Cup A only fixes where the pyramid stands, so its tolerance is loose; the others are relative to it.
            if k == 0:
                placed = cup_in_place(d, body, self.stage_target(), False, pos_tol=0.04)
            elif k == 1:  # what matters for the pyramid is the distance to cup A
                placed = base_pair_ok(d, ids.cup_bodies, cfg)
            else:
                placed = top_cup_resting(d, ids.cup_bodies, cfg)
            still = np.linalg.norm(d.cvel[body][3:]) < 0.05
            released = not self.hand_touches(ids.cup_geoms[k]) and self.grip < 0
            if placed and still and released:
                self.placed[k] = self.cup_pos(k)
                return True, False
            return False, False
        # Toss.
        in_cup = cap_in_cup(d, ids.cap_body, ids.cup_bodies[2], cfg.cap_radius, cfg)
        if in_cup and pyramid_standing(d, ids.cup_bodies, cfg):
            far_enough = self.release_distance is not None and self.release_distance >= cfg.min_throw_distance
            return (True, False) if far_enough else self._fail("cap dropped in from too close")
        if self.release_distance is not None and self.cap_pos[2] < T + 0.03:
            return self._fail("cap missed the cup")
        return False, False

    def _fail(self, reason: str) -> tuple[bool, bool]:
        self.fail_reason = reason
        return False, True

    def _potential(self) -> float:
        """Staged progress in the current stage: reaching, holding the object near its target, placed (or the
        cap thrown towards the mouth)."""
        to_obj = np.linalg.norm(self.active_grasp_point() - self.site_pos)
        r = 0.2 * (1 - np.tanh(8 * to_obj))
        if self.stage < 3:
            k = self.stage
            holding = self.hand_touches(self.ids.cup_geoms[k]) and self.grip > 0
            if holding:
                to_target = np.linalg.norm(self.stage_target() - self.cup_pos(k))
                r = 0.4 + 0.6 * (1 - np.tanh(6 * to_target))
            elif cup_in_place(self.data, self.ids.cup_bodies[k], self.stage_target(), ORIENTATIONS[k] is UP):
                r = 1.2  # placed; now let go and back off
        else:
            if self.cap_was_held and self.release_distance is None:
                r = 0.4
            elif self.release_distance is not None:
                mouth = self.stage_target()
                r = 0.4 + 0.8 * (1 - np.tanh(6 * np.linalg.norm(self.cap_pos - mouth)))
        return float(r)

    # ------------------------------------------------------------------ observation
    def _obs(self) -> np.ndarray:
        d, ids = self.data, self.ids
        cups = np.concatenate([np.concatenate([self.cup_pos(k), cup_axis(d, ids.cup_bodies[k])]) for k in range(3)])
        cap_vel = d.qvel[ids.cap_dof:ids.cap_dof + 3]
        stage = np.zeros(N_STAGES)
        stage[min(self.stage, 3)] = 1.0
        target = self.stage_target()
        obs = np.concatenate([
            d.qpos[ids.arm_qpos], d.qvel[ids.arm_dofs] * 0.1, d.qpos[ids.hand_qpos],
            self.site_pos, self.hand_mat[:, :2].T.ravel(), cups, self.cap_pos, cap_vel, stage, target,
            self.active_grasp_point() - self.site_pos, target - self.active_object(), [self.grip],
            self.cmd_pos - self.site_pos, rotation_error(self.hand_mat, self.cmd_mat),
        ])
        return obs.astype(np.float32)

    def _info(self, success: bool) -> dict:
        return {"is_success": bool(success), "stage": int(self.stage), "events": list(self.events),
                "release_distance": self.release_distance, "fail_reason": self.fail_reason}

    def render(self):
        if self._renderer is None:
            raise RuntimeError("Create the env with render_mode='rgb_array' to render.")
        opt = mujoco.MjvOption()
        self._renderer.update_scene(self.data, camera=self.camera, scene_option=opt)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register("G1Place-v0", entry_point=G1PyramidEnv, kwargs={"task": "place"})
gym.register("G1Toss-v0", entry_point=G1PyramidEnv, kwargs={"task": "toss"})
gym.register("G1Pyramid-v0", entry_point=G1PyramidEnv, kwargs={"task": "full"})
