"""Gymnasium environments for the G1 cup pyramid.

One class, three tasks:
- "place": put one cup (stage 0: base A, 1: base B, 2: top) from where it lies (mouth down) onto the pyramid;
  earlier cups are already in place. The top cup goes on mouth-up, so it has to be turned over.
- "toss": the pyramid is built; pick the cap off the table and throw it into the top cup from at least
  `min_throw_distance` away (stage 3). A miss is not the end: the cap is picked up wherever it landed and
  thrown again. Once it is in, a thumbs-up (stage 4).
- "full": everything from the start. The stage advances automatically when a stage succeeds; the
  observation tells the policy which stage it is in.
"""

import gymnasium as gym
import mujoco
import numpy as np

from robot_arm.humanoid.config import G1_CFG, G1TaskConfig
from robot_arm.humanoid.controller import (G1HandController, hand_targets, pinch_frame, rotation_error,
                                           rotvec_to_mat, site_for_pinch)
from robot_arm.humanoid.objects import (DOWN, ORIENTATIONS, START_ORIENTATIONS, UP, PyramidLayout,
                                        base_pair_ok, base_spacing, cap_in_cup, cup_axis, cup_grasp_point,
                                        cup_in_place, pyramid_standing, sample_layout, top_cup_resting)
from robot_arm.humanoid.scene import OBJ_TYPE, RIGHT_ARM_COLLIDERS, G1Ids, build_model

N_STAGES = 5  # place cup0, place cup1, place cup2, toss, thumbs-up
TOSS, THUMBS_UP = 3, 4
SUCCESS_REWARD = 10.0
# Observation layout.
OBS_SITE = slice(27, 30)
OBS_HAND_ROT = slice(30, 36)
OBS_CUPS = slice(36, 54)  # 3 x (position, mouth axis)
OBS_CAP = slice(54, 57)
OBS_CAP_VEL = slice(57, 60)
OBS_STAGE = slice(60, 65)
OBS_TARGET = slice(65, 68)
OBS_TO_OBJECT = slice(68, 71)  # active object's grasp point - grasp site (thumbs-up: the spot - grasp site)
OBS_OBJ_TO_TARGET = slice(71, 74)  # target - active object
OBS_GRIP = 74
OBS_CMD_POS = slice(75, 78)  # commanded grasp-site position - actual
OBS_CMD_ROT = slice(78, 81)  # rotation from the actual hand orientation to the commanded one
OBS_DIM = 81
# With cfg.real these follow: pelvis position (3), the pelvis's up axis (3), arm motor temperatures (10).
OBS_REAL = slice(81, 97)
OBS_DIM_REAL = 97
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
        self.body, self.seen = None, None
        if cfg.real:
            from robot_arm.humanoid.real import REAL, RealBody, spec_of

            self.body = RealBody(self.model, self.data, self.ids)
            self.controller.max_speed = np.minimum(REAL.speed_margin * np.array([spec_of(n)[1] for n in self.ids.arm_names]),
                                                   REAL.working_speed)
            self.controller.max_speed[:3] = REAL.waist_speed
            self.controller.posture_gain = REAL.posture_gain
            self.controller.max_accel = np.full(10, np.inf)
            self.controller.max_accel[:3] = REAL.waist_accel
            self.controller.joint_scale[:3] = 1.0 / REAL.waist_weight
        self.pinch_posture = self._pinch_posture(cfg.cap_spot)
        self.forearm_keep_out = self._forearm_keep_out()
        self.render_mode, self.camera = render_mode, camera
        self._renderer = mujoco.Renderer(self.model, height, width) if render_mode else None
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM_REAL if cfg.real else OBS_DIM,),
                                                np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        seconds = {"place": cfg.place_seconds, "toss": cfg.toss_seconds,
                   "full": 3 * cfg.place_seconds + cfg.toss_seconds}[task]
        self.max_steps = int(round(seconds / cfg.control_dt))
        self.ws_lo = np.array([0.05, -0.55, cfg.table_height + 0.01])
        self.ws_hi = np.array([0.65, 0.35, cfg.table_height + 0.55])

    def _pinch_posture(self, cap_xy) -> np.ndarray:
        """Arm posture for the cap pinch (hand turned palm-down over the cap): the IK's null space pulls towards
        it while tossing, so the arm does not wander into its joint limits while turning the hand."""
        if "pinch" not in self.ids.hand.poses:  # (a hand whose pinch isn't designed yet)
            return self.controller.home
        u = np.array([np.cos(self.cfg.pyramid_yaw), np.sin(self.cfg.pyramid_yaw), 0.0])
        c, s = np.cos(self.ids.hand.pick_turn), np.sin(self.ids.hand.pick_turn)
        R = pinch_frame(-self.ids.hand.throw_side * np.array([c * u[0] - s * u[1], s * u[0] + c * u[1], 0.0]), self.ids.hand)
        above_cap = np.array([*cap_xy[:2], self.cfg.table_height + 0.12])
        return self.controller.solve(self.ids.ready_qpos, site_for_pinch(above_cap, R, self.ids.hand), R,
                                     q_start=self.controller.home, iters=400)

    def _forearm_keep_out(self) -> dict[int, np.ndarray]:
        """For base cups A and B: xy points (relative to the cup's target) along the forearm, from the elbow to the
        grasp site, that are lower than a waiting cup's top (plus the links' thickness) while the cup is put down.
        The arm reaches the pyramid low and from behind: with the Shadow hand the wrist passes 10-30 cm behind the
        target at the cups' height, and lowering cup A onto it would rest the wrist on a cup waiting there."""
        from robot_arm.humanoid.scripted import grasp_frame

        cfg, m, ids = self.cfg, self.model, self.ids
        d = mujoco.MjData(m)
        T, h = cfg.table_height, cfg.cup.height
        R = grasp_frame(ids.hand.expert.get("place_yaw", 0.0), False, cfg)
        links = [m.body(n).id for n in RIGHT_ARM_COLLIDERS]
        layout = PyramidLayout.make(np.asarray(cfg.pyramid_center), cfg.pyramid_yaw, cfg)
        out = {}
        for k, target in enumerate((layout.base_a, layout.base_b)):
            pts = []
            for lift in (0.0, 0.04):  # standing on its target, and on the way down
                site = cup_grasp_point(np.asarray(target) + [0, 0, lift], np.array([0, 0, -1.0]),
                                       cfg.grasp_height_frac, cfg)
                d.qpos[:] = ids.ready_qpos
                d.qpos[ids.arm_qpos] = self.controller.solve(ids.ready_qpos, site, R, q_start=self.controller.home,
                                                             iters=300)
                mujoco.mj_kinematics(m, d)
                chain = [d.xpos[b].copy() for b in links] + [d.site_xpos[ids.grasp_site].copy()]
                for a, b in zip(chain[:-1], chain[1:]):
                    pts += [p[:2] - target[:2] for p in np.linspace(a, b, 6) if p[2] < T + h + 0.04]
            out[k] = np.array(pts).reshape(-1, 2)
        return out

    # ------------------------------------------------------------------ state
    @property
    def site_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ids.grasp_site].copy()

    @property
    def hand_mat(self) -> np.ndarray:
        return self.data.xmat[self.ids.hand_body].reshape(3, 3).copy()

    def cup_pos(self, k: int) -> np.ndarray:
        return self.seen[k].copy() if self.seen is not None else self.data.xpos[self.ids.cup_bodies[k]].copy()

    @property
    def cap_pos(self) -> np.ndarray:
        return self.seen[3].copy() if self.seen is not None else self.data.xpos[self.ids.cap_body].copy()

    @property
    def thumb_up(self) -> float:
        """Upward component of the thumb's direction (1: pointing straight up)."""
        return float((self.hand_mat @ self.ids.hand.thumb_dir)[2])

    @property
    def mouth(self) -> np.ndarray:
        return self.cup_pos(2) + cup_axis(self.data, self.ids.cup_bodies[2]) * self.cfg.cup.height

    def hand_touches(self, geoms: set) -> bool:
        hand = self.ids.hand_geoms
        for c in self.data.contact[: self.data.ncon]:
            if (c.geom1 in hand and c.geom2 in geoms) or (c.geom2 in hand and c.geom1 in geoms):
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
        if self.cfg.real:  # a pyramid already standing, the hand at its ready pose: the throw alone
            return self._reset_scene(options, stage=TOSS)
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
        self.stage = TOSS
        self.placed[2] = self.cup_pos(2)
        self.steps, self.events, self.fail_reason, self.release_distance = 0, [], None, None
        self.misses, self.thumbs_steps = 0, 0
        self._phi = self._potential()
        return self._obs(), self._info(False)

    def _reset_scene(self, options: dict, stage: int | None = None):
        """Layout, cups and cap, the arm at its ready pose. Cups before the start stage stand in place."""
        rng, cfg = self.np_random, self.cfg
        m, d, ids = self.model, self.data, self.ids
        mujoco.mj_resetData(m, d)
        if self.body is not None:
            self.body.sample_world(rng)
        self.seen = None
        d.qpos[:] = ids.ready_qpos
        d.qpos[ids.arm_qpos[3:]] += rng.normal(0, 0.03, 7)

        j = cfg.layout_jitter
        center = np.asarray(options.get("pyramid_center", cfg.pyramid_center)) + rng.uniform(-j, j, 2)
        yaw = options.get("pyramid_yaw", cfg.pyramid_yaw + rng.uniform(-0.15, 0.15))
        self.layout = PyramidLayout.make(center, yaw, cfg)
        if "supply" in options or not cfg.random_layout:
            supply = [np.asarray(p) + rng.uniform(-j, j, 2) for p in options.get("supply", cfg.supply_cups)]
            cap_xy = np.asarray(options.get("cap", cfg.cap_spot)) + rng.uniform(-j, j, 2)
        else:
            supply, cap_xy = sample_layout(rng, self.layout, cfg, keep_out=self.forearm_keep_out)
            cap_xy = np.asarray(options.get("cap", cap_xy))

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
        self._set_free(ids.cap_qpos, [*cap_xy, T + cfg.cap_half_height + 0.001], UP)
        self.pinch_posture = self._pinch_posture(cap_xy)

        for i in range(m.nu):
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
        self.pinch_ready = False
        d.ctrl[ids.hand_ctrl] = hand_targets(-1.0, "wrap", ids.hand)
        mujoco.mj_forward(m, d)
        if self.body is not None:
            self._settle_on_feet()
        else:
            mujoco.mj_step(m, d, nstep=100)  # let everything settle

        self.grip = -1.0
        self.hand_state = -1.0
        self.cmd_pos, self.cmd_mat = self.site_pos, self.hand_mat
        self.controller.reset()
        self.seen = None
        self.steps = 0
        # Where each finished cup was left; it must stay there.
        self.placed = {k: self.cup_pos(k) for k in range(prebuilt)}
        self.cap_was_held = False
        self.release_distance = None
        self.misses, self.thumbs_steps, self.cap_still, self.settled = 0, 0, 0, 0
        self.events: list[str] = []
        self.fail_reason = None
        self._phi = self._potential()
        return self._obs(), self._info(False)

    def _settle_on_feet(self):
        m, d, body = self.model, self.data, self.body
        drop = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in range(m.ngeom)
                   if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and m.geom_contype[g] == OBJ_TYPE)
        d.qpos[2] -= drop  # the soles on the floor
        mujoco.mj_forward(m, d)
        arm = d.ctrl[self.ids.arm_ctrl].copy()
        body.stand_reset()
        for _ in range(int(0.4 / (self.cfg.control_dt / self.cfg.ik_substeps))):
            d.ctrl[self.ids.arm_ctrl] = arm
            body.tick(self.cfg.control_dt / self.cfg.ik_substeps)
            mujoco.mj_step(m, d, nstep=self.cfg.frame_skip // self.cfg.ik_substeps)

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
        if self.stage == TOSS:
            return self.mouth
        x, y, z = self.cfg.thumbs_up_spot
        return np.array([x, y, self.cfg.table_height + z])

    def active_grasp_point(self) -> np.ndarray:
        if self.stage == TOSS:
            return self.cap_pos
        if self.stage == THUMBS_UP:
            return self.stage_target()
        k = self.stage
        frac = self.cfg.top_grasp_height_frac if k == 2 else self.cfg.grasp_height_frac
        return cup_grasp_point(self.cup_pos(k), cup_axis(self.data, self.ids.cup_bodies[k]), frac, self.cfg)

    def active_object(self) -> np.ndarray:
        if self.stage == THUMBS_UP:
            return self.site_pos  # the hand itself goes to the spot
        return self.cap_pos if self.stage == TOSS else self.cup_pos(self.stage)

    # ------------------------------------------------------------------ step
    def max_step(self) -> float:
        return self.cfg.toss_max_step if self.stage >= TOSS else self.cfg.max_step

    def step(self, action):
        a = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float64)), -1.0, 1.0)
        cfg = self.cfg
        # Actions move a stored command pose, so a zero action holds the arm still and a stop is a real stop.
        # A leash keeps the command near the actual hand (e.g. when pushing against the table).
        leash = cfg.leash * (self.max_step() / cfg.max_step)
        if cfg.real:  # the command only runs ahead of the hand as far as this step asks: the throw's reach is for the swing
            leash = max(cfg.leash, min(leash, 1.5 * float(np.abs(a[:3]).max()) * self.max_step()))
        site = self.site_pos
        prev_pos, prev_mat = self.cmd_pos.copy(), self.cmd_mat.copy()
        target_pos = self.cmd_pos + a[:3] * self.max_step()
        self.cmd_pos = np.clip(site + np.clip(target_pos - site, -leash, leash), self.ws_lo, self.ws_hi)
        self.cmd_mat = rotvec_to_mat(a[3:6] * cfg.max_rot_step) @ self.cmd_mat
        rot_leash = 0.12 if cfg.real else cfg.rot_leash
        if np.linalg.norm(rotation_error(self.hand_mat, self.cmd_mat)) > rot_leash:
            # Pull the commanded orientation back towards the actual one.
            err = rotation_error(self.hand_mat, self.cmd_mat)
            self.cmd_mat = rotvec_to_mat(err * rot_leash / np.linalg.norm(err)) @ self.hand_mat
        self.grip = float(a[6])
        # Fingers move at a finite speed (a snap-open flings a light cup away); a throw needs a quick release.
        close, open_ = (cfg.toss_grip_rate,) * 2 if self.stage == TOSS else (cfg.grip_rate, cfg.release_rate)
        self.hand_state += np.clip(self.grip - self.hand_state, -open_, close)
        # Interpolate the command over the control period so fast motions are tracked smoothly.
        n = cfg.ik_substeps
        rot = rotation_error(prev_mat, self.cmd_mat)
        # The hand takes the pinch shape for the cap only once it is above the pyramid: changing shape right
        # after letting go of the top cup would knock it with the fingers.
        if self.stage == TOSS and not self.pinch_ready:
            self.pinch_ready = self.site_pos[2] > cfg.table_height + 2 * cfg.cup.height + 0.05
        pinch = self.stage == TOSS and self.pinch_ready
        shape = "pinch" if pinch else "thumbs_up" if self.stage == THUMBS_UP else "wrap"
        self.controller.posture = self.pinch_posture if pinch else self.controller.home
        late = 0 if self.body is None else self.body.action_delay  # the new command reaches the servos this many ticks late
        for i in range(1, n + 1):
            f = max(i - late, 0) / (n - late)
            self.controller.apply(self.data, prev_pos + f * (self.cmd_pos - prev_pos),
                                  rotvec_to_mat(f * rot) @ prev_mat, self.hand_state, cfg.control_dt / n,
                                  shape=shape)
            if self.body is not None:
                self.body.tick(cfg.control_dt / n)
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
            if self.task == "full" and self.stage < TOSS:  # the task ends with the cap in the cup: no thumbs-up
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
        if self.body is not None:
            if self.body.fallen:
                return self._fail("robot fell")
            if self.body.fault:
                return self._fail(self.body.fault)
        for k in range(3):
            if self.cup_pos(k)[2] < T - 0.03:
                return self._fail(f"cup{k} fell off the table")
        if self.cap_pos[2] < T - 0.03:
            return self._fail("cap fell off the table")
        if not self._others_in_place():
            return self._fail("knocked a placed cup")
        if self.stage < TOSS:
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
            # Let go, and the hand out of the way (backing the hand out of a top cup can still knock it over).
            centre = self.cup_pos(k) + cup_axis(d, body) * cfg.cup.height / 2
            released = not self.hand_touches(ids.cup_geoms[k]) and self.grip < 0 and \
                np.linalg.norm(self.site_pos - centre) > 0.10
            # The top cup rests on two rims: it must stay put for half a second (one let go of badly tips over).
            self.settled = self.settled + 1 if placed and still and released else 0
            if self.settled >= (10 if k == 2 else 1):
                self.placed[k] = self.cup_pos(k)
                self.settled = 0
                return True, False
            return False, False
        in_cup = cap_in_cup(d, ids.cap_body, ids.cup_bodies[2], cfg.cap_radius, cfg)
        if self.stage == THUMBS_UP:
            if not in_cup:
                return self._fail("cap came out of the cup")
            clear = np.linalg.norm(self.site_pos - self.mouth) > 0.12
            shown = self.grip > 0 and self.hand_state > 0.8 and self.thumb_up > cfg.thumbs_up_min_tilt and clear
            self.thumbs_steps = self.thumbs_steps + 1 if shown else 0
            return self.thumbs_steps * cfg.control_dt >= cfg.thumbs_up_hold, False
        # Toss.
        if in_cup and pyramid_standing(d, ids.cup_bodies, cfg):
            far_enough = self.release_distance is not None and self.release_distance >= cfg.min_throw_distance
            return (True, False) if far_enough else self._fail("cap dropped in from too close")
        # A miss: once the thrown cap has come to rest outside the cup, it can be picked up and thrown again.
        if self.release_distance is not None and not self.hand_touches({ids.cap_geom}):
            slow = np.linalg.norm(d.qvel[ids.cap_dof:ids.cap_dof + 3]) < 0.03
            self.cap_still = self.cap_still + 1 if slow else 0
            if self.cap_still >= 3:
                self.misses += 1
                self.events.append(f"miss@{self.steps}")
                self.release_distance, self.cap_was_held, self.cap_still = None, False, 0
        return False, False

    def _fail(self, reason: str) -> tuple[bool, bool]:
        self.fail_reason = reason
        return False, True

    def _potential(self) -> float:
        """Staged progress in the current stage: reaching, holding the object near its target, placed (or the
        cap thrown towards the mouth)."""
        to_obj = np.linalg.norm(self.active_grasp_point() - self.site_pos)
        r = 0.2 * (1 - np.tanh(8 * to_obj))
        if self.stage == THUMBS_UP:
            shape = max(0.0, (self.hand_state + 1) / 2) * max(0.0, self.thumb_up)
            hold = min(1.0, self.thumbs_steps * self.cfg.control_dt / self.cfg.thumbs_up_hold)
            return float(r + 0.4 * shape + 0.4 * hold)
        if self.stage < TOSS:
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
        if self.body is None:
            return self._read_obs()
        ids = self.ids
        truth = np.array([self.data.xpos[b] for b in ids.cup_bodies] + [self.data.xpos[ids.cap_body]])
        self.seen = self.body.camera(truth, self.site_pos, self.hand_mat, self.grip > 0, min(self.stage, 3))  # during this, the objects are where the robot believes
        try:
            obs = self._read_obs()
        finally:
            self.seen = None
        obs[:10] += self.body.rng.normal(0, self.body.spec.joint_noise, 10)
        pelvis = self.data.xpos[self.body.pelvis]
        up = self.data.xmat[self.body.pelvis][6:9]
        return np.concatenate([obs, pelvis, up, self.body.arm_temps / 100]).astype(np.float32)

    def _read_obs(self) -> np.ndarray:
        d, ids = self.data, self.ids
        cups = np.concatenate([np.concatenate([self.cup_pos(k), cup_axis(d, ids.cup_bodies[k])]) for k in range(3)])
        cap_vel = d.qvel[ids.cap_dof:ids.cap_dof + 3]
        stage = np.zeros(N_STAGES)
        stage[self.stage] = 1.0
        target = self.stage_target()
        obs = np.concatenate([
            d.qpos[ids.arm_qpos], d.qvel[ids.arm_dofs] * 0.1, d.qpos[ids.hand_qpos],
            self.site_pos, self.hand_mat[:, :2].T.ravel(), cups, self.cap_pos, cap_vel, stage, target,
            self.active_grasp_point() - self.site_pos, target - self.active_object(), [self.grip],
            self.cmd_pos - self.site_pos, rotation_error(self.hand_mat, self.cmd_mat),
        ])
        return obs.astype(np.float32)

    def _info(self, success: bool) -> dict:
        # How much the placed cups were disturbed this step (for reward terms outside the env, e.g. VLA + RL):
        # the hand touching one, and the farthest any of them has moved from where it was left, in metres.
        placed = set().union(*(self.ids.cup_geoms[k] for k in self.placed)) if self.placed else set()
        shift = max((float(np.linalg.norm(self.cup_pos(k) - p)) for k, p in self.placed.items()), default=0.0)
        return {"is_success": bool(success), "stage": int(self.stage), "events": list(self.events),
                "release_distance": self.release_distance, "fail_reason": self.fail_reason, "misses": self.misses,
                "placed_touch": bool(placed) and self.hand_touches(placed), "placed_shift": shift}

    def render(self, camera: str | None = None):
        if self._renderer is None:
            raise RuntimeError("Create the env with render_mode='rgb_array' to render.")
        opt = mujoco.MjvOption()
        self._renderer.update_scene(self.data, camera=camera or self.camera, scene_option=opt)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register("G1Place-v0", entry_point=G1PyramidEnv, kwargs={"task": "place"})
gym.register("G1Toss-v0", entry_point=G1PyramidEnv, kwargs={"task": "toss"})
gym.register("G1Pyramid-v0", entry_point=G1PyramidEnv, kwargs={"task": "full"})
