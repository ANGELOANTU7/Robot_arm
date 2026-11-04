"""The G1 as hardware, for G1_REAL=1: what the plain task leaves out.

- Limits from Unitree's g1_29dof_rev_1_0.urdf (torque and speed), with torque falling off towards top speed.
- Motors carry gravity: no gravity compensation in the physics. Joint servos have the gains the robot runs at, and a
  gravity feed-forward as the real robot's controller would send.
- Motor heat: first-order I^2 R model per joint; hot motors lose torque, too hot and the try stops.
- The pelvis is free and the feet stand on the floor: `RealBody.tick` keeps the robot balanced.
- A new robot and room every try (`sample_world`): gains, friction, masses, delay, what the camera gets wrong.

The heat numbers are not from Unitree, which does not publish them: `rated_frac`, `heat_tc` and the temperatures are
typical for this class of motor, and should be replaced by a measurement on the robot.
"""

import os
from dataclasses import dataclass

import mujoco
import numpy as np

# joint suffix: (peak torque Nm, top speed rad/s, servo kp, servo kd)
JOINT_SPECS = {
    "hip_pitch_joint": (88, 32, 300, 6), "hip_roll_joint": (139, 20, 300, 6), "hip_yaw_joint": (88, 32, 300, 6),
    "knee_joint": (139, 20, 300, 6), "ankle_pitch_joint": (35, 30, 200, 4), "ankle_roll_joint": (35, 30, 200, 4),
    "waist_yaw_joint": (88, 32, 200, 5), "waist_roll_joint": (35, 30, 200, 5), "waist_pitch_joint": (35, 30, 200, 5),
    "shoulder_pitch_joint": (25, 37, 100, 3), "shoulder_roll_joint": (25, 37, 100, 3),
    "shoulder_yaw_joint": (25, 37, 100, 3), "elbow_joint": (25, 37, 100, 3), "wrist_roll_joint": (25, 37, 40, 1.5),
    "wrist_pitch_joint": (5, 22, 40, 1.5), "wrist_yaw_joint": (5, 22, 40, 1.5),
}
STAND_LEGS = {"hip_pitch_joint": -0.20, "knee_joint": 0.42, "ankle_pitch_joint": -0.23}
LEGS = [f"{side}_{j}" for side in ("left", "right") for j in ("hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint",
                                                              "knee_joint", "ankle_pitch_joint", "ankle_roll_joint")]
LEFT_ARM = [f"left_{j}" for j in ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint",
                                  "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint")]
HAND_KG = 1.8  # Shadow Super Lite, the lightest real Shadow hand: a 2.4 kg one holds the 5 Nm wrist at 88%, 4.3 kg can't
PELVIS_HEIGHT = 0.777  # standing on slightly bent knees


def spec_of(joint: str) -> tuple:
    for suffix, spec in JOINT_SPECS.items():
        if joint.endswith(suffix):
            return spec
    raise KeyError(joint)


@dataclass(frozen=True)
class RealSpec:
    rated_frac: float = 0.4  # torque the motor can hold for ever, as a fraction of peak
    rated_rise: float = 60.0  # K above ambient at the rated torque
    heat_tc: float = 120.0  # thermal time constant (s)
    derate_from: float = 75.0  # C; torque falls linearly from here...
    shut_down: float = 90.0  # ...to nothing here, where the drive stops
    ambient: tuple = (20.0, 32.0)
    start_temp: tuple = (0.0, 25.0)  # K above ambient: a robot that has been working
    speed_margin: float = 0.85  # the IK stays below this fraction of top speed
    working_speed: float = 8.0  # rad/s: and below this. Top speed is for an empty motor; near it the arm whips the whole robot
    com_box: tuple = (0.05, 0.07)  # m either side of the feet's middle in which the COM is left alone
    com_gain: float = 0.5  # COM error to pelvis shift, per second
    lean_rate: float = 5.0  # 1/s: how fast a lean of the COM is caught
    waist_weight: float = 10.0  # the arm reaches, not the waist: swinging the torso throws the COM about
    waist_speed: float = 0.5  # rad/s
    waist_accel: float = 1.5  # rad/s^2
    posture_gain: float = 0.02  # 0.05 and up stop the top cup, which has to leave this posture
    # What changes from try to try (factors unless noted).
    kp: tuple = (0.8, 1.2)
    grip_friction: tuple = (0.6, 1.1)
    object_friction: tuple = (0.7, 1.2)
    cup_mass: tuple = (0.8, 1.4)
    cap_mass: tuple = (0.7, 1.5)
    hand_mass: tuple = (0.95, 1.05)
    joint_friction: tuple = (0.05, 0.4)  # Nm, on the arm and waist
    torque: tuple = (0.9, 1.0)  # battery sag
    action_delay: tuple = (0, 3)  # IK ticks, 10 ms each: policy, bus and servo loop
    camera_delay: tuple = (1, 1)  # control steps, 50 ms each: a 30 fps frame and its processing
    held_reach: float = 0.12  # m from the grip: an object this close to the closed hand moves with it
    camera_bias: float = 0.003  # m, std: calibration error, the same for all objects in a try
    object_bias: float = 0.002  # m, std: per object
    camera_jitter: float = 0.0015  # m, std: per frame
    camera_dropout: float = 0.03  # chance a frame is lost, and the last one is used
    joint_noise: float = 0.002  # rad, encoders
    push: float = 8.0  # N, std of a slow nudge on the torso


CONDITIONS = {  # each with its value in a perfect world
    "motors": {"kp": (1, 1), "torque": (1, 1), "joint_friction": (0.1, 0.1)},
    "friction": {"grip_friction": (1, 1), "object_friction": (1, 1)},
    "mass": {"cup_mass": (1, 1), "cap_mass": (1, 1), "hand_mass": (1, 1)},
    "action_delay": {"action_delay": (0, 0)},
    "camera_delay": {"camera_delay": (0, 0)},
    "camera": {"camera_bias": 0.0, "object_bias": 0.0, "camera_jitter": 0.0, "camera_dropout": 0.0, "joint_noise": 0.0},
    "push": {"push": 0.0},
}


def real_spec(setting: str) -> RealSpec:
    """G1_REAL: `1` every condition; `clean` none; `clean,camera,delay` only those, to see what each one costs."""
    names = setting.split(",")
    if names[0] != "clean":
        return RealSpec()
    off = {k: v for name, values in CONDITIONS.items() if name not in names for k, v in values.items()}
    return RealSpec(**off)


REAL = real_spec(os.environ.get("G1_REAL", "1"))


class RealBody:
    """Everything that runs next to the arm controller on the robot: balance, feed-forward, limits, heat."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ids, spec: RealSpec = REAL):
        self.m, self.d, self.ids, self.spec = model, data, ids, spec
        self.pelvis = model.body("pelvis").id
        self.scratch = mujoco.MjData(model)
        self.mass = float(model.body_subtreemass[self.pelvis])
        self.feet = [model.site("left_foot").id, model.site("right_foot").id]
        self.leg_q = [np.array([model.joint(n).qposadr[0] for n in LEGS[i:i + 6]]) for i in (0, 6)]
        self.leg_v = [np.array([model.joint(n).dofadr[0] for n in LEGS[i:i + 6]]) for i in (0, 6)]
        self.leg_ctrl = [np.array([model.actuator(n).id for n in LEGS[i:i + 6]]) for i in (0, 6)]
        self.body_names = LEGS + LEFT_ARM + ids.arm_names
        self.joints = np.array([model.joint(n).id for n in self.body_names])
        self.v_adr = model.jnt_dofadr[self.joints]
        self.ctrl = np.array([model.actuator(n).id for n in self.body_names])
        self.left_ctrl = np.array([model.actuator(n).id for n in LEFT_ARM])
        self.left_q = np.array([model.joint(n).qposadr[0] for n in LEFT_ARM])
        self.arm_slice = slice(len(LEGS) + len(LEFT_ARM), None)
        specs = np.array([spec_of(n) for n in self.body_names], float)
        self.peak, self.top_speed = specs[:, 0], specs[:, 1]
        self.rated = spec.rated_frac * self.peak
        self.kp0, self.kv0 = model.actuator_gainprm[self.ctrl, 0].copy(), -model.actuator_biasprm[self.ctrl, 2]
        hand = [b for b in range(model.nbody) if model.body(b).name.startswith(ids.hand.prefix)]
        self.hand_dofs = np.array([v for v in range(model.nv) if model.dof_bodyid[v] in hand or
                                   "_hand_" in model.body(model.dof_bodyid[v]).name])
        self.nominal = {"mass": model.body_mass.copy(), "inertia": model.body_inertia.copy(),
                        "friction": model.geom_friction.copy(), "frictionloss": model.dof_frictionloss.copy()}
        self.hand_bodies = hand
        self.temp = np.full(len(self.joints), 30.0)
        self.torque_scale, self.ambient = 1.0, 25.0
        self.fault = None

    # ------------------------------------------------------------------ a new robot and room
    def sample_world(self, rng: np.random.Generator) -> None:
        m, s, n, ids = self.m, self.spec, self.nominal, self.ids
        u = lambda r: float(rng.uniform(*r))
        m.body_mass[:] = n["mass"]
        m.body_inertia[:] = n["inertia"]
        m.geom_friction[:] = n["friction"]
        m.dof_frictionloss[:] = n["frictionloss"]
        for bodies, factor in ((self.hand_bodies, u(s.hand_mass)), (ids.cup_bodies, u(s.cup_mass)),
                               ([ids.cap_body], u(s.cap_mass))):
            m.body_mass[bodies] *= factor
            m.body_inertia[bodies] *= factor
        grip, objects = u(s.grip_friction), u(s.object_friction)
        m.geom_friction[list(ids.hand_geoms)] *= grip
        for geoms in ids.cup_geoms:
            m.geom_friction[list(geoms)] *= objects
        m.geom_friction[ids.cap_geom] *= objects
        m.dof_frictionloss[ids.arm_dofs] = rng.uniform(*s.joint_friction, len(ids.arm_dofs))
        kp = self.kp0 * rng.uniform(*s.kp, len(self.ctrl))
        kv = self.kv0 * np.sqrt(kp / self.kp0)
        m.actuator_gainprm[self.ctrl, 0] = kp
        m.actuator_biasprm[self.ctrl, 1] = -kp
        m.actuator_biasprm[self.ctrl, 2] = -kv
        self.kp = kp
        mujoco.mj_setConst(m, self.d)
        self.torque_scale = u(s.torque)
        self.ambient = u(s.ambient)
        self.temp = self.ambient + rng.uniform(*s.start_temp, len(self.joints))
        self.action_delay = int(rng.integers(s.action_delay[0], s.action_delay[1] + 1))
        self.camera_delay = int(rng.integers(s.camera_delay[0], s.camera_delay[1] + 1))
        common = rng.normal(0, s.camera_bias, 3) * [1, 1, 0.5]
        self.object_bias = common + rng.normal(0, s.object_bias, (4, 3)) * [1, 1, 0.5]
        self.push_dir = rng.normal(0, s.push, 2)
        self.push_at = float(rng.uniform(5, 30))
        self.rng, self.seen, self.history, self.clock = rng, None, [], 0.0
        self.fault = None

    def stand_reset(self) -> None:
        d = self.d
        self.left_home = d.qpos[self.left_q].copy()
        self.left_ref = self.left_home
        self.leg_home = [d.qpos[q].copy() for q in self.leg_q]
        self.foot_pos = [d.site_xpos[s].copy() for s in self.feet]
        self.com_ref = (self.foot_pos[0][:2] + self.foot_pos[1][:2]) / 2 + [0.03, 0]
        self.leg_ref = [q.copy() for q in self.leg_home]
        self.pelvis_shift = np.zeros(2)
        self.leg_map = self._plan_shift()

    # ------------------------------------------------------------------ the camera
    def camera(self, truth: np.ndarray, hand: np.ndarray, hand_mat: np.ndarray, closed: bool, active: int) -> np.ndarray:
        """What the robot believes about the 4 objects (rows of world positions): the camera's last frame, which is
        late, and for an object in the closed hand moved on by what the hand has done since (encoders are not late)."""
        s = self.spec
        self.history = (self.history + [(truth.copy(), hand.copy(), hand_mat.copy())])[-(self.camera_delay + 1):]
        if self.seen is None or self.rng.random() > s.camera_dropout:
            frame, self.seen_hand, self.seen_mat = self.history[0]
            self.seen = frame + self.object_bias + self.rng.normal(0, s.camera_jitter, truth.shape)
        out = self.seen.copy()
        if closed and np.linalg.norm(out[active] - self.seen_hand) < s.held_reach:
            # (Only this one: the others stand still, however close. The hand's turn counts too, for a cup held off-centre.)
            out[active] = hand + hand_mat @ self.seen_mat.T @ (out[active] - self.seen_hand)
        return out

    # ------------------------------------------------------------------ balance
    def _stance(self, dt: float) -> None:
        """Stiff legs in a fixed pose stand like a statue while the COM is over the feet. Only near the edge of the feet
        does the pelvis shift, slowly, along a planned leg motion. Targets are never built from measured joints: that
        feeds every disturbance back into the command, and the robot sways more and more until it falls."""
        com = self.d.subtree_com[self.pelvis][:2]
        com_err = np.clip(com, self.com_ref - self.spec.com_box, self.com_ref + self.spec.com_box) - com
        self.pelvis_shift = np.clip(self.pelvis_shift + self.spec.com_gain * com_err * dt, -0.05, 0.05)
        for i in range(2):
            self.leg_ref[i] = self.leg_home[i] + self.leg_map[i] @ self.pelvis_shift

    def _plan_shift(self) -> list:
        """For each leg, the joint motion per metre of pelvis shift in x, y with the feet staying put."""
        d, m = self.d, self.m
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        out = []
        for i in range(2):
            mujoco.mj_jacSite(m, d, jp, jr, self.feet[i])
            J = np.vstack([jp, jr])[:, self.leg_v[i]]
            out.append(np.linalg.solve(J, -np.eye(6)[:, :2]))
        return out

    def _feed_forward(self) -> np.ndarray:
        """Joint torques that hold the robot up: inverse dynamics with the weight shared by the two feet."""
        d, m = self.d, self.m
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        rows = []
        for s in self.feet:
            mujoco.mj_jacSite(m, d, jp, jr, s)
            rows.append(np.vstack([jp, jr]))
        J = np.vstack(rows)
        # Shift weight between the feet rather than lean on the ankles: a foot's moment costs more than its force.
        weight = np.tile([1, 1, 1, 0.05, 0.05, 0.05], 2)
        gravity = self._gravity()
        # A standing robot is an inverted pendulum, which stiff joints alone hold only just: push on the floor against
        # the COM's lean and its speed (on the robot these come from the IMU and the joint angles).
        com = d.subtree_com[self.pelvis]
        vel = d.subtree_linvel[self.pelvis]
        s = self.spec
        force = np.zeros(3)
        force[:2] = -self.mass * (s.lean_rate ** 2 * (com[:2] - self.com_ref) + 2 * s.lean_rate * vel[:2])
        catch = np.concatenate([force, np.cross(com - d.xpos[self.pelvis], force)])
        catch[3:] = d.xmat[self.pelvis].reshape(3, 3).T @ catch[3:]  # (the free joint's moments are in the body frame)
        wrench = weight * np.linalg.lstsq(J[:, :6].T * weight, gravity[:6] + catch, rcond=None)[0]
        return gravity - J.T @ wrench

    def _gravity(self) -> np.ndarray:
        """Joint forces of gravity alone. Velocity terms are left out: if a foot leaves the floor they make the legs
        push harder, and feed the bounce."""
        s, m = self.scratch, self.m
        s.qpos[:] = self.d.qpos
        mujoco.mj_kinematics(m, s)
        mujoco.mj_comPos(m, s)
        out = np.zeros(m.nv)
        mujoco.mj_rne(m, s, 0, out)
        return out

    def tick(self, dt: float) -> None:
        """Run after the arm controller has written its targets. dt: time to the next tick."""
        d, m = self.d, self.m
        self._stance(dt)
        for i in range(2):
            d.ctrl[self.leg_ctrl[i]] = self.leg_ref[i]
        d.ctrl[self.left_ctrl] = self.left_ref
        tau = self._feed_forward()
        d.ctrl[self.ctrl] += tau[self.v_adr] / self.kp
        d.qfrc_applied[self.hand_dofs] = tau[self.hand_dofs]  # the fingers' own motors carry the fingers
        self.clock += dt
        pushing = self.push_at < self.clock < self.push_at + 0.5
        d.xfrc_applied[self.ids_torso, :2] = self.push_dir if pushing else 0.0

        speed = np.abs(d.qvel[self.v_adr])
        cap = self.peak * self.torque_scale
        cap = cap * np.clip((self.spec.shut_down - self.temp) / (self.spec.shut_down - self.spec.derate_from), 0, 1)
        cap = cap * np.clip(2 * (1 - speed / self.top_speed), 0.05, 1)  # back-EMF: full torque up to half speed
        m.jnt_actfrcrange[self.joints, 0], m.jnt_actfrcrange[self.joints, 1] = -cap, cap
        torque = np.abs(d.qfrc_actuator[self.v_adr])
        s = self.spec
        self.temp += dt * ((torque / self.rated) ** 2 * s.rated_rise - (self.temp - self.ambient)) / s.heat_tc
        if self.temp.max() >= s.shut_down and self.fault is None:
            self.fault = f"motor overheated: {self.body_names[int(self.temp.argmax())]}"

    @property
    def ids_torso(self) -> int:
        return self.m.body("torso_link").id

    @property
    def arm_temps(self) -> np.ndarray:
        return self.temp[self.arm_slice]

    @property
    def fallen(self) -> bool:
        return bool(self.d.xpos[self.pelvis][2] < PELVIS_HEIGHT - 0.1 or self.d.xmat[self.pelvis][8] < 0.9)
