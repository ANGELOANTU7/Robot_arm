"""Build the MuJoCo scene: Menagerie Panda + ball + cup + cameras."""

import math

import mujoco
import numpy as np

from robot_arm.assets import ensure_panda
from robot_arm.config import CFG, HOME_QPOS, TaskConfig

# Distance from the Panda "hand" body origin to the point between the fingertip pads.
TCP_OFFSET = 0.1034


def look_at(pos, target) -> list[float]:
    """MuJoCo camera `xyaxes` for a camera at `pos` looking at `target` with world z up."""
    forward = np.asarray(target, float) - np.asarray(pos, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return [*right, *up]


def build_spec(cfg: TaskConfig = CFG) -> mujoco.MjSpec:
    scene_xml = ensure_panda().parent / "scene.xml"
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    spec.option.timestep = cfg.sim_dt
    # Elliptic friction cones with a high impratio stop the held ball from slowly creeping out of the grasp.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    # Gravity compensation on the arm (as on the real robot), so the TCP does not sag below its target.
    for name in [f"link{i}" for i in range(1, 8)] + ["hand", "left_finger", "right_finger"]:
        spec.body(name).gravcomp = 1.0

    # Collision bits: the ball (bit 2) touches the floor, the cup, the hand and the fingertip pads (bits 1|2),
    # but not the finger meshes, whose slanted faces would squeeze it out of the grasp.
    for name in ("left_finger", "right_finger"):
        for g in spec.body(name).geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_BOX:
                g.contype = g.conaffinity = 3
    for g in spec.body("hand").geoms:
        if g.contype:
            g.contype = g.conaffinity = 3
    spec.geom("floor").contype = spec.geom("floor").conaffinity = 3

    # Tool centre point between the fingertips.
    spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.004, 0, 0],
                               rgba=[1, 0, 0, 0.5], group=4)

    world = spec.worldbody
    spec.material("groundplane").reflectance = 0.05

    # Ball: free body, high friction so the parallel gripper can hold it.
    ball = world.add_body(name="ball", pos=[0.5, -0.15, cfg.ball_radius])
    ball.add_freejoint(name="ball_joint")
    ball.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[cfg.ball_radius, 0, 0],
                  mass=cfg.ball_mass, rgba=[1.0, 0.45, 0.05, 1], condim=4,
                  friction=[1.5, 0.02, 0.002], solref=[0.004, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2],
                  contype=2, conaffinity=2)

    # Cup: mocap body (moved at reset, never by contacts), bottom disc + ring of wall boxes.
    cup = world.add_body(name="cup", pos=[0.5, 0.15, 0], mocap=True)
    r_in, wall, h = cfg.cup_inner_radius, cfg.cup_wall, cfg.cup_height
    cup_rgba = [0.15, 0.35, 0.9, 1]
    cup.add_geom(name="cup_bottom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[r_in + wall, 0.003, 0], pos=[0, 0, 0.003], rgba=cup_rgba, contype=3, conaffinity=3)
    n = cfg.cup_segments
    r_mid = r_in + wall / 2
    half_len = r_mid * math.tan(math.pi / n) + 1e-3  # small overlap so there are no gaps
    for i in range(n):
        a = 2 * math.pi * i / n
        cup.add_geom(name=f"cup_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[wall / 2, half_len, h / 2],
                     pos=[r_mid * math.cos(a), r_mid * math.sin(a), h / 2],
                     euler=[0, 0, a], rgba=cup_rgba, contype=3, conaffinity=3)

    # Cameras for rendering rollouts, all looking at the middle of the workspace.
    look = [0.5, 0.0, 0.12]
    for name, pos in (("front", [1.55, 0.75, 0.95]), ("side", [0.45, -1.35, 0.7])):
        world.add_camera(name=name, pos=pos, xyaxes=look_at(pos, look), fovy=45)
    world.add_camera(name="top", pos=[0.5, 0.0, 1.3], xyaxes=[0, -1, 0, 1, 0, 0], fovy=50)

    # Home keyframe now also needs the ball's free joint (7 numbers) and the cup mocap pose.
    key = spec.key("home")
    key.qpos = list(HOME_QPOS) + [0.04, 0.04] + [0.5, -0.15, cfg.ball_radius, 1, 0, 0, 0]
    key.ctrl = list(HOME_QPOS) + [255]
    key.mpos = [0.5, 0.15, 0]
    key.mquat = [1, 0, 0, 0]
    return spec


def build_model(cfg: TaskConfig = CFG) -> mujoco.MjModel:
    return build_spec(cfg).compile()


class SceneIds:
    """Indices of the objects the env and controller touch every step."""

    def __init__(self, model: mujoco.MjModel):
        self.arm_joints = np.array([model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_dofs = np.array([model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.finger_qpos = np.array([model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)])
        self.tcp_site = model.site("tcp").id
        self.hand_body = model.body("hand").id
        self.ball_body = model.body("ball").id
        self.ball_qpos = model.joint("ball_joint").qposadr[0]
        self.ball_dof = model.joint("ball_joint").dofadr[0]
        self.ball_geom = model.geom("ball_geom").id
        self.cup_mocap = model.body("cup").mocapid[0]
        self.left_finger_geoms = {g for g in range(model.ngeom)
                                  if model.geom_bodyid[g] == model.body("left_finger").id}
        self.right_finger_geoms = {g for g in range(model.ngeom)
                                   if model.geom_bodyid[g] == model.body("right_finger").id}
        self.finger_geoms = self.left_finger_geoms | self.right_finger_geoms
        self.home_key = model.key("home").id
