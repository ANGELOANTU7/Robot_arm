"""MuJoCo scene: Unitree G1 (Dex3 hands, pelvis fixed) at a table with three party cups and a bottle cap."""

import math

import mujoco
import numpy as np

from robot_arm.assets import ensure_g1
from robot_arm.humanoid.config import G1_CFG, CupSpec, G1TaskConfig
from robot_arm.scene import look_at

# Joints the right-hand skills move; everything else holds the stand posture.
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
RIGHT_ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
             "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
RIGHT_HAND = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
              "right_hand_index_0_joint", "right_hand_index_1_joint",
              "right_hand_middle_0_joint", "right_hand_middle_1_joint"]
RIGHT_COLLIDERS = ["right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
                   "right_hand_thumb_0_link", "right_hand_thumb_1_link", "right_hand_thumb_2_link",
                   "right_hand_index_0_link", "right_hand_index_1_link",
                   "right_hand_middle_0_link", "right_hand_middle_1_link"]

# Stand posture (Menagerie "stand" keyframe for the arms); unlisted joints are 0.
STAND = {
    "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 1.28,
    "left_hand_thumb_1_joint": 1.05,
    "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2, "right_elbow_joint": 1.28,
    "right_hand_thumb_1_joint": -1.05,
}

# Ready posture of the waist + right arm (grasp site ~22 cm above the table in front of the right shoulder,
# hand pointing forward): episodes start here and the IK's null space pulls towards it.
READY = {
    "right_shoulder_pitch_joint": 0.462, "right_shoulder_roll_joint": -1.076, "right_shoulder_yaw_joint": -0.036,
    "right_elbow_joint": -0.469, "right_wrist_roll_joint": 0.989, "right_wrist_pitch_joint": -0.174,
    "right_wrist_yaw_joint": -0.271,
}

# Where a held cup's axis sits, between the palm/fingers and the thumb, in the right_wrist_yaw_link frame.
# The open fingers reach y = +0.009, so a cup of radius r (3.6 cm where the party cups are held) needs its
# axis at y > 0.009 + r to slide in without being pushed; the closing fingers then wrap its far side.
GRASP_SITE_POS = (0.11, 0.050, 0.0)

CUP_NAMES = ("cup0", "cup1", "cup2")  # two base cups (given mouth-down), the top cup (given mouth-up)

# Collision bits: objects and the table touch everything; the right arm/hand touches objects and the table only.
OBJ_TYPE, OBJ_AFF = 1, 3
HAND_TYPE, HAND_AFF = 2, 1


def add_cup(world: mujoco.MjsBody, name: str, cup: CupSpec, rgba) -> mujoco.MjsBody:
    """Tapered open cup: bottom disc + a ring of tilted wall boxes. Body origin at the centre of the bottom,
    body z axis pointing out of the mouth."""
    body = world.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    n, h, w = cup.segments, cup.height, cup.wall
    rb, rt = cup.bottom_radius, cup.mouth_radius
    tilt = math.atan2(rt - rb, h)
    slant = math.hypot(h, rt - rb)
    r_mid = (rb + rt) / 2 + w / 2
    half_len = (rt + w) * math.tan(math.pi / n) + 1e-3  # wide enough to close the ring at the mouth
    geoms = [body.add_geom(name=f"{name}_bottom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[rb + w, w / 2, 0], pos=[0, 0, w / 2])]
    for i in range(n):
        a = 2 * math.pi * i / n
        # Rotate about z to face outwards, then tilt the top outwards about the box's tangential (y) axis.
        q_yaw = np.array([math.cos(a / 2), 0, 0, math.sin(a / 2)])
        q_tilt = np.array([math.cos(tilt / 2), 0, math.sin(tilt / 2), 0])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, q_yaw, q_tilt)
        geoms.append(body.add_geom(name=f"{name}_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                   size=[w / 2, half_len, slant / 2],
                                   pos=[r_mid * math.cos(a), r_mid * math.sin(a), h / 2], quat=quat))
    mass_each = cup.mass / len(geoms)
    for g in geoms:
        g.mass = mass_each
        g.rgba = rgba
        g.contype, g.conaffinity = OBJ_TYPE, OBJ_AFF
        g.friction = [1.0, 0.05, 0.001]
        g.condim = 4
    body.add_site(name=f"{name}_center", pos=[0, 0, h / 2], size=[0.004, 0, 0], group=4)
    return body


def build_spec(cfg: G1TaskConfig = G1_CFG) -> mujoco.MjSpec:
    scene_xml = ensure_g1().parent / "scene_with_hands.xml"
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    spec.option.timestep = cfg.sim_dt
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    # Fixed pelvis: remove the floating base (the pelvis body keeps its stand height of 0.793 m).
    spec.delete(spec.joint("floating_base_joint"))
    for key in list(spec.keys):
        spec.delete(key)

    right_colliders = set(RIGHT_COLLIDERS)
    for body in spec.bodies:
        if body.name == "world":
            continue
        body.gravcomp = 1.0  # like the real robot's feed-forward gravity compensation
        for g in body.geoms:
            if g.contype == 0 and g.conaffinity == 0:
                continue  # visual geom
            if body.name in right_colliders:
                g.contype, g.conaffinity = HAND_TYPE, HAND_AFF
                # Soft finger pads: high sliding and torsional friction, so a held cup does not swing.
                g.friction = [1.5, 0.08, 0.002]
                g.condim = 4
            else:
                g.contype = g.conaffinity = 0
    spec.geom("floor").contype, spec.geom("floor").conaffinity = OBJ_TYPE, OBJ_AFF

    spec.body("right_wrist_yaw_link").add_site(name="grasp", pos=list(GRASP_SITE_POS), size=[0.006, 0, 0],
                                               rgba=[1, 0, 0, 0.6], group=4)

    world = spec.worldbody
    cx, cy = cfg.table_center
    hx, hy = cfg.table_half
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, cfg.table_height / 2],
                   pos=[cx, cy, cfg.table_height / 2], rgba=[0.55, 0.42, 0.3, 1],
                   contype=OBJ_TYPE, conaffinity=OBJ_AFF, friction=[0.8, 0.01, 0.001])

    for name in CUP_NAMES:
        add_cup(world, name, cfg.cup, [0.8, 0.08, 0.08, 1])  # red party cups

    cap = world.add_body(name="cap")
    cap.add_freejoint(name="cap_joint")
    cap.add_geom(name="cap_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[cfg.cap_radius, cfg.cap_half_height, 0],
                 mass=cfg.cap_mass, rgba=[0.1, 0.3, 0.85, 1], condim=4, friction=[1.0, 0.01, 0.001],
                 solref=[0.005, 0.6], contype=OBJ_TYPE, conaffinity=OBJ_AFF)

    look = [0.34, -0.12, cfg.table_height + 0.08]
    for name, pos in (("front", [1.25, -0.35, 1.25]), ("side", [0.35, -1.05, 1.05]),
                      ("over", [0.75, -0.05, 1.55])):
        world.add_camera(name=name, pos=pos, xyaxes=look_at(pos, look), fovy=45)
    return spec


def build_model(cfg: G1TaskConfig = G1_CFG) -> mujoco.MjModel:
    return build_spec(cfg).compile()


class G1Ids:
    """Indices into qpos/qvel/ctrl for the joints and objects the task uses."""

    def __init__(self, model: mujoco.MjModel):
        def jq(name):
            return model.joint(name).qposadr[0]

        def jd(name):
            return model.joint(name).dofadr[0]

        self.arm_names = WAIST + RIGHT_ARM
        self.arm_qpos = np.array([jq(n) for n in self.arm_names])
        self.arm_dofs = np.array([jd(n) for n in self.arm_names])
        self.arm_ctrl = np.array([model.actuator(n).id for n in self.arm_names])
        self.hand_qpos = np.array([jq(n) for n in RIGHT_HAND])
        self.hand_ctrl = np.array([model.actuator(n).id for n in RIGHT_HAND])
        self.arm_range = model.jnt_range[[model.joint(n).id for n in self.arm_names]].copy()
        self.grasp_site = model.site("grasp").id
        self.hand_body = model.body("right_wrist_yaw_link").id
        self.cup_bodies = [model.body(n).id for n in CUP_NAMES]
        self.cup_qpos = [jq(f"{n}_joint") for n in CUP_NAMES]
        self.cup_dofs = [jd(f"{n}_joint") for n in CUP_NAMES]
        self.cup_geoms = [{g for g in range(model.ngeom) if model.geom_bodyid[g] == b} for b in self.cup_bodies]
        self.cap_body = model.body("cap").id
        self.cap_qpos = jq("cap_joint")
        self.cap_dof = jd("cap_joint")
        self.cap_geom = model.geom("cap_geom").id
        self.hand_geoms = {g for g in range(model.ngeom)
                           if model.geom_contype[g] == HAND_TYPE}
        self.table_geom = model.geom("table").id
        self.stand_qpos = np.zeros(model.nq)
        for name, value in STAND.items():
            self.stand_qpos[jq(name)] = value
        self.ready_qpos = self.stand_qpos.copy()
        for name, value in READY.items():
            self.ready_qpos[jq(name)] = value
