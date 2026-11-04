"""MuJoCo scene: Unitree G1 (pelvis fixed) at a table with three party cups and a bottle cap. The right hand is the
Dex3 or a five-fingered Shadow hand (see hands.py)."""

import math

import mujoco
import numpy as np

from robot_arm.assets import ensure_g1, ensure_model
from robot_arm.humanoid.config import G1_CFG, CupSpec, G1TaskConfig
from robot_arm.humanoid.hands import DEX3, SHADOW, Hand, get_hand
from robot_arm.scene import look_at

# Joints the right-hand skills move; everything else holds the stand posture.
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
RIGHT_ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
             "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
RIGHT_ARM_COLLIDERS = ["right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link"]

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

WRIST_CAM_POS = (0.0, 0.0, 0.08)  # in the wrist frame, above the back of the hand
WRIST_CAM_XYAXES = (0.27, -0.963, 0.0, 0.217, 0.061, 0.974)  # looking past the fingers at the grasp
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


def replace_right_hand(spec: mujoco.MjSpec, hand: Hand) -> None:
    """Swap the right Dex3 (its palm is part of the wrist link; the fingers are child bodies) for `hand`, a
    Menagerie hand whose root body is mounted on the wrist link."""
    for a in list(spec.actuators):
        if a.name.startswith("right_hand_"):
            spec.delete(a)
    wrist = spec.body("right_wrist_yaw_link")
    for name in ("right_hand_thumb_0_link", "right_hand_middle_0_link", "right_hand_index_0_link"):
        spec.delete(spec.body(name))
    for g in list(wrist.geoms):
        if g.meshname == "right_hand_palm_link":
            spec.delete(g)
    child = mujoco.MjSpec.from_file(str(ensure_model(*hand.menagerie)))
    root = child.body(hand.root_body)
    # The hand's own wrist joints are dropped: the G1's wrist does that job.
    for j in list(root.joints):
        for a in list(child.actuators):
            if a.target == j.name:
                child.delete(a)
        child.delete(j)
    for a in list(child.actuators):  # actuators of joints above the root (the hand's forearm and wrist)
        if a.target in ("rh_WRJ1", "rh_WRJ2"):
            child.delete(a)
    frame = wrist.add_frame(pos=list(hand.mount_pos), quat=list(hand.mount_quat))
    frame.attach_body(root, hand.prefix, "")


def build_spec(cfg: G1TaskConfig = G1_CFG) -> mujoco.MjSpec:
    scene_xml = ensure_g1().parent / "scene_with_hands.xml"
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    spec.option.timestep = cfg.sim_dt
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    # Fixed pelvis: remove the floating base (the pelvis body keeps its stand height of 0.793 m). G1_REAL keeps it.
    if not cfg.real:
        spec.delete(spec.joint("floating_base_joint"))
    for key in list(spec.keys):
        spec.delete(key)
    hand = get_hand(cfg.hand)
    if hand.menagerie is not None:
        replace_right_hand(spec, hand)

    right_colliders = set(RIGHT_ARM_COLLIDERS) | set(hand.colliders)
    for body in spec.bodies:
        if body.name == "world":
            continue
        body.gravcomp = 0.0 if cfg.real else 1.0  # like the real robot's feed-forward gravity compensation
        for g in body.geoms:
            if g.contype == 0 and g.conaffinity == 0:
                continue  # visual geom
            if cfg.real and body.name.endswith("ankle_roll_link") and g.type == mujoco.mjtGeom.mjGEOM_SPHERE:
                g.contype, g.conaffinity = OBJ_TYPE, OBJ_AFF
            elif body.name in right_colliders:
                g.contype, g.conaffinity = HAND_TYPE, HAND_AFF
                # Soft finger pads: high sliding and torsional friction, so a held cup does not swing.
                g.friction = [1.5, 0.08, 0.002]
                g.condim = 4
            else:
                g.contype = g.conaffinity = 0
    spec.geom("floor").contype, spec.geom("floor").conaffinity = OBJ_TYPE, OBJ_AFF

    spec.body("right_wrist_yaw_link").add_site(name="grasp", pos=list(hand.grasp_site), size=[0.006, 0, 0],
                                               rgba=[1, 0, 0, 0.6], group=4)
    # The robot's own cameras (for the VLA): the head RGB camera where the G1's D435 sits (on the torso, pitched
    # 47.6 degrees down), and one on the wrist looking past the fingers. They move with the robot.
    th = 0.8308
    spec.body("torso_link").add_camera(name="head", pos=[0.0576, 0.0175, 0.4299], fovy=85,
                                       xyaxes=[0, -1, 0, np.sin(th), 0, np.cos(th)])
    spec.body("right_wrist_yaw_link").add_camera(name="wrist", pos=list(WRIST_CAM_POS), fovy=80,
                                                 xyaxes=list(WRIST_CAM_XYAXES))

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
    spec = build_spec(cfg)
    if not cfg.real:
        return spec.compile()
    from robot_arm.humanoid.real import HAND_KG, spec_of

    model = spec.compile()
    hand = get_hand(cfg.hand)
    hand_bodies = [b for b in range(model.nbody) if model.body(b).name.startswith(hand.prefix)]
    # The weight of a real hand sits in its palm and motors, not in the fingers.
    palm = hand_bodies[0]
    factor = 1 + (HAND_KG - model.body_mass[hand_bodies].sum()) / model.body_mass[palm]
    model.body_mass[palm] *= factor
    model.body_inertia[palm] *= factor
    for a in range(model.nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue  # (a tendon of the hand)
        joint = model.actuator_trnid[a, 0]
        name = model.joint(joint).name
        if name.startswith(hand.prefix) or "_hand_" in name:
            continue
        peak, _, kp, kd = spec_of(name)
        model.jnt_actfrcrange[joint] = -peak, peak
        model.actuator_gainprm[a, 0], model.actuator_biasprm[a, 1], model.actuator_biasprm[a, 2] = kp, -kp, -kd
        model.actuator_ctrllimited[a] = 0  # targets carry the gravity feed-forward, so they can pass the joint's range
    return model


class G1Ids:
    """Indices into qpos/qvel/ctrl for the joints and objects the task uses."""

    def __init__(self, model: mujoco.MjModel, hand: Hand | None = None):
        def jq(name):
            return model.joint(name).qposadr[0]

        def jd(name):
            return model.joint(name).dofadr[0]

        self.arm_names = WAIST + RIGHT_ARM
        self.arm_qpos = np.array([jq(n) for n in self.arm_names])
        self.arm_dofs = np.array([jd(n) for n in self.arm_names])
        self.arm_ctrl = np.array([model.actuator(n).id for n in self.arm_names])
        if hand is None:  # the one the model was built with
            hand = SHADOW if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SHADOW.colliders[0]) >= 0 else DEX3
        self.hand = hand
        self.hand_qpos = np.array([jq(n) for n in hand.obs_joints])
        self.hand_ctrl = np.array([model.actuator(n).id for n in hand.actuators])
        self.arm_range = model.jnt_range[[model.joint(n).id for n in self.arm_names]].copy()
        self.grasp_site = model.site("grasp").id
        self.hand_body = model.body("right_wrist_yaw_link").id
        self.cup_bodies = [model.body(n).id for n in CUP_NAMES]
        self.cup_qpos = [jq(f"{n}_joint") for n in CUP_NAMES]
        self.cup_geoms = [{g for g in range(model.ngeom) if model.geom_bodyid[g] == b} for b in self.cup_bodies]
        self.cap_body = model.body("cap").id
        self.cap_qpos = jq("cap_joint")
        self.cap_dof = jd("cap_joint")
        self.cap_geom = model.geom("cap_geom").id
        self.hand_geoms = {g for g in range(model.ngeom)
                           if model.geom_contype[g] == HAND_TYPE}
        self.stand_qpos = np.zeros(model.nq)
        for name, value in STAND.items():
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:  # (the Dex3's thumb, if fitted)
                self.stand_qpos[jq(name)] = value
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint") >= 0:
            from robot_arm.humanoid.real import PELVIS_HEIGHT, STAND_LEGS

            self.stand_qpos[jq("floating_base_joint"):jq("floating_base_joint") + 7] = 0, 0, PELVIS_HEIGHT, 1, 0, 0, 0
            for side in ("left", "right"):
                for name, value in STAND_LEGS.items():
                    self.stand_qpos[jq(f"{side}_{name}")] = value
        self.ready_qpos = self.stand_qpos.copy()
        for name, value in READY.items():
            self.ready_qpos[jq(name)] = value
