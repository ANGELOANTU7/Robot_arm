"""The G1's right hand: the Unitree Dex3 (three fingers) or a five-fingered Shadow Dexterous Hand.

Everything the task needs to know about a hand: which actuators drive it, which joints the policy sees, how it
is mounted on the wrist, where a held cup's axis sits, the poses the grip synergy blends between (wrap a cup,
pinch the cap, thumbs-up) and the geometry of the cap pinch and of the thumb. Vectors are in the
right_wrist_yaw_link frame, where both hands point their fingers along +x with the palm facing +y.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Hand:
    name: str
    actuators: tuple[str, ...]  # hand actuators, in the order of the pose vectors
    obs_joints: tuple[str, ...]  # 7 joints whose angles the policy sees (the same observation size for every hand)
    colliders: tuple[str, ...]  # hand bodies that touch the cups, the cap and the table
    grasp_site: tuple[float, float, float]  # where a held cup's axis sits, between the palm/fingers and the thumb
    poses: dict = field(default_factory=dict)  # synergy shape -> key actuator targets from open to closed
    pinch_point: np.ndarray = field(default_factory=lambda: np.zeros(3))  # where the cap sits between the tips
    pinch_down: np.ndarray = field(default_factory=lambda: np.zeros(3))  # must point down while pinching
    pinch_axis: np.ndarray = field(default_factory=lambda: np.zeros(3))  # thumb-to-finger axis (kept horizontal)
    throw_side: float = 1.0  # the cap is released on cup B's side of the pyramid (1) or on cup A's (-1)
    throw_long: float = 0.0  # the throw is aimed this far past the mouth: the fingers let go early, at 0.05 s a step
    pick_turn: float = 0.0  # the cap is picked up with the hand turned this far from its heading for the throw (rad)
    thumb_dir: np.ndarray = field(default_factory=lambda: np.zeros(3))  # the thumb's direction in the thumbs-up
    thumbs_up_roll: float = 0.0  # roll about the fingers from the side grasp that turns the thumb up
    expert: dict = field(default_factory=dict)  # PlaceExpert settings this hand needs (see scripted.py)
    # How the hand reaches a cup, as a direction in the expert's grasp frame (x: the heading the hand points along,
    # z: up, the cup's axis): the Dex3 slides in along its fingers; a hand that holds the cup in front of its palm
    # can come down onto it from above (-z).
    approach: tuple[float, float, float] = (1.0, 0.0, 0.0)
    # How high the grasp site passes over the top of a cup standing on the table while the hand is empty (the retargeted
    # paths, retarget.plan_place): the open hand's reach below the site, and some margin.
    clearance: float = 0.07
    # G1TaskConfig values this hand needs (config.task_config): where it holds the cups, where the pyramid goes.
    task: dict = field(default_factory=dict)
    # Where the hand and forearm go while picking a cup, relative to it: (behind, ahead) along the reach heading and
    # (right, left) of it. Nothing else may lie there when a cup is picked (objects.sample_layout).
    corridor: tuple[float, float, float, float] = (-0.24, 0.12, -0.06, 0.06)
    menagerie: tuple[str, str] | None = None  # (folder, xml) of a hand that replaces the Dex3
    mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # where its root body goes on the wrist
    mount_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    root_body: str = ""
    prefix: str = ""


DEX3_JOINTS = ("right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
               "right_hand_index_0_joint", "right_hand_index_1_joint",
               "right_hand_middle_0_joint", "right_hand_middle_1_joint")

DEX3 = Hand(
    name="dex3",
    actuators=DEX3_JOINTS,  # one position actuator per joint, named after it
    obs_joints=DEX3_JOINTS,
    colliders=("right_hand_thumb_0_link", "right_hand_thumb_1_link", "right_hand_thumb_2_link",
               "right_hand_index_0_link", "right_hand_index_1_link",
               "right_hand_middle_0_link", "right_hand_middle_1_link"),
    # The open fingers reach y = +0.009, so a cup of radius r (3.6 cm where the party cups are held) needs its
    # axis at y > 0.009 + r to slide in without being pushed; the closing fingers then wrap its far side.
    grasp_site=(0.11, 0.050, 0.0),
    poses={
        # (thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1)
        "wrap": (np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]),  # thumb swung back for a wide opening
                 np.array([0.0, -0.9, -1.2, 1.2, 1.3, 1.2, 1.3])),
        # Claw pinch for the bottle cap lying on the table: palm facing down, the index finger hooked down at its
        # base joint and the thumb meeting its tip; the middle finger is curled out of the way. Found by searching
        # joint poses for fingertips that are the lowest points of the hand (5 mm below everything else), so they
        # reach the rim of a 12 mm-tall cap without the knuckles touching the table, then by pick trials.
        # Open: tips ~4 cm apart; closed: squeezed together.
        "pinch": (np.array([-0.51, -0.34, -0.44, 1.33, 0.00, 1.57, 1.75]),
                  np.array([-0.454, -0.644, -0.116, 1.436, 0.059, 1.570, 1.750])),
        # The two fingers curled into a fist, the thumb straight out of the palm side.
        "thumbs_up": (np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]),
                      np.array([0.0, 0.4, 0.0, 1.4, 1.6, 1.4, 1.6])),
    },
    pinch_point=np.array([0.1275, 0.0942, 0.0258]),
    pinch_down=np.array([0.283, 0.937, 0.204]),
    pinch_axis=np.array([-0.939, 0.242, -0.246]),
    # From the thumb's middle joint to its tip: out of the palm side, so the hand rolls a quarter turn about its
    # fingers' axis from the side grasp to have the thumb pointing up.
    thumb_dir=np.array([-0.392, 0.92, -0.003]),
    thumbs_up_roll=np.pi / 2,
)


SHADOW_ACTUATORS = ("THJ5", "THJ4", "THJ3", "THJ2", "THJ1", "FFJ4", "FFJ3", "FFJ0", "MFJ4", "MFJ3", "MFJ0",
                    "RFJ4", "RFJ3", "RFJ0", "LFJ5", "LFJ4", "LFJ3", "LFJ0")
SHADOW_PREFIX = "sh_rh_"


# The middle, ring and little fingers while the thumb and index finger hold a cup: slightly bent, out of the way.
# Curled further they would reach into the cup below the grip (it widens towards its mouth).
SPARE = {f"{f}J{j}": 0.4 for f in ("MF", "RF", "LF") for j in (3, 0)}


def _shadow_pose(**joints) -> np.ndarray:
    """Actuator targets for the Shadow hand from named values (unnamed: 0). J0 drives a finger's two distal
    joints together (their sum)."""
    return np.array([float(joints.get(a, 0.0)) for a in SHADOW_ACTUATORS])


SHADOW = Hand(
    name="shadow",
    actuators=tuple(f"{SHADOW_PREFIX}A_{a}" for a in SHADOW_ACTUATORS),
    obs_joints=tuple(f"{SHADOW_PREFIX}{j}" for j in ("THJ5", "THJ4", "THJ2", "FFJ3", "FFJ2", "MFJ3", "MFJ2")),
    colliders=tuple(f"{SHADOW_PREFIX}{b}" for b in (
        "palm", "ffknuckle", "ffproximal", "ffmiddle", "ffdistal", "mfknuckle", "mfproximal", "mfmiddle",
        "mfdistal", "rfknuckle", "rfproximal", "rfmiddle", "rfdistal", "lfmetacarpal", "lfknuckle", "lfproximal",
        "lfmiddle", "lfdistal", "thbase", "thproximal", "thhub", "thmiddle", "thdistal")),
    # Two-finger cup grip, as in your clips: the thumb's and the index finger's pads on opposite sides of the cup,
    # the cup standing in front of the palm (7 cm out) with its axis along the thumb's side; the middle, ring and
    # little fingers stay slightly bent, out of the way (6 mm off the cup, which widens below the grip). Found with
    # scripts/g1_shadow_poses.py (cup --side): at a 6.4 cm gap the rest of the hand stays 8 mm off the cup's wall.
    # The hand comes down onto the cup from above, thumb up, the fingers along the heading and the open pads
    # straddling the cup's narrow end. The hand reaches 8.5 cm below the grip and 2.2 cm above it, so the cups are
    # held 2.8 cm from their narrow end: high enough over the table to pick a mouth-down cup, and, once the top cup
    # is turned over (a roll about the line between the two pads, which turns the hand thumb-down), high enough over
    # the base cups to put it down.
    grasp_site=(0.142, 0.073, 0.043),
    approach=(0.0, 0.0, -1.0),
    clearance=0.115,  # the open fingers reach 8.5 cm below the grip
    # The cups are held 2.8 cm from their narrow end (see above). The turned-over hand reaches the pyramid with the
    # most room from the arm's joint limits (0.25 rad) at (0.48, -0.16) with its fingers at +0.2 rad; it lets go of
    # the top cup with its fingers across the base cups' line (along it, the opening thumb lands on a base cup), so
    # the line runs at 0.2 - 90 degrees. The supply spots stay 16 cm clear of that pyramid.
    task={"grasp_height_frac": 0.77, "top_grasp_height_frac": 0.23, "pyramid_center": (0.48, -0.16),
          "pyramid_yaw": 0.2 - np.pi / 2, "supply_cups": ((0.30, -0.18), (0.30, -0.35), (0.34, 0.02)),
          "cap_spot": (0.255, -0.03), "cap_near_spot": True},
    corridor=(-0.20, 0.10, -0.12, 0.03),
    poses={
        # Open: pads 12 cm apart (3 cm room on each side of a cup); closed: 2.7 cm, a squeeze through a 6 cm cup.
        "wrap": (_shadow_pose(THJ5=-0.229, THJ4=1.178, THJ3=-0.019, THJ2=-0.37, THJ1=0.971, FFJ4=-0.212, FFJ3=0.356,
                              FFJ0=0.089, **SPARE),
                 _shadow_pose(THJ5=0.221, THJ4=1.178, THJ3=-0.019, THJ2=-0.37, THJ1=1.571, FFJ4=-0.212, FFJ3=0.806,
                              FFJ0=1.389, **SPARE)),
        # Cap pinch, as in your throw clips: the thumb's and the index finger's tips meet across the cap, palm down, the
        # other three fingers straight. Found with scripts/g1_shadow_poses.py (pinch): the two tips are the lowest
        # points of the hand, 4 mm below everything else (the search asked for 5), so they reach the cap's sides
        # with 3 mm between the knuckles and the cap's top. Open: tips 44 mm apart; closed: 20 mm (a squeeze through
        # the 28 mm cap), centred within 3 mm of each other.
        "pinch": (_shadow_pose(THJ5=0.623, THJ4=0.801, THJ3=0.182, THJ2=0.048, THJ1=-0.175, FFJ4=0.003, FFJ3=0.45,
                               FFJ0=0.919),
                  _shadow_pose(THJ5=0.823, THJ4=0.801, THJ3=0.182, THJ2=0.048, THJ1=-0.175, FFJ4=0.003, FFJ3=0.25,
                               FFJ0=1.719)),
        # Thumbs-up: the four fingers curl into a fist, the thumb stands up from it (THJ4 0.6, THJ5 -0.5).
        "thumbs_up": (_shadow_pose(THJ4=0.6, THJ5=-0.5),
                      _shadow_pose(THJ4=0.6, THJ5=-0.5, **{f"{f}J{j}": v for f in ("FF", "MF", "RF", "LF")
                                                           for j, v in ((3, 1.57), (0, 2.6))})),
    },
    # From the thumb's middle joint to its tip in the thumbs-up; rolled 0.5 rad about the fingers from the upright
    # grip it points 95% straight up (the check asks for 82%).
    thumb_dir=np.array([0.306, 0.458, 0.834]),
    thumbs_up_roll=0.502,
    pinch_point=np.array([0.1646, 0.0595, 0.0388]),
    pinch_down=np.array([0.263, 0.462, 0.847]),
    pinch_axis=np.array([0.964, -0.164, -0.210]),
    pick_turn=0.8,  # the throw's heading leaves the arm at its joint limits over the near cap spots
    throw_long=0.03,
    throw_side=-1.0,  # thrown from cup A's side: the pinch's heading from B's side is out of this arm's reach
    # Cups are put down with the fingers pointing forward (the hand on a base cup's right, away from cup A), the
    # top cup with them across the base cups' line; a placed top cup is left upwards and out along its sloping wall
    # (rising straight up, the little finger catches the cup where it widens). The cup is turned over rolling the
    # hand one way round: the other way the shoulder runs into its joint limit (8/8 vs 0/8 placed).
    # Heading 1.3 rad, not pi/2: with the elbow kept off the chest, pi/2 runs two wrist joints into their limits.
    expert={"place_yaw": 0.0, "top_yaw_rel": 1.3, "pre_dist": 0.10, "flip_taper_sign": 0.0, "flip_offset": 0.0,
            "flip_exit_out": 2.5, "flip_dir": 1.0},
    menagerie=("shadow_hand", "right_hand.xml"),
    # The Shadow palm's own frame has the fingers along +z, the palm facing -y and the thumb on the +x side:
    # turned half a turn about (x + z) it points its fingers along the wrist's +x with the palm facing +y and the
    # thumb up, like the Dex3 (and like your right hand held out for a handshake).
    mount_pos=(0.005, 0.0, 0.0),
    mount_quat=(0.0, 0.70710678, 0.0, 0.70710678),
    root_body="rh_palm",
    prefix="sh_",
)


HANDS = {"dex3": DEX3, "shadow": SHADOW}


def get_hand(name: str) -> Hand:
    try:
        return HANDS[name]
    except KeyError:
        raise ValueError(f"Unknown hand {name!r}: use one of {sorted(HANDS)}") from None
