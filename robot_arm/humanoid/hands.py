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
    thumb_dir: np.ndarray = field(default_factory=lambda: np.zeros(3))  # the thumb's direction in the thumbs-up
    thumbs_up_roll: float = 0.0  # roll about the fingers from the side grasp that turns the thumb up
    expert: dict = field(default_factory=dict)  # PlaceExpert settings this hand needs (see scripted.py)
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
    # The cup's axis sits beside the knuckles, 1.2 cm above the wrist axis: lower, the thumb (which arches above its
    # base) lands on the cup's top instead of behind it.
    grasp_site=(0.125, 0.054, -0.012),
    poses={
        # Power grasp: the four fingers wrap the cup's far side, the thumb presses its back. The thumb starts
        # pre-shaped behind where the cup ends up (clear of it while the hand slides in), so closing only moves it
        # 2 cm. Finger targets are 0.08 rad (J3) / 0.12 rad (J0) past touching the cup and the thumb's goes into it:
        # the Shadow's thumb servos are weak, and a harder finger squeeze shoves the cup 3 cm back past the thumb.
        "wrap": (_shadow_pose(THJ5=-0.693, THJ4=1.22, THJ2=0.69, THJ1=1.341),
                 _shadow_pose(FFJ3=1.05, FFJ0=1.35, MFJ3=1.05, MFJ0=1.28, RFJ3=0.83, RFJ0=1.36, LFJ3=0.53, LFJ0=1.37,
                              THJ5=-0.2, THJ4=1.22, THJ2=0.69, THJ1=1.57)),
        # Not designed yet: "pinch" (the cap) and "thumbs_up". See docs/HANDOFF_FIVE_FINGER.md.
    },
    # Cup B is put down with the palm facing cup A (the fingers and thumb wrap the side away from it), the top cup
    # with the fingers across the base cups' line so the thumb, under the cup once it is turned over, hangs in the
    # gap between them. A placed top cup is left by sliding the hand back along its fingers.
    expert={"place_yaw": -0.785, "top_place_yaw": -0.785, "flip_taper_sign": 0.0, "flip_exit_out": None,
            "flip_exit_rise": 0.0},
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
