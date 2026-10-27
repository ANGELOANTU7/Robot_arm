"""Palm pose and finger closure from MediaPipe hand landmarks (for retargeting onto a dexterous hand).

MediaPipe's world landmarks are metric and camera-aligned (checked on real photos: an identity rotation
reprojects them best), so the palm frame computed from them is the palm orientation in the camera frame.
"""

import numpy as np

from robot_arm.video.lift3d import hand_translation

WRIST, THUMB_CMC, THUMB_TIP = 0, 1, 4
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17
FINGERS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]  # MCP, PIP, DIP, TIP
PALM_POINTS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def palm_frame(world_m: np.ndarray, right_hand: bool = True) -> np.ndarray:
    """3x3 rotation [x y z]: x from the wrist towards the middle knuckle (finger direction), z out of the palm
    (the side the fingers curl towards), y = z x x (for a right hand that points away from the thumb)."""
    x = world_m[MIDDLE_MCP] - world_m[WRIST]
    x /= np.linalg.norm(x)
    across = world_m[INDEX_MCP] - world_m[PINKY_MCP]  # towards the thumb side
    z = np.cross(across, x) if right_hand else np.cross(x, across)
    z -= np.dot(z, x) * x
    z /= np.linalg.norm(z)
    return np.column_stack([x, np.cross(z, x), z])


def palm_center_cam(image_px: np.ndarray, world_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Centre of the palm (mean of wrist and knuckles) in the camera frame."""
    T = hand_translation(image_px, world_m, K)
    return world_m[PALM_POINTS].mean(axis=0) + T


def finger_curl(world_m: np.ndarray) -> float:
    """0 for a flat open hand, ~1 for a fist: mean bend of the four fingers (radians / pi)."""
    bends = []
    for mcp, pip, dip, tip in FINGERS:
        a = world_m[pip] - world_m[mcp]
        b = world_m[tip] - world_m[pip]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        bends.append(np.arccos(np.clip(cos, -1, 1)))
    return float(np.mean(bends) / np.pi)


def closure(world_m: np.ndarray) -> float:
    """Grasp closure in [0, 1] combining finger curl and how close the thumb tip is to the index knuckle."""
    hand_size = np.linalg.norm(world_m[MIDDLE_MCP] - world_m[WRIST])
    thumb = np.linalg.norm(world_m[THUMB_TIP] - world_m[INDEX_MCP]) / hand_size  # ~1.2 open, ~0.4 closed
    thumb_closure = np.clip((1.2 - thumb) / 0.8, 0, 1)
    curl = np.clip(finger_curl(world_m) / 0.6, 0, 1)
    return float(0.5 * thumb_closure + 0.5 * curl)
