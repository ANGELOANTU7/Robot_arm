"""Monocular 3D hand position from MediaPipe landmarks.

MediaPipe gives two things per frame: pixel landmarks, and metric landmarks centred on the hand (camera-aligned
axes, but no distance from the camera). If the hand sits at translation T in the camera frame, each landmark
X_i + T must project onto pixel (u_i, v_i):

    fx (X_i + Tx) = (u_i - cx) (Z_i + Tz)      fy (Y_i + Ty) = (v_i - cy) (Z_i + Tz)

which is linear in T, so the 21 landmarks give a small least-squares problem.

The depth this recovers is only as right as MediaPipe's guess of the hand's size: a hand s times bigger than
assumed puts every point s times further along its camera ray. That factor is fixed for a person, so it is fitted
afterwards from the grasp anchor (see segment.py).
"""

import numpy as np

from robot_arm.video.tracking import INDEX_TIP, THUMB_TIP


def hand_translation(image_px: np.ndarray, world_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame translation of MediaPipe's hand-centred world landmarks."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    du = image_px[:, 0] - cx
    dv = image_px[:, 1] - cy
    X, Y, Z = world_m.T
    n = len(world_m)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[:n, 0], A[:n, 2], b[:n] = fx, -du, du * Z - fx * X
    A[n:, 1], A[n:, 2], b[n:] = fy, -dv, dv * Z - fy * Y
    T, *_ = np.linalg.lstsq(A, b, rcond=None)
    return T


def pinch_point_cam(image_px: np.ndarray, world_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Midpoint between thumb tip and index tip, camera frame (metres)."""
    T = hand_translation(image_px, world_m, K)
    return world_m[[THUMB_TIP, INDEX_TIP]].mean(axis=0) + T


def aperture(world_m: np.ndarray) -> float:
    """Thumb-tip to index-tip distance (metres); independent of the camera distance."""
    return float(np.linalg.norm(world_m[THUMB_TIP] - world_m[INDEX_TIP]))


def reprojection_error(image_px: np.ndarray, world_m: np.ndarray, K: np.ndarray, T: np.ndarray) -> float:
    pc = world_m + T
    uv = pc @ K.T
    return float(np.linalg.norm(uv[:, :2] / uv[:, 2:3] - image_px, axis=1).mean())
