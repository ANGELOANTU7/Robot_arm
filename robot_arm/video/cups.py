"""Where same-colour cups stand, from their blobs in a calibrated image (any camera tilt, from straight down to
oblique).

A cup's outline is its widest circle (a mouth-down cup: the mouth on the table) on the side facing the point of
the table straight below the camera; on the far side the rest of the cup can stick out beyond it. So the circle
is fitted, with its known radius, to the outline points that face the camera's ground point, after mapping them
onto the circle's plane.
"""

import cv2
import numpy as np

from robot_arm.video.calibrate import Camera


def backproject(camera: Camera, px: np.ndarray, z: float) -> np.ndarray:
    """(N, 2) pixels -> (N, 3) points on the horizontal plane at height z (table frame)."""
    px = np.atleast_2d(np.asarray(px, float))
    d = np.linalg.solve(camera.K, np.column_stack([px, np.ones(len(px))]).T).T @ camera.R  # table-frame rays
    o = camera.center_table
    s = (z - o[2]) / d[:, 2]
    return o + s[:, None] * d


def outline(mask: np.ndarray) -> np.ndarray:
    """(N, 2) pixels of the outer contour of a mask's largest blob."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2))
    return max(contours, key=len)[:, 0, :].astype(float)


def fit_circle(points: np.ndarray, radius: float, centre: np.ndarray, ground: np.ndarray, near: bool = True,
               iters: int = 6) -> np.ndarray:
    """Centre of a circle of known radius through the outline points (N, 2) on the side facing `ground` (or the
    far side), refined from an initial centre."""
    c = np.asarray(centre, float).copy()
    for _ in range(iters):
        toward = ground - c
        dist = np.linalg.norm(toward)
        if dist > radius:  # the camera's ground point is outside the circle: one side is the true outline
            side = (points - c) @ (toward / dist)
            sel = points[side > 0] if near else points[side < 0]
        else:
            sel = points
        if len(sel) < 8:
            break
        for _ in range(10):  # Gauss-Newton on sum (|p - c| - r)^2
            d = sel - c
            n = np.linalg.norm(d, axis=1, keepdims=True)
            res = n[:, 0] - radius
            J = -d / np.maximum(n, 1e-9)
            step = np.linalg.lstsq(J, -res, rcond=None)[0]
            c += step
            if np.linalg.norm(step) < 1e-5:
                break
    return c


def cup_centre(mask: np.ndarray, camera: Camera, z: float, radius: float, near: bool = True) -> np.ndarray | None:
    """Table-frame centre (at height z) of the circle of `radius` outlining a cup's blob."""
    pts = outline(mask)
    if len(pts) < 20:
        return None
    P = backproject(camera, pts, z)[:, :2]
    c = fit_circle(P, radius, P.mean(axis=0), camera.center_table[:2], near)
    return np.array([c[0], c[1], z])


def two_cup_centres(mask: np.ndarray, camera: Camera, z: float, radius: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Centres of two touching cups forming one blob: the blob is split along its long axis, and each cup's
    circle is fitted to its own part of the outer outline."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return None
    # The blob mapped onto the table plane is stretched away from the camera's ground point by the cups' walls, so
    # its shape doesn't give the pair's direction: try every direction and keep the two circles that follow the
    # outline's near side best and stand a rim diameter apart.
    out = backproject(camera, outline(mask), z)[:, :2]
    mean = out.mean(axis=0)
    ground = camera.center_table[:2]
    best, best_cost = None, np.inf
    for angle in np.radians(np.arange(0, 180, 10)):
        axis = np.array([np.cos(angle), np.sin(angle)])
        side = (out - mean) @ axis
        if (side > 0).sum() < 20 or (side < 0).sum() < 20:
            continue
        cs = [fit_circle(out[side * sign > 0], radius, mean + sign * radius * axis, ground) for sign in (-1.0, 1.0)]
        res = []
        for c, sign in zip(cs, (-1.0, 1.0)):
            pts = out[side * sign > 0]
            res.append(np.abs(np.linalg.norm(pts[(pts - c) @ (ground - c) > 0] - c, axis=1) - radius))
        res = np.concatenate(res)
        cost = np.median(res) + abs(np.linalg.norm(cs[0] - cs[1]) - 2 * radius) if len(res) else np.inf
        if cost < best_cost:
            best, best_cost = cs, cost
    if best is None:
        return None
    centres = best
    return tuple(np.array([c[0], c[1], z]) for c in centres)


def long_axis(mask: np.ndarray, camera: Camera, z: float = 0.0) -> np.ndarray:
    """Horizontal unit vector along a blob's longest extent on the table."""
    ys, xs = np.nonzero(mask)
    P = backproject(camera, np.column_stack([xs, ys])[:: max(1, len(xs) // 4000)], z)[:, :2]
    a = np.linalg.svd(P - P.mean(axis=0), full_matrices=False)[2][0]
    return np.array([a[0], a[1], 0.0])
