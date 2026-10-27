"""Synthetic human skill demonstrations for the cup pyramid (to try the pipeline before recording, and tests).

Table frame as in the videos: origin at the marker sheet, the camera on the -y side, the person on the +y side
reaching towards -y. The person's right is -x.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from robot_arm.synthetic import min_jerk
from robot_arm.video.calibrate import Camera
from robot_arm.video.skills import GRAVITY, SkillDemo, SkillTrack
from robot_arm.video.synthetic_hand import hand_landmarks

CUP_H, CUP_R = 0.075, 0.031  # the person's cups (outer radius at the table-side rim)
BALL_R = 0.02
BALL_DEPTH = BALL_R + 0.008  # ball centre from the palm centre along the palm normal while held


def side_grasp_rot(yaw: float = 0.0) -> np.ndarray:
    """Right hand reaching forward (-y), palm facing the person's left (+x), thumb up; `yaw` turns it about z."""
    c, s = np.cos(yaw), np.sin(yaw)
    x = np.array([s, -c, 0.0])  # fingers forward
    z = np.array([c, s, 0.0])  # palm normal: the person's left
    return np.column_stack([x, np.cross(z, x), z])


def palm_down_rot() -> np.ndarray:
    """Fingers forward, palm facing down (picking a ball from above)."""
    x = np.array([0.0, -1.0, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


def palm_up_rot(direction: np.ndarray) -> np.ndarray:
    """Palm facing up, fingers along the horizontal part of `direction` (an underhand toss)."""
    x = np.array([direction[0], direction[1], 0.0])
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, 1.0])
    return np.column_stack([x, np.cross(z, x), z])


def _slerp(R0: np.ndarray, R1: np.ndarray, s: np.ndarray) -> np.ndarray:
    rel = Rotation.from_matrix(R0.T @ R1).as_rotvec()
    return R0 @ Rotation.from_rotvec(s[:, None] * rel).as_matrix()


def _segments(start_pos, start_clos, start_rot, segments, fps):
    """Min-jerk segments (target, seconds, closure at the end, palm rotation at the end).

    Returns positions, closures, rotations and the index of each segment's last frame."""
    pos, clos, rots = [np.asarray(start_pos, float)[None]], [np.array([start_clos])], [start_rot[None]]
    ends, total = [], 1
    for target, secs, c_end, rot in segments:
        n = max(2, int(round(secs * fps)))
        s = min_jerk(np.zeros(1), np.ones(1), n)[1:, 0]
        pos.append(pos[-1][-1] + s[:, None] * (np.asarray(target, float) - pos[-1][-1]))
        clos.append(clos[-1][-1] + s * (c_end - clos[-1][-1]))
        rots.append(_slerp(rots[-1][-1], rot, s))
        total += n - 1
        ends.append(total - 1)
    return np.concatenate(pos), np.concatenate(clos), np.concatenate(rots), ends


def pyramid_layout(rng: np.random.Generator) -> dict:
    """Where the person's pyramid goes: base cups A and B (diagonal, like the robot's) and the top cup."""
    pyr = np.array([rng.uniform(-0.04, 0.04), rng.uniform(0.15, 0.22)])
    u = np.array([np.cos(-2.4), np.sin(-2.4)])
    spacing = 2 * CUP_R + rng.uniform(0.004, 0.012)
    return {"A": pyr - u * spacing / 2, "B": pyr + u * spacing / 2, "top": pyr}


def place_demo(rng: np.random.Generator, kind: str = "A", fps: float = 30.0) -> tuple[SkillDemo, list]:
    """Pick a cup from the supply area and put it in its pyramid spot. kind: 'A', 'B' or 'top'.

    Returns the demo and the other (static) cups in view as table-side rim centres."""
    layout = pyramid_layout(rng)
    supply = {"A": [-0.20, 0.26], "B": [-0.24, 0.16], "top": [-0.16, 0.34]}[kind]
    start = np.array([*(np.asarray(supply) + rng.uniform(-0.02, 0.02, 2)), 0.0])
    end = np.array([*layout[kind], CUP_H if kind == "top" else 0.0])
    static = {"A": [[-0.26, 0.10, 0.0]], "B": [[*layout["A"], 0.0]],
              "top": [[*layout["A"], 0.0], [*layout["B"], 0.0]]}[kind]
    grasp_h = CUP_H * (0.7 if kind == "top" else 0.6)
    rot = side_grasp_rot(rng.uniform(-0.2, 0.2))
    palm_off = -0.05 * rot[:, 2]  # the palm sits beside the cup, the cup on the palm's side
    grasp = start + [0, 0, grasp_h] + palm_off
    place = end + [0, 0, grasp_h] + palm_off
    rest = grasp + [rng.uniform(-0.05, 0.05), rng.uniform(0.18, 0.25), rng.uniform(0.10, 0.18)]
    carry_z = max(grasp[2], place[2]) + rng.uniform(0.08, 0.14)
    open_c, closed_c = 0.1, rng.uniform(0.75, 0.9)
    segs = [
        (rest, rng.uniform(0.6, 0.9), 0.15, rot),  # still
        (grasp + [-0.02, 0.10, 0.03], rng.uniform(0.6, 0.9), open_c, rot),
        (grasp, rng.uniform(0.3, 0.5), open_c, rot),
        (grasp, rng.uniform(0.25, 0.35), closed_c, rot),  # close
        (np.array([grasp[0], grasp[1], carry_z]), rng.uniform(0.4, 0.6), closed_c, rot),
        (np.array([place[0], place[1], carry_z]), rng.uniform(0.6, 0.9), closed_c, rot),
        (place, rng.uniform(0.4, 0.6), closed_c, rot),
        (place, rng.uniform(0.1, 0.2), closed_c, rot),  # settle
        (place, rng.uniform(0.2, 0.3), open_c, rot),  # open
        (place + [-0.04, 0.06, 0.08], rng.uniform(0.4, 0.6), 0.15, rot),
        (place + [-0.04, 0.06, 0.08], 0.6, 0.15, rot),
    ]
    palm, clos, rots, ends = _segments(rest, 0.15, rot, segs, fps)
    t = np.arange(len(palm)) / fps
    skill = "place_top" if kind == "top" else "place_base"
    demo = SkillDemo(skill, t, palm, rots, clos, start, end, float(t[ends[2]]), float(t[ends[7]]),
                     info={"synthetic": 1.0, "grasp_height": grasp_h})
    return demo, [np.array(s, float) for s in static]


def toss_demo(rng: np.random.Generator, fps: float = 60.0) -> SkillDemo:
    """Pick the ball off the table, swing it palm-up and toss it underhand into the top cup."""
    mouth = np.array([*pyramid_layout(rng)["top"], 2 * CUP_H])
    ball = np.array([rng.uniform(-0.22, -0.16), rng.uniform(0.22, 0.30), BALL_R])
    # Released on the person's right of the cup, so the ball flies across the camera's view.
    bearing = np.pi - rng.uniform(0.0, 0.5)
    dist = rng.uniform(0.25, 0.30)
    R = mouth + np.array([dist * np.cos(bearing), dist * np.sin(bearing), rng.uniform(0.06, 0.12)])
    ball_release = R + [0, 0, BALL_DEPTH]  # palm up: the ball rests on the palm
    tf = rng.uniform(0.28, 0.36)
    v = (mouth - ball_release + np.array([0, 0, 0.5 * GRAVITY * tf * tf])) / tf
    # Swing: constant acceleration from rest at the wind-up point, reaching v at R after n frames.
    n_acc = int(round(rng.uniform(0.18, 0.26) * fps))
    t_acc = n_acc / fps
    windup = R - v * t_acc / 2
    down, up = palm_down_rot(), palm_up_rot(v)
    grab = ball - BALL_DEPTH * down[:, 2]
    rest = grab + [0.0, 0.18, 0.15]
    segs = [
        (rest, rng.uniform(0.6, 0.9), 0.15, down),  # still
        (grab + [0, 0.02, 0.06], rng.uniform(0.5, 0.7), 0.1, down),
        (grab, 0.3, 0.1, down),
        (grab, 0.25, 0.8, down),  # grab
        (grab + [0, 0, 0.08], 0.3, 0.8, down),  # lift
        (windup, rng.uniform(0.6, 0.8), 0.6, up),  # turn palm-up, cupped hand
        (windup, 0.2, 0.6, up),
    ]
    palm, clos, rots, ends = _segments(rest, 0.15, down, segs, fps)
    tt = np.arange(1, n_acc + 1)[:, None] / fps
    swing = windup + 0.5 * (v / t_acc) * tt ** 2
    t_stop = 0.15  # follow-through: the hand slows to a stop
    tt = np.arange(1, int(0.4 * fps))[:, None] / fps
    tt = np.minimum(tt, t_stop)
    follow = R + v * tt - 0.5 * (v / t_stop) * tt ** 2
    release_idx = len(palm) + n_acc - 1
    palm = np.concatenate([palm, swing, follow])
    clos = np.concatenate([clos, np.full(n_acc, 0.6), np.linspace(0.6, 0.1, len(follow))])
    rots = np.concatenate([rots, np.repeat(up[None], n_acc + len(follow), 0)])
    t = np.arange(len(palm)) / fps
    return SkillDemo("toss", t, palm, rots, clos, ball, mouth, float(t[ends[2]]), float(t[release_idx]), v,
                     ball_release, {"synthetic": 1.0})


def look_at_camera(pos=(-0.05, -0.85, 0.55), look=(-0.06, 0.18, 0.12), width: int = 1280, height: int = 720,
                   focal: float = 1000.0) -> Camera:
    """A pinhole camera at `pos` looking at `look` (table frame). The default frames the whole skill workspace:
    the sheet, the cups, the reach above them, the wind-up and the ball's flight."""
    pos, look = np.array(pos, float), np.array(look, float)
    fwd = (look - pos) / np.linalg.norm(look - pos)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    return Camera(K, R, -R @ pos, width, height)


def synthetic_skill_demos(rng: np.random.Generator, n_per_skill: int = 10, camera: Camera | None = None
                          ) -> list[tuple[str, SkillDemo]]:
    """Synthetic human skill demos passed through the video reconstruction (so they carry its errors).

    place_base clips alternate between cup A and cup B. Returns (name, demo) pairs."""
    from robot_arm.video.skills import reconstruct_skill

    camera = camera or look_at_camera()
    out = []
    for i in range(n_per_skill):
        for kind, name in ((["A", "B"][i % 2], "place_base"), ("top", "place_top"), (None, "toss")):
            if kind is None:
                truth, static = toss_demo(rng), []
            else:
                truth, static = place_demo(rng, kind)
            track = render_track(truth, camera, rng, static, size_error=rng.uniform(0.85, 1.2))
            demo = reconstruct_skill(track, camera, truth.skill, cup_height=CUP_H, cup_radius=CUP_R,
                                     ball_radius=BALL_R)
            demo.info["synthetic"] = 1.0
            out.append((f"{name}_{i:02d}", demo))
    return out


def cup_front_px(camera: Camera, base: np.ndarray, radius: float = CUP_R) -> np.ndarray:
    """Pixel of a cup's table-side rim point nearest the camera (the lowest pixel of its blob)."""
    view = base - camera.center_table
    view[2] = 0
    return camera.project(base - radius * view / np.linalg.norm(view))[0]


def object_paths(demo: SkillDemo) -> np.ndarray:
    """(T, 3) path of the manipulated object: a cup's table-side rim centre, or the ball centre."""
    T = len(demo.t)
    g = int(np.searchsorted(demo.t, demo.t_grasp)) + 1
    r = int(np.searchsorted(demo.t, demo.t_release))
    path = np.repeat(demo.obj_start[None], T, 0)
    if demo.skill == "toss":
        held = demo.palm + BALL_DEPTH * demo.palm_rot[:, :, 2]
        path[g:r + 1] = held[g:r + 1]
        dt = (demo.t[r:] - demo.t[r])[:, None]
        path[r:] = held[r] + demo.release_vel * dt + 0.5 * np.array([0, 0, -GRAVITY]) * dt ** 2
    else:
        path[g:r] = demo.obj_start + demo.palm[g:r] - demo.palm[g]
        path[r:] = demo.obj_end
    return path


def render_track(demo: SkillDemo, camera: Camera, rng: np.random.Generator, static_cups=(),
                 size_error: float = 1.0, pixel_noise: float = 1.0) -> SkillTrack:
    """What the trackers would see: hand landmarks (MediaPipe-like, wrong hand size) and object blobs.

    Columns of objects_px: the manipulated object, then `static_cups`. For the toss the top cup (the last
    column) is added from the demo's mouth."""
    T = len(demo.t)
    hand_px, hand_world = np.zeros((T, 21, 2)), np.zeros((T, 21, 3))
    for i in range(T):
        pc = camera.to_cam(hand_landmarks(demo.closure[i], demo.palm_rot[i], demo.palm[i]))
        uv = pc @ camera.K.T
        hand_px[i] = uv[:, :2] / uv[:, 2:3]
        hand_world[i] = (pc - pc.mean(axis=0)) / size_error
    hand_px += rng.normal(0, pixel_noise, hand_px.shape)
    hand_world += rng.normal(0, 0.002, hand_world.shape)
    hand_px[8:10] = np.nan  # a couple of dropped detections

    path = object_paths(demo)
    if demo.skill == "toss":
        static_cups = [*static_cups, demo.obj_end - [0, 0, CUP_H]]
    cols, radii = [], []
    if demo.skill == "toss":
        cols.append(camera.project(path))
        pc = camera.to_cam(path)
        dist = np.linalg.norm(pc, axis=1)
        cos = pc[:, 2] / dist  # the ball's image is an ellipse, long semi-axis ~ f R / (d cos^2)
        radii.append(camera.K[0, 0] * BALL_R / (dist * cos ** 2) + rng.normal(0, 0.3, T))
        below = path[:, 2] < demo.obj_end[2] - 0.01  # inside the cup: out of sight
        r = int(np.searchsorted(demo.t, demo.t_release))
        cols[0][r:][below[r:]] = np.nan
    else:
        cols.append(np.array([cup_front_px(camera, p) for p in path]))
        radii.append(np.full(T, camera.K[0, 0] * CUP_R / np.linalg.norm(camera.to_cam(demo.obj_start[None]))))
    for base in static_cups:
        cols.append(np.repeat(cup_front_px(camera, np.asarray(base))[None], T, 0))
        radii.append(np.full(T, camera.K[0, 0] * CUP_R / np.linalg.norm(camera.to_cam(np.asarray(base)[None]))))
    objects_px = np.stack(cols, 1) + rng.normal(0, pixel_noise, (T, len(cols), 2))
    return SkillTrack(demo.t.copy(), hand_px, hand_world, objects_px, np.stack(radii, 1))
