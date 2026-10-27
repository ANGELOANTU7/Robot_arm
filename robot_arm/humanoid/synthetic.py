"""Synthetic human skill demonstrations for the cup pyramid (to try the pipeline before recording, and tests).

Table frame as in the videos: origin at the marker sheet, the camera on the -y side, the person on the +y side
reaching towards -y. The person's right is -x. The base cups stand side by side across the camera's view, cup B
on the person's right, so a sideways throw along them (from B's side) crosses the image.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from robot_arm.synthetic import min_jerk
from robot_arm.video.calibrate import Camera
from robot_arm.video.skills import GRAVITY, PINCH_TIPS, Props, SkillDemo, SkillTrack
from robot_arm.video.synthetic_hand import hand_landmarks

PROPS = Props()  # the person's party cups and bottle cap
H = PROPS.cup_height
SUPPLY = {"A": (-0.24, 0.36), "B": (-0.40, 0.20), "top": (-0.30, 0.50)}  # where each cup waits (person's right)


def pinch_offset(curl: float) -> np.ndarray:
    """The point between the thumb tip and the index fingertip in the palm frame (where a pinched cap is)."""
    return hand_landmarks(curl)[list(PINCH_TIPS)].mean(axis=0)


def side_grasp_rot(yaw: float = 0.0) -> np.ndarray:
    """Right hand reaching forward (-y), palm facing the person's left (+x), thumb up; `yaw` turns it about z."""
    c, s = np.cos(yaw), np.sin(yaw)
    x = np.array([s, -c, 0.0])  # fingers forward
    z = np.array([c, s, 0.0])  # palm normal: the person's left
    return np.column_stack([x, np.cross(z, x), z])


def palm_down_rot(yaw: float = 0.0) -> np.ndarray:
    """Fingers forward, palm facing down (pinching the cap from above); `yaw` turns it about z."""
    c, s = np.cos(yaw), np.sin(yaw)
    x = np.array([s, -c, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


def _rot2(a: float) -> np.ndarray:
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


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
    """Where the person's pyramid goes: base cups A and B (A on the person's left), the top cup, and the unit
    vector from A to B."""
    pyr = np.array([rng.uniform(-0.04, 0.04), rng.uniform(0.17, 0.23)])
    u = _rot2(np.pi + rng.uniform(-0.15, 0.15)) @ np.array([1.0, 0.0])
    spacing = 2 * PROPS.mouth_radius + rng.uniform(0.002, 0.008)
    return {"A": pyr - u * spacing / 2, "B": pyr + u * spacing / 2, "top": pyr, "axis": u}


def place_demo(rng: np.random.Generator, kind: str = "A", fps: float = 30.0) -> tuple[SkillDemo, list]:
    """Pick a cup from the supply area and put it in its pyramid spot. kind: 'A', 'B' or 'top'.

    Returns the demo and the other (static) cups in view: (name, table-side rim centre, mouth up)."""
    layout = pyramid_layout(rng)
    start = np.array([*(np.asarray(SUPPLY[kind]) + rng.uniform(-0.02, 0.02, 2)), 0.0])
    end = np.array([*layout[kind], H if kind == "top" else 0.0])
    supply = lambda k: (k, np.array([*SUPPLY[k], 0.0]), k == "top")  # noqa: E731
    pyramid = lambda k: (k, np.array([*layout[k], 0.0]), False)  # noqa: E731
    static = {"A": [supply("B"), supply("top")], "B": [pyramid("A"), supply("top")],
              "top": [pyramid("A"), pyramid("B")]}[kind]
    grasp_h = H * rng.uniform(0.45, 0.6)
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
        (place + [-0.04, 0.10, 0.08], rng.uniform(0.4, 0.6), 0.15, rot),
        (rest + [0, 0.15, 0.05], 0.6, 0.15, rot),  # out of the way for the still frames at the end
        (rest + [0, 0.15, 0.05], 0.8, 0.15, rot),
    ]
    palm, clos, rots, ends = _segments(rest, 0.15, rot, segs, fps)
    t = np.arange(len(palm)) / fps
    skill = "place_top" if kind == "top" else "place_base"
    demo = SkillDemo(skill, t, palm, rots, clos, start, end, float(t[ends[2]]), float(t[ends[7]]),
                     info={"synthetic": 1.0, "grasp_height": grasp_h})
    return demo, static


def toss_demo(rng: np.random.Generator, fps: float = 60.0) -> tuple[SkillDemo, list]:
    """Pinch the cap off the table and throw it sideways into the top cup: released on cup B's side, 5-30 cm
    from the mouth, flying along the base cups. Returns the demo and the pyramid's cups (as place_demo)."""
    layout = pyramid_layout(rng)
    u = np.array([*layout["axis"], 0.0])
    mouth = np.array([*layout["top"], 2 * H])
    cap = np.array([rng.uniform(-0.32, -0.24), rng.uniform(0.06, 0.12), PROPS.cap_height / 2])
    side = np.array([*(_rot2(rng.uniform(-0.25, 0.25)) @ u[:2]), 0.0])
    R = mouth + rng.uniform(0.05, 0.30) * side + [0, 0, rng.uniform(0.04, 0.12)]
    tf = rng.uniform(0.22, 0.34)
    v = (mouth - R + np.array([0, 0, 0.5 * GRAVITY * tf * tf])) / tf
    down = palm_down_rot(rng.uniform(-0.2, 0.2))
    c_hold = rng.uniform(0.45, 0.6)
    off = down @ pinch_offset(c_hold)  # palm centre -> cap while pinched
    # Swing: constant acceleration from rest at the wind-up point, reaching v at R after n frames.
    n_acc = int(round(rng.uniform(0.15, 0.22) * fps))
    t_acc = n_acc / fps
    windup = R - v * t_acc / 2
    grab = cap - off
    rest = grab + [0.0, 0.18, 0.15]
    segs = [
        (rest, rng.uniform(0.6, 0.9), 0.15, down),  # still
        (grab + [0, 0.02, 0.06], rng.uniform(0.5, 0.7), 0.1, down),
        (grab, 0.3, 0.1, down),
        (grab, 0.25, c_hold, down),  # pinch
        (grab + [0, 0, 0.08], 0.3, c_hold, down),  # lift
        (windup - off, rng.uniform(0.5, 0.8), c_hold, down),
        (windup - off, 0.2, c_hold, down),
    ]
    palm, clos, rots, ends = _segments(rest, 0.15, down, segs, fps)
    tt = np.arange(1, n_acc + 1)[:, None] / fps
    swing = windup + 0.5 * (v / t_acc) * tt ** 2 - off
    t_stop = 0.15  # follow-through: the hand slows to a stop while the fingers open
    tt = np.minimum(np.arange(1, int(0.6 * fps))[:, None] / fps, t_stop)
    follow = R - off + v * tt - 0.5 * (v / t_stop) * tt ** 2
    release_idx = len(palm) + n_acc - 1
    palm = np.concatenate([palm, swing, follow])
    opening = np.interp(np.arange(len(follow)), [0, int(0.1 * fps)], [c_hold, 0.1])
    clos = np.concatenate([clos, np.full(n_acc, c_hold), opening])
    rots = np.concatenate([rots, np.repeat(down[None], n_acc + len(follow), 0)])
    t = np.arange(len(palm)) / fps
    demo = SkillDemo("toss", t, palm, rots, clos, cap, mouth, float(t[ends[2]]), float(t[release_idx]), v, R,
                     {"synthetic": 1.0, "pinch_closure": c_hold, "axis_x": u[0], "axis_y": u[1]})
    return demo, [("A", np.array([*layout["A"], 0.0]), False), ("B", np.array([*layout["B"], 0.0]), False)]


def look_at_camera(pos=(-0.05, -0.85, 0.55), look=(-0.10, 0.22, 0.10), width: int = 1280, height: int = 720,
                   focal: float = 1000.0) -> Camera:
    """A pinhole camera at `pos` looking at `look` (table frame). The default frames the whole skill workspace:
    the sheet, the supply cups, the pyramid, the reach above them and the cap's flight."""
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
            truth, static = toss_demo(rng) if kind is None else place_demo(rng, kind)
            track = render_track(truth, camera, rng, static, size_error=rng.uniform(0.85, 1.2))
            demo = reconstruct_skill(track, camera, truth.skill, PROPS)
            demo.info["synthetic"] = 1.0
            out.append((f"{name}_{i:02d}", demo))
    return out


def cup_front_px(camera: Camera, base: np.ndarray, mouth_up: bool, props: Props = PROPS) -> np.ndarray:
    """Lowest pixel of a cup standing with its table-side rim centred at `base`: the red wall's point nearest the
    camera (above the white lip for a mouth-down cup)."""
    z = 0.0 if mouth_up else props.lip
    view = base - camera.center_table
    view[2] = 0
    return camera.project(base + [0, 0, z] - props.radius_at(z, mouth_up) * view / np.linalg.norm(view))[0]


def rim_px(camera: Camera, mouth: np.ndarray, props: Props = PROPS) -> np.ndarray:
    """Centre of a mouth-up cup's widest red row: its mouth centre, just under the lip."""
    return camera.project(mouth - [0, 0, props.lip])[0]


def object_paths(demo: SkillDemo) -> np.ndarray:
    """(T, 3) path of the manipulated object: a cup's table-side rim centre, or the cap's centre."""
    T = len(demo.t)
    g = int(np.searchsorted(demo.t, demo.t_grasp)) + 1
    r = int(np.searchsorted(demo.t, demo.t_release))
    path = np.repeat(demo.obj_start[None], T, 0)
    if demo.skill == "toss":
        held = demo.palm + demo.palm_rot @ pinch_offset(demo.info["pinch_closure"])
        path[g:r + 1] = held[g:r + 1]
        dt = (demo.t[r:] - demo.t[r])[:, None]
        path[r:] = held[r] + demo.release_vel * dt + 0.5 * np.array([0, 0, -GRAVITY]) * dt ** 2
    else:
        path[g:r] = demo.obj_start + demo.palm[g:r] - demo.palm[g]
        path[r:] = demo.obj_end
    return path


def cap_hidden(demo: SkillDemo, path: np.ndarray) -> np.ndarray:
    """Frames where the thrown cap is inside the top cup (out of sight)."""
    r = int(np.searchsorted(demo.t, demo.t_release))
    hidden = np.zeros(len(path), bool)
    inside = (path[:, 2] < demo.obj_end[2] - 0.005) & (np.linalg.norm(path[:, :2] - demo.obj_end[:2], axis=1)
                                                         < PROPS.mouth_radius)
    hidden[r:] = np.maximum.accumulate(inside[r:])
    return hidden


def render_track(demo: SkillDemo, camera: Camera, rng: np.random.Generator, static_cups=(),
                 size_error: float = 1.0, pixel_noise: float = 1.0) -> SkillTrack:
    """What the trackers would see: hand landmarks (MediaPipe-like, wrong hand size), the moved cup's lowest
    pixel or the cap, and the marks in the still frames (see SkillTrack)."""
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
    dist = np.linalg.norm(camera.to_cam(path), axis=1)
    f = camera.K[0, 0]
    marks = {}
    if demo.skill == "toss":
        obj_px = camera.project(path)
        obj_px[cap_hidden(demo, path)] = np.nan
        obj_r = f * PROPS.cap_radius / dist
        marks["rim"] = rim_px(camera, demo.obj_end)
    else:
        mouth_up = demo.skill == "place_top"
        obj_px = np.array([cup_front_px(camera, p, mouth_up) for p in path])
        obj_r = f * 0.07 / dist  # about the cup's half-diagonal
        marks["start"] = cup_front_px(camera, demo.obj_start, mouth_up)
        if mouth_up:
            marks["rim"] = rim_px(camera, demo.obj_end + [0, 0, H])
        else:
            marks["end"] = cup_front_px(camera, demo.obj_end, mouth_up)
    named = {name: base for name, base, _ in static_cups}
    if demo.skill != "place_base" and "A" in named and "B" in named:
        marks["base_a"], marks["base_b"] = (cup_front_px(camera, named[k], False) for k in "AB")
    marks = {k: v + rng.normal(0, pixel_noise, 2) for k, v in marks.items()}
    obj_px = obj_px + rng.normal(0, pixel_noise, obj_px.shape)
    return SkillTrack(demo.t.copy(), hand_px, hand_world, obj_px, obj_r + rng.normal(0, 0.3, T), marks)
