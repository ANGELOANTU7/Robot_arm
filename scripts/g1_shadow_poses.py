"""Design the five-fingered Shadow hand's cap pinch and thumbs-up for the G1 (the values go into hands.py).

    python scripts/g1_shadow_poses.py pinch      # search open/closed pinch poses, print the Hand fields
    python scripts/g1_shadow_poses.py thumbs     # thumbs-up pose, thumb direction and roll

Kinematic search over the hand's actuator targets (a distal tendon's target is split evenly between its two
joints), on every collision point of the hand (mesh vertices, box corners) in the right_wrist_yaw_link frame.

The pinch is a claw pinch like the Dex3's, for a 2.8 cm cap lying flat (1.2 cm tall): the thumb's and the index
finger's pads face each other across the cap, and along some "down" direction (perpendicular to the line between
the pads) the two fingertips must be the lowest points of the hand, so the pads reach the cap's sides 6 mm above
the table while nothing else comes below the cap's top. The search maximises the worst of these clearances.
"""

import argparse
import dataclasses

import mujoco
import numpy as np

from robot_arm.humanoid.config import G1_CFG
from robot_arm.humanoid.hands import SHADOW
from robot_arm.humanoid.scene import HAND_TYPE, G1Ids, build_model

PINCH_BODIES = ("sh_rh_thdistal", "sh_rh_ffdistal")
PAD_HEIGHT = 0.006  # the pads meet the cap halfway up its side
TABLE_CLEAR = 0.001  # every point of the hand above the table (relative to the pads: -5 mm)
CAP_CLEAR = 0.010  # everything but the two fingertips above the cap's top (12 mm) with some room


class HandGeometry:
    """Collision points of the Shadow hand for given actuator targets, in the wrist frame."""

    def __init__(self):
        cfg = dataclasses.replace(G1_CFG, hand="shadow")
        self.m = build_model(cfg)
        self.d = mujoco.MjData(self.m)
        self.ids = G1Ids(self.m)
        self.d.qpos[:] = self.ids.ready_qpos
        m = self.m
        self.wrist = self.ids.hand_body
        self.geoms, self.names = [], []
        for g in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
            if m.geom_contype[g] != HAND_TYPE or body not in SHADOW.colliders:
                continue
            self.geoms.append((g, self._local_points(g)))
            self.names.append(body)
        # Actuator -> joints it sets (a tendon: its two joints, half each).
        self.targets = []
        for name in SHADOW.actuators:
            a = m.actuator(name).id
            tid = m.actuator_trnid[a, 0]
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON:
                adr, num = m.tendon_adr[tid], m.tendon_num[tid]
                joints = [m.wrap_objid[adr + k] for k in range(num)]
                self.targets.append([(m.jnt_qposadr[j], 1.0 / num, m.jnt_range[j]) for j in joints])
            else:
                self.targets.append([(m.jnt_qposadr[tid], 1.0, m.jnt_range[tid])])
        self.ctrl_range = np.array([m.actuator_ctrlrange[m.actuator(n).id] for n in SHADOW.actuators])

    def _local_points(self, g: int) -> np.ndarray:
        m = self.m
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            adr, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            verts = m.mesh_vert[adr:adr + n].astype(float)
            return verts[::max(1, len(verts) // 150)]  # enough to bound the shape to a millimetre or two
        s = m.geom_size[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            return np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])])
        r = s[0]
        half = s[1] if m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER) else 0.0
        dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                         [.7, .7, 0], [-.7, .7, 0], [.7, -.7, 0], [-.7, -.7, 0]])
        return np.concatenate([np.array([0, 0, e]) + r * dirs for e in (-half, half)])

    def points(self, pose: np.ndarray) -> dict[str, np.ndarray]:
        m, d = self.m, self.d
        for value, joints in zip(np.clip(pose, *self.ctrl_range.T), self.targets):
            for adr, share, rng in joints:
                d.qpos[adr] = np.clip(value * share, *rng)
        mujoco.mj_kinematics(m, d)
        Rw, xw = d.xmat[self.wrist].reshape(3, 3), d.xpos[self.wrist]
        out: dict[str, list] = {}
        for (g, local), body in zip(self.geoms, self.names):
            world = d.geom_xpos[g] + local @ d.geom_xmat[g].reshape(3, 3).T
            out.setdefault(body, []).append((world - xw) @ Rw)
        return {k: np.concatenate(v) for k, v in out.items()}


def pads(pts: dict) -> tuple[np.ndarray, np.ndarray]:
    """The closest points of the thumb's and the index finger's tips: where they meet the cap."""
    th, ff = pts[PINCH_BODIES[0]], pts[PINCH_BODIES[1]]
    dist = np.linalg.norm(th[:, None] - ff[None], axis=2)
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    return th[i], ff[j]


def pinch_score(pts: dict, n_dirs: int = 72) -> tuple[float, np.ndarray]:
    """(clearance, down direction): the worst clearance of the table and cap constraints for the best down
    direction perpendicular to the pads' line. Positive: feasible with that much room (metres)."""
    p_th, p_ff = pads(pts)
    mid, axis = (p_th + p_ff) / 2, (p_ff - p_th) / np.linalg.norm(p_ff - p_th)
    e1 = np.cross(axis, [0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.cross(axis, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    tips = np.concatenate([pts[b] for b in PINCH_BODIES]) - mid
    rest = np.concatenate([v for k, v in pts.items() if k not in PINCH_BODIES]) - mid
    phi = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    G = np.outer(e1, np.cos(phi)) + np.outer(e2, np.sin(phi))  # (3, n_dirs) candidate down directions
    # Height above the pads (positive up): points further along g are lower.
    score = np.minimum((-(tips @ G)).min(0) + PAD_HEIGHT - TABLE_CLEAR, (-(rest @ G)).min(0) - CAP_CLEAR)
    k = int(np.argmax(score))
    best, best_g = score[k], G[:, k]
    return float(best), best_g


def search_pinch(geo: HandGeometry, rng: np.random.Generator, samples: int, gap: float, others: dict) -> list:
    """Random search plus local refinement for a pose whose pads are `gap` apart."""
    lo, hi = geo.ctrl_range.T
    free = [i for i, a in enumerate(SHADOW.actuators) if a.split("_")[-1] in
            ("THJ5", "THJ4", "THJ3", "THJ2", "THJ1", "FFJ4", "FFJ3", "FFJ0")]
    base = np.zeros(len(SHADOW.actuators))
    for k, v in others.items():
        base[[a.split("_")[-1] for a in SHADOW.actuators].index(k)] = v

    def objective(pose):
        pts = geo.points(pose)
        p_th, p_ff = pads(pts)
        score, g = pinch_score(pts)
        return score - 2.0 * abs(np.linalg.norm(p_ff - p_th) - gap), g

    results = []
    for _ in range(samples):
        pose = base.copy()
        pose[free] = rng.uniform(lo[free], hi[free])
        results.append((objective(pose)[0], pose))
    results.sort(key=lambda r: -r[0])
    refined = []
    for score, pose in results[:8]:
        step = 0.15
        for _ in range(400):
            cand = pose.copy()
            cand[free] = np.clip(cand[free] + rng.normal(0, step, len(free)), lo[free], hi[free])
            s = objective(cand)[0]
            if s > score:
                score, pose = s, cand
            else:
                step = max(0.01, step * 0.995)
        refined.append((score, pose))
    refined.sort(key=lambda r: -r[0])
    return refined


def cup_score(pts: dict, radius: float, half_len: float = 0.06, margin: float = 0.004, n_dirs: int = 72,
              palm_weight: float = 0.01, prefer: int = 1) -> tuple[float, np.ndarray, np.ndarray]:
    """Two-finger cup grip: the thumb's and the index finger's pads on opposite sides of a cup of `radius`. For the
    best direction of the cup's axis (perpendicular to the line between the pads): (score, axis, pads' midpoint).
    The score is the worst clearance of the rest of the hand from the cup's wall (within `half_len` along the axis
    from the grip), plus a small bonus for the palm facing along the axis (a flat palm, so turning the cup over
    leaves nothing under it; `prefer` = 1), or along the thumb's side (`prefer` = 2: an upright hand beside the
    cup)."""
    p_th, p_ff = pads(pts)
    mid, a = (p_th + p_ff) / 2, (p_ff - p_th) / np.linalg.norm(p_ff - p_th)
    e1 = np.cross(a, [0, 1.0, 0]) if abs(a[1]) < 0.9 else np.cross(a, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    rest = np.concatenate([v for k, v in pts.items() if k not in PINCH_BODIES]) - mid
    tips = np.concatenate([pts[b] for b in PINCH_BODIES]) - mid
    phi = np.linspace(0, np.pi, n_dirs // 2, endpoint=False)  # an axis and its opposite are the same
    C = np.outer(e1, np.cos(phi)) + np.outer(e2, np.sin(phi))  # (3, n) candidate cup axes
    along_r, along_t = rest @ C, tips @ C
    dist_r = np.sqrt(np.maximum((rest ** 2).sum(1)[:, None] - along_r ** 2, 0))
    dist_t = np.sqrt(np.maximum((tips ** 2).sum(1)[:, None] - along_t ** 2, 0))
    dist_r = np.where(np.abs(along_r) < half_len, dist_r, np.inf).min(0)
    clear = np.minimum(np.minimum(dist_r, 0.05 + radius + margin) - radius - margin, dist_t.min(0) - radius + 0.004)
    score = clear + palm_weight * np.abs(C[prefer])
    k = int(np.argmax(score))
    best = (score[k], C[:, k])
    return float(best[0]), best[1], mid


def search_cup(geo: HandGeometry, rng: np.random.Generator, samples: int, radius: float, others: dict,
               prefer: int = 1) -> list:
    """Poses whose pads are a cup's diameter apart, the rest of the hand clear of the cup."""
    lo, hi = geo.ctrl_range.T
    free = [i for i, a in enumerate(SHADOW.actuators) if a.split("_")[-1] in
            ("THJ5", "THJ4", "THJ3", "THJ2", "THJ1", "FFJ4", "FFJ3", "FFJ0")]
    base = np.zeros(len(SHADOW.actuators))
    for k, v in others.items():
        base[[a.split("_")[-1] for a in SHADOW.actuators].index(k)] = v

    def objective(pose):
        pts = geo.points(pose)
        p_th, p_ff = pads(pts)
        score = cup_score(pts, radius, prefer=prefer)[0]
        return score - 2.0 * abs(np.linalg.norm(p_ff - p_th) - 2 * radius)

    results = []
    for _ in range(samples):
        pose = base.copy()
        pose[free] = rng.uniform(lo[free], hi[free])
        results.append((objective(pose), pose))
    results.sort(key=lambda r: -r[0])
    refined = []
    for score, pose in results[:8]:
        step = 0.15
        for _ in range(400):
            cand = pose.copy()
            cand[free] = np.clip(cand[free] + rng.normal(0, step, len(free)), lo[free], hi[free])
            s = objective(cand)
            if s > score:
                score, pose = s, cand
            else:
                step = max(0.01, step * 0.995)
        refined.append((score, pose))
    refined.sort(key=lambda r: -r[0])
    return refined


def describe_cup(geo: HandGeometry, pose: np.ndarray, radius: float, prefer: int = 1) -> dict:
    pts = geo.points(pose)
    p_th, p_ff = pads(pts)
    score, c, mid = cup_score(pts, radius, prefer=prefer)
    return {"score_mm": round(1000 * score, 1), "gap_mm": round(1000 * np.linalg.norm(p_ff - p_th), 1),
            "grip_mid": mid.round(4), "cup_axis": c.round(3),
            "pinch_axis": ((p_ff - p_th) / np.linalg.norm(p_ff - p_th)).round(3)}


def describe(geo: HandGeometry, pose: np.ndarray) -> dict:
    pts = geo.points(pose)
    p_th, p_ff = pads(pts)
    score, g = pinch_score(pts)
    return {"clearance_mm": round(1000 * score, 1), "gap_mm": round(1000 * np.linalg.norm(p_ff - p_th), 1),
            "pinch_point": ((p_th + p_ff) / 2).round(4), "pinch_down": g.round(3),
            "pinch_axis": ((p_ff - p_th) / np.linalg.norm(p_ff - p_th)).round(3)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["pinch", "cup", "thumbs"])
    p.add_argument("--radius", type=float, default=0.033, help="cup: radius where it is gripped")
    p.add_argument("--side", action="store_true", help="cup: the cup's axis along the thumb (hand upright beside "
                   "the cup) instead of along the palm's normal (hand above it)")
    p.add_argument("--samples", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    geo = HandGeometry()
    rng = np.random.default_rng(args.seed)
    names = [a.split("_")[-1] for a in SHADOW.actuators]

    def fingers(j3, j0):  # the middle, ring and little fingers, all the same
        return {k: v for f in ("MF", "RF", "LF") for k, v in ((f"{f}J3", j3), (f"{f}J0", j0))}

    if args.what == "pinch":
        curls = {"curled": fingers(1.57, 2.6), "straight": fingers(0.0, 0.0), "half": fingers(0.8, 1.2)}
        for label, others in curls.items():
            closed = search_pinch(geo, rng, args.samples, 0.020, others)[0]
            print(f"[{label}] closed: score {closed[0] * 1000:.1f} mm", describe(geo, closed[1]))
            print("   pose", dict(zip(names, closed[1].round(3))), flush=True)
    elif args.what == "cup":
        for label, others in (("curled", fingers(1.57, 2.6)), ("straight", fingers(0.0, 0.0))):
            prefer = 2 if args.side else 1
            best = search_cup(geo, rng, args.samples, args.radius, others, prefer)[0]
            print(f"[{label}] cup grip: score {best[0] * 1000:.1f}", describe_cup(geo, best[1], args.radius, prefer))
            print("   pose", dict(zip(names, best[1].round(3))), flush=True)
    else:
        raise SystemExit("thumbs: see hands.py (not searched)")


if __name__ == "__main__":
    main()
