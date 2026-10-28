"""G1 training data from human skill demos: retargeted rollouts (parallel), split into place and toss datasets."""

import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from robot_arm.video.skills import SkillDemo

_ENVS = {}


def demo_problem(demo: SkillDemo) -> str | None:
    """Why a reconstructed demo is too unreliable to learn from, or None."""
    info = demo.info
    if info.get("grasp_anchor_err", 0.0) > 0.12:
        return "the hand is far from the object at the grasp (depth scale)"
    if not 0.52 < info.get("scale", 1.0) < 1.98:
        return "depth scale at the edge of its search range"
    if demo.skill != "toss" and demo.t_release - demo.t_grasp < 0.25:
        return "grasp and release not told apart"
    if demo.skill == "toss" and not np.isfinite(demo.release_vel).all():
        return "no throw"
    return None


def load_skill_demos(folder: str | Path, check: bool = True) -> dict[str, list[SkillDemo]]:
    """SkillDemos in a folder, by skill (place_base, place_top, toss). With `check`, demos whose reconstruction
    is unreliable (see demo_problem) are left out; their names are in out["rejected"]."""
    paths = sorted(p for p in Path(folder).glob("*.npz") if not p.stem.endswith("_camera"))
    if not paths:
        raise FileNotFoundError(f"No skill demos (*.npz) in {folder}")
    out: dict[str, list] = {"place_base": [], "place_top": [], "toss": [], "rejected": []}
    for p in paths:
        demo = SkillDemo.load(p)
        demo.info["name"] = p.stem  # kept in memory only
        problem = demo_problem(demo) if check else None
        if problem:
            out["rejected"].append((p.stem, problem))
            continue
        out.setdefault(demo.skill, []).append(demo)
    return out


def throw_styles(toss_demos: list[SkillDemo], pool=None, calibrate: bool = True) -> list[dict]:
    """TossExpert parameters for every human throw: style in the trust region, aim calibrated in simulation."""
    from robot_arm.humanoid.retarget import calibrate_aim, throw_style, toss_params

    styles = []
    for demo in toss_demos:
        human = throw_style(demo)
        params = toss_params(human)
        if calibrate:
            params = calibrate_aim(params, pool=pool)
        params["human"] = {k: round(v, 3) for k, v in human.items()}
        styles.append(params)
    return styles


def _worker(job):
    from robot_arm.humanoid.envs import G1PyramidEnv
    from robot_arm.humanoid.retarget import G1Retargeter, expert_params, rollout

    task, place, style, seed, noise = job
    if task not in _ENVS:
        _ENVS[task] = G1PyramidEnv(task=task)
    env = _ENVS[task]
    rng = np.random.default_rng(seed)
    policy = G1Retargeter(place, expert_params(style))
    options = {"stage": int(rng.integers(3))} if task == "place" else None
    obs, actions, stages, done, info, _ = rollout(env, policy, rng, noise=noise, seed=seed, options=options)
    return obs.astype(np.float32), actions.astype(np.float32), stages.astype(np.int8), sorted(done)


def generate_g1(demos: dict[str, list[SkillDemo]], styles: list[dict], episodes: int = 600, noise: float = 0.3,
                clean_fraction: float = 0.2, task_mix=(("full", 0.5), ("place", 0.15), ("toss", 0.35)),
                seed: int = 0, workers: int = 4) -> dict:
    """Roll out the retargeted experts; keep each stage of an episode that the stage completed.

    Every episode draws one human demo per place stage (place_base for both base cups, place_top for the top
    cup) and one throw style."""
    from robot_arm.humanoid.retarget import place_demos_by_cup

    rng = np.random.default_rng(seed)
    if not demos.get("place_base") or not demos.get("place_top") or not styles:
        raise ValueError("Need at least one place_base, one place_top and one toss demo.")
    base_a, base_b, top = place_demos_by_cup(demos)
    tasks, probs = zip(*task_mix)
    jobs, meta = [], []
    for _ in range(episodes):
        task = str(rng.choice(tasks, p=np.array(probs) / sum(probs)))
        ia, ib, it, js = (int(rng.integers(len(base_a))), int(rng.integers(len(base_b))),
                          int(rng.integers(len(top))), int(rng.integers(len(styles))))
        place = {0: base_a[ia], 1: base_b[ib], 2: top[it]}
        ep_noise = 0.0 if rng.random() < clean_fraction else noise
        jobs.append((task, place, styles[js], int(rng.integers(1 << 30)), ep_noise))
        meta.append({"task": task, "demos": [ia, ib, it], "style": js, "noise": ep_noise})

    with get_context("spawn").Pool(workers) as pool:
        results = pool.map(_worker, jobs, chunksize=2)

    keep = {"place": ([], [], [], []), "toss": ([], [], [], [])}
    for e, ((o, a, st, done), m) in enumerate(zip(results, meta)):
        m["done_stages"] = done
        for stage in done:
            sel = st == stage
            key = "toss" if stage >= 3 else "place"
            for buf, x in zip(keep[key], (o[sel], a[sel], np.full(sel.sum(), e), st[sel])):
                buf.append(x)
    out = {"meta": meta}
    for key, (o, a, ep, st) in keep.items():
        if o:
            out[key] = {"obs": np.concatenate(o), "actions": np.concatenate(a), "episode": np.concatenate(ep),
                        "stage": np.concatenate(st)}
    return out


def summarize(meta: list[dict]) -> dict:
    """Success rates of the retargeted experts, per stage and per task."""
    out = {}
    for task in sorted({m["task"] for m in meta}):
        ms = [m for m in meta if m["task"] == task]
        row = {"episodes": len(ms)}
        if task == "full":
            for s in range(5):
                row[f"stage{s}"] = round(float(np.mean([s in m["done_stages"] for m in ms])), 3)
        else:
            last = 4 if task == "toss" else None  # the throw, then the thumbs-up
            row["success"] = round(float(np.mean([(last in m["done_stages"]) if last else len(m["done_stages"]) > 0
                                                  for m in ms])), 3)
        out[task] = row
    tossed = [m for m in meta if m["task"] == "toss" or (m["task"] == "full" and 2 in m["done_stages"])]
    by_style = {}
    for m in tossed:
        by_style.setdefault(m["style"], []).append(4 in m["done_stages"])
    out["toss_success_by_style"] = {int(k): round(float(np.mean(v)), 2) for k, v in sorted(by_style.items())}
    return out


def save_dataset(data: dict, out_dir: str | Path, styles: list[dict], extra: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for key in ("place", "toss"):
        if key in data:
            path = out_dir / f"g1_{key}.npz"
            np.savez_compressed(path, **data[key])
            files[key] = {"path": str(path), "transitions": int(len(data[key]["obs"])),
                          "episodes": int(len(np.unique(data[key]["episode"])))}
    summary = {"files": files, "rollouts": summarize(data["meta"]), **(extra or {})}
    (out_dir / "g1_styles.json").write_text(json.dumps(styles, indent=2))
    (out_dir / "g1_dataset_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_replay_policy(human_dir: str | Path, styles_path: str | Path | None = None):
    """G1ReplayPolicy for a folder of human skill demos. Throw styles come from `styles_path` (default: the
    g1_styles.json written next to the datasets, data/processed/), else they are made without aim calibration."""
    from robot_arm.humanoid.retarget import G1ReplayPolicy

    demos = load_skill_demos(human_dir)
    styles_path = Path(styles_path) if styles_path else Path(human_dir).parent / "g1_styles.json"
    styles = json.loads(styles_path.read_text()) if styles_path.exists() else throw_styles(demos["toss"],
                                                                                            calibrate=False)
    return G1ReplayPolicy(demos, styles)
