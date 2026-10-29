"""Record videos of G1 tries (successes and failures) to follow what the robot does.

    MUJOCO_GL=osmesa python scripts/g1_record_attempts.py --policy scripted --task full --episodes 3
    MUJOCO_GL=osmesa python scripts/g1_record_attempts.py --policy place=runs/g1_place_ppo/policy.pt,toss=runs/g1_toss_ppo/policy.pt

Each try becomes sim_videos/<label>/<task>_<seed>_<outcome>.mp4 (small: 320x240, side and front views side by
side, with the stage and the outcome written on it), and sim_videos/README.md lists them all.
"""

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

STAGE_NAMES = ["place cup A", "place cup B", "turn over + place top cup", "throw the cap", "thumbs-up"]


def record(job) -> dict:
    from robot_arm.humanoid.envs import G1PyramidEnv
    from robot_arm.humanoid.evaluate import make_policy

    task, stage, spec, label, seed, out_dir = job
    env = G1PyramidEnv(task=task, stage=stage, render_mode="rgb_array", camera="side", width=320, height=240)
    policy = make_policy(spec)
    if hasattr(policy, "start_at"):
        policy.start_at(seed)
    obs, _ = env.reset(seed=seed)
    policy.reset()

    def frame(text: str) -> np.ndarray:
        tiles = []
        for cam in ("side", "front"):
            env.camera = cam
            tiles.append(env.render().copy())
        img = np.concatenate(tiles, axis=1)
        cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    frames = []
    info = {"stage": env.stage, "is_success": False, "fail_reason": None, "misses": 0}
    for t in range(env.max_steps):
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if t % 2 == 0:  # 10 frames per second of simulated time
            frames.append(frame(f"{label} | {STAGE_NAMES[env.stage]} | t={t * env.cfg.control_dt:4.1f}s"
                                f" | misses {env.misses}"))
        if terminated or truncated:
            break
    outcome = "success" if info["is_success"] else (info["fail_reason"] or "timeout").replace(" ", "-")
    for _ in range(10):  # hold the last frame for a second, with the outcome
        frames.append(frame(f"{label} | {outcome.upper()} (reached: {STAGE_NAMES[env.stage]})"))
    path = Path(out_dir) / f"{task}_{seed}_{outcome}.mp4"
    imageio.mimsave(path, frames, fps=10, macro_block_size=1, quality=6)
    env.close()
    return {"file": path.name, "task": task, "seed": seed, "outcome": outcome, "reached": STAGE_NAMES[env.stage],
            "misses": int(info.get("misses", 0)), "seconds": round(t * env.cfg.control_dt, 1)}


def write_index(root: Path) -> None:
    lines = ["# Videos of the G1's tries", "",
             "Recorded with `scripts/g1_record_attempts.py`: every try is kept, failures included. Left: side "
             "view, right: front view. The stage the robot is in and the outcome are written on the video.", ""]
    for folder in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        meta = folder / "tries.json"
        if not meta.exists():
            continue
        info = json.loads(meta.read_text())
        note = (folder / "notes.txt").read_text().strip() if (folder / "notes.txt").exists() else ""
        lines += [f"## {folder.name}", "", f"Policy: `{info['policy']}`", ""] + ([note, ""] if note else [])
        lines += ["| Video | Task | Outcome | Reached | Missed throws |", "|---|---|---|---|---|"]
        for r in info["tries"]:
            lines.append(f"| [{r['file']}]({folder.name}/{r['file']}) | {r['task']} | {r['outcome']} | {r['reached']} "
                         f"| {r['misses']} |")
        lines.append("")
    (root / "README.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="scripted", help="as for g1_evaluate.py: scripted, replay:<dir>, place=..,toss=..")
    p.add_argument("--label", help="folder name (default: time and a short policy name)")
    p.add_argument("--task", default="full", choices=["place", "toss", "full"])
    p.add_argument("--stage", type=int, help="place task: which cup (0, 1, 2)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=95_000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("sim_videos"))
    args = p.parse_args()

    short = args.policy if len(args.policy) < 20 else args.policy.split("/")[-2]
    label = args.label or time.strftime("%Y%m%d-%H%M") + "_" + short.replace(":", "-").replace("/", "-")
    out_dir = args.out / label
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.task, args.stage, args.policy, label, args.seed + i, str(out_dir)) for i in range(args.episodes)]
    with get_context("spawn").Pool(args.workers) as pool:
        tries = pool.map(record, jobs)
    meta = out_dir / "tries.json"
    old = json.loads(meta.read_text())["tries"] if meta.exists() else []
    meta.write_text(json.dumps({"policy": args.policy, "tries": old + tries}, indent=1))
    write_index(args.out)
    for r in tries:
        print(out_dir / r["file"], r["outcome"])


if __name__ == "__main__":
    main()
