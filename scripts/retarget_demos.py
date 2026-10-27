"""Retarget human demos onto the Panda and generate the robot training dataset.

    python scripts/retarget_demos.py --human data/processed/human
    python scripts/retarget_demos.py --human data/processed/human_synthetic     # before recording

Each human demo is replayed by the robot on many random ball/cup layouts (with exploration noise for
corrective labels); successful episodes are saved to --out. With --videos, the first few demos are also
rendered on the human's own layout (next to the tracking overlay when it exists).
"""

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from robot_arm.dataset import generate, load_human_demos


def render_side_by_side(demo, overlay_path: Path | None, out_path: Path, camera: str = "side") -> bool:
    from robot_arm.env import PandaBallCupEnv
    from robot_arm.retarget import canonical_layout, rollout

    env = PandaBallCupEnv(render_mode="rgb_array", camera=camera)
    ball, cup = canonical_layout(demo)
    _, _, success, sim_frames = rollout(env, demo, ball, cup, np.random.default_rng(0), seed=0, render=True)
    env.close()
    fps = 1 / env.cfg.control_dt
    if overlay_path is not None and overlay_path.exists():
        human = imageio.mimread(overlay_path, memtest=False)
        h = sim_frames[0].shape[0]
        idx = np.linspace(0, len(human) - 1, len(sim_frames)).astype(int)  # stretch the human clip to match
        frames = []
        for i, f in zip(idx, sim_frames):
            hf = human[i]
            w = int(hf.shape[1] * h / hf.shape[0])
            frames.append(np.concatenate([cv2.resize(hf, (w, h)), f], axis=1))
    else:
        frames = sim_frames
    imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
    return success


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--human", type=Path, default=Path("data/processed/human"))
    p.add_argument("--out", type=Path, default=Path("data/processed/robot_demos.npz"))
    p.add_argument("--layouts", type=int, default=50, help="robot layouts per human demo")
    p.add_argument("--noise", type=float, default=0.3, help="exploration noise (action units) for DART")
    p.add_argument("--clean-fraction", type=float, default=0.2, help="episodes without noise")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--videos", type=int, default=3, help="render this many demos on their own layout")
    args = p.parse_args()

    demos = load_human_demos(args.human)
    print(f"Loaded {len(demos)} human demos from {args.human}")
    data = generate(demos, args.layouts, args.noise, args.clean_fraction, args.seed, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)

    ok = data["ep_success"]
    per_demo = [float(ok[data["ep_demo"] == i].mean()) for i in range(len(demos))]
    summary = {
        "episodes": int(len(ok)), "successful": int(ok.sum()), "transitions": int(len(data["obs"])),
        "success_clean": float(ok[data["ep_noise"] == 0].mean()) if (data["ep_noise"] == 0).any() else None,
        "success_noisy": float(ok[data["ep_noise"] > 0].mean()) if (data["ep_noise"] > 0).any() else None,
        "success_per_demo": [round(x, 2) for x in per_demo],
    }
    print(json.dumps(summary, indent=2))
    bad = [i for i, x in enumerate(per_demo) if x < 0.5]
    if bad:
        print(f"Demos {bad} rarely succeed on the robot; check their overlay/plot (grasp time, path).")

    if args.videos:
        vid_dir = args.out.parent / "retarget_videos"
        vid_dir.mkdir(exist_ok=True)
        names = sorted(p.stem for p in args.human.glob("*.npz") if not p.stem.endswith("_camera"))
        for name, demo in list(zip(names, demos))[: args.videos]:
            out = vid_dir / f"{name}.mp4"
            success = render_side_by_side(demo, args.human / f"{name}_overlay.mp4", out)
            print(f"{out} ({'success' if success else 'fail'})")


if __name__ == "__main__":
    main()
