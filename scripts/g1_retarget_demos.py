"""Retarget human skill demos onto the G1 and generate the training datasets for the cup pyramid.

    python scripts/g1_make_synthetic_human.py                                  # before recording
    python scripts/g1_retarget_demos.py --human data/processed/human_pyramid_synthetic
    python scripts/g1_retarget_demos.py --human data/processed/human_pyramid --episodes 1000

Steps:
1. Each human throw becomes a throw style for the G1 (release height/distance/side, arc, wind-up), clipped to a
   trust region around the calibrated throw, and its aim is calibrated in simulation.
2. The retargeted experts run full pyramid episodes (plus some single-skill episodes) on random layouts with
   DART noise. Every stage that succeeds is kept: g1_place.npz (stages 0-2) and g1_toss.npz.
3. With --videos, a few demos per skill are also rendered on the G1, next to the clip's tracking overlay when
   it exists.
"""

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from robot_arm.humanoid.dataset import generate_g1, load_skill_demos, save_dataset, throw_styles
from robot_arm.humanoid.retarget import expert_params


def render_skill(demo, style: dict | None, overlay: Path | None, out_path: Path, seed: int = 0) -> bool:
    """One episode of a single skill on the G1 (stage 0 for place_base, 2 for place_top), side by side with the
    human clip's overlay video if there is one."""
    from robot_arm.humanoid.envs import G1PyramidEnv
    from robot_arm.humanoid.retarget import G1Retargeter, rollout

    if demo.skill == "toss":
        env, place, options = G1PyramidEnv(task="toss", render_mode="rgb_array", camera="side"), {}, None
    else:
        stage = 2 if demo.skill == "place_top" else 0
        env = G1PyramidEnv(task="place", render_mode="rgb_array", camera="front")
        place, options = {stage: demo}, {"stage": stage}
    policy = G1Retargeter(place, expert_params(style))
    *_, info, frames = rollout(env, policy, np.random.default_rng(seed), seed=seed, options=options, render=True)
    env.close()
    if overlay is not None and overlay.exists():
        human = imageio.mimread(overlay, memtest=False)
        h = frames[0].shape[0]
        idx = np.linspace(0, len(human) - 1, len(frames)).astype(int)  # stretch the human clip to match
        frames = [np.concatenate([cv2.resize(human[i], (int(human[i].shape[1] * h / human[i].shape[0]), h)), f],
                                 axis=1) for i, f in zip(idx, frames)]
    imageio.mimsave(out_path, frames, fps=20, macro_block_size=1)
    return bool(info["is_success"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--human", type=Path, default=Path("data/processed/human_pyramid"))
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--noise", type=float, default=0.3, help="exploration noise (action units) for DART")
    p.add_argument("--clean-fraction", type=float, default=0.2, help="episodes without noise")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-calibrate", action="store_true", help="skip the per-style aim calibration")
    p.add_argument("--styles", type=Path, help="reuse throw styles saved by an earlier run (g1_styles.json)")
    p.add_argument("--task-mix", default="full=0.5,place=0.15,toss=0.35",
                   help="share of episodes per env task, e.g. toss=1 for more toss data only")
    p.add_argument("--videos", type=int, default=1, help="render this many demos per skill on the G1")
    args = p.parse_args()

    demos = load_skill_demos(args.human)
    print(f"Loaded from {args.human}: " + ", ".join(f"{len(v)} {k}" for k, v in demos.items()))
    if args.styles:
        styles = json.loads(args.styles.read_text())
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            styles = throw_styles(demos["toss"], pool=pool, calibrate=not args.no_calibrate)
    for demo, style in zip(demos["toss"], styles):
        hist = style.get("aim_history", [])
        print(f"  {demo.info['name']}: style {style['human']} -> aim {np.round(style['aim_offset'][:2], 3)}, "
              f"calibration success {[h[1] for h in hist]}")

    mix = tuple((k, float(v)) for k, v in (item.split("=") for item in args.task_mix.split(",")))
    data = generate_g1(demos, styles, args.episodes, args.noise, args.clean_fraction, task_mix=mix, seed=args.seed,
                       workers=args.workers)
    summary = save_dataset(data, args.out, styles, {"human": str(args.human), "episodes": args.episodes,
                                                    "noise": args.noise})
    print(json.dumps(summary, indent=2))

    if args.videos:
        vid_dir = args.out / "g1_retarget_videos"
        vid_dir.mkdir(exist_ok=True)
        for skill, items in demos.items():
            for k, demo in enumerate(items[: args.videos]):
                name = demo.info["name"]
                style = styles[k] if skill == "toss" else None
                out = vid_dir / f"{name}.mp4"
                ok = render_skill(demo, style, args.human / f"{name}_overlay.mp4", out)
                print(f"{out} ({'success' if ok else 'fail'})")


if __name__ == "__main__":
    main()
