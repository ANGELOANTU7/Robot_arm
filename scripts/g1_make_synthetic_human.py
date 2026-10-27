"""Write synthetic human skill demos for the cup pyramid, to try the whole pipeline before recording.

    python scripts/g1_make_synthetic_human.py --n 10      # -> data/processed/human_pyramid_synthetic/

Each demo is a scripted human motion (place a base cup, place the top cup, throw the cap) rendered into what the
trackers would see from a camera and reconstructed by the same code as real clips, so it carries realistic errors.
"""

import argparse
from pathlib import Path

import numpy as np

from robot_arm.humanoid.synthetic import synthetic_skill_demos
from robot_arm.video.skill_process import save_skill_plot


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10, help="demos per skill")
    p.add_argument("--out", type=Path, default=Path("data/processed/human_pyramid_synthetic"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plots", type=int, default=1, help="trajectory plots for the first few of each skill")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    demos = synthetic_skill_demos(np.random.default_rng(args.seed), args.n)
    for i, (name, demo) in enumerate(demos):
        demo.save(args.out / f"{name}.npz")
        if i < 3 * args.plots:
            save_skill_plot(demo, args.out / f"{name}_trajectory.png")
    print(f"Wrote {len(demos)} synthetic skill demos to {args.out}")


if __name__ == "__main__":
    main()
