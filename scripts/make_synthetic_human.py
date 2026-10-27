"""Write synthetic human demonstrations, to try the whole pipeline before recording real videos.

    python scripts/make_synthetic_human.py --n 15      # -> data/processed/human_synthetic/
"""

import argparse
from pathlib import Path

import numpy as np

from robot_arm.synthetic import synthetic_demo
from robot_arm.video.process import save_plot


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--out", type=Path, default=Path("data/processed/human_synthetic"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plots", type=int, default=3, help="also save trajectory plots for the first few")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        demo = synthetic_demo(rng)
        demo.save(args.out / f"synthetic_{i:03d}.npz")
        if i < args.plots:
            save_plot(demo, args.out / f"synthetic_{i:03d}_trajectory.png")
    print(f"Wrote {args.n} synthetic demos to {args.out}")


if __name__ == "__main__":
    main()
