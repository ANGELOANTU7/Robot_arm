"""Train a behavior-cloning policy on the retargeted robot demonstrations.

    python scripts/train_bc.py --data data/processed/robot_demos.npz --out runs/bc
    python scripts/evaluate.py --policy runs/bc/policy.pt --videos runs/bc/videos

    # G1 cup pyramid: one policy per skill
    python scripts/train_bc.py --task g1-place --data data/processed/g1_place.npz --out runs/g1_place_bc
    python scripts/train_bc.py --task g1-toss --data data/processed/g1_toss.npz --out runs/g1_toss_bc
"""

import argparse
import json

from robot_arm.bc import train_bc
from robot_arm.tasks import TASKS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="panda", choices=TASKS)
    p.add_argument("--data", default="data/processed/robot_demos.npz")
    p.add_argument("--out", default="runs/bc")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--eval-workers", type=int, default=4, help="processes for the G1 evaluations")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    result = train_bc(args.data, args.out, args.epochs, args.batch_size, args.lr, args.hidden, args.layers,
                      args.eval_every, args.eval_episodes, device=args.device, seed=args.seed, task=args.task,
                      eval_workers=args.eval_workers)
    print("best:", json.dumps(result["best"]))


if __name__ == "__main__":
    main()
