"""Train a chunked policy (state-based Diffusion Policy, or the chunked-MLP baseline) on the retargeted demos.

    G1_HAND=shadow python scripts/train_diffusion.py --task g1-place --data data/processed/shadow_v4/g1_place.npz \
        --out runs/shadow_place_dp --epochs 40
    G1_HAND=shadow python scripts/train_diffusion.py --kind chunk_mlp --data ... --out runs/shadow_place_chunk_mlp

Each sample is the last --obs-horizon observations and the next --horizon actions; the policy executes the first
--n-exec of them and plans again (robot_arm/diffusion.py). Trains on the Apple GPU (mps) when there is one; the
evaluations run in --eval-workers CPU processes, as in train_bc.py. The checkpoint that succeeds most often is
copied to <out>/policy.pt, and loads like any other (evaluate.make_policy: place=<out>/policy.pt).
"""

import argparse
import json

from robot_arm.diffusion import train_chunked
from robot_arm.tasks import TASKS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="g1-place", choices=TASKS)
    p.add_argument("--data", required=True, help="the dataset(s), comma-separated")
    p.add_argument("--out", required=True)
    p.add_argument("--kind", default="diffusion", choices=["diffusion", "chunk_mlp"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--obs-horizon", type=int, default=2)
    p.add_argument("--horizon", type=int, default=16, help="actions predicted per plan (a multiple of 4)")
    p.add_argument("--n-exec", type=int, default=8, help="of those, how many are executed before planning again")
    p.add_argument("--down-dims", default="64,128,256", help="the U-Net's channel widths")
    p.add_argument("--hidden", type=int, default=512, help="the chunk MLP's width")
    p.add_argument("--ddim-steps", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--eval-workers", type=int, default=6)
    p.add_argument("--device", default=None, help="mps / cuda / cpu (default: the best there is)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    result = train_chunked(args.data.split(","), args.out, args.kind, args.task, args.epochs, args.batch_size, args.lr,
                           args.obs_horizon, args.horizon, args.n_exec, tuple(int(d) for d in args.down_dims.split(",")),
                           args.hidden, args.ddim_steps, args.eval_every, args.eval_episodes, args.eval_workers,
                           args.device, args.seed)
    print("best:", json.dumps(result["best"]))


if __name__ == "__main__":
    main()
