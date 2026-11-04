"""Score a fine-tuned VLA closed loop (run in .venv-vla), optionally with an RL residual on top.

    G1_HAND=shadow G1_REAL=1 MUJOCO_GL=egl python scripts/eval_vla.py --vla runs/vla_smolvla
    ... --vla runs/vla_smolvla3 --residual runs/vla_rl_cup1/policy.pt --only cup1
"""

import argparse
import json

import torch

from robot_arm.humanoid.vla_model import evaluate, load
from robot_arm.policy import ActorCritic

SUITE = {"cup0": ("place", 0, 48), "cup1": ("place", 1, 48), "cup2": ("place", 2, 48), "throw": ("toss", None, 96),
         "whole": ("full", None, 96)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vla", required=True)
    p.add_argument("--only", default=",".join(SUITE), help="comma-separated subset of " + ",".join(SUITE))
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--n-exec", type=int, default=None, help="actions run from each chunk (default: as trained)")
    p.add_argument("--residual", default=None, help="policy.pt from scripts/train_vla_rl.py, added to the VLA")
    args = p.parse_args()
    policy, stats = load(args.vla, n_exec=args.n_exec)
    residual = None
    if args.residual:
        ckpt = torch.load(args.residual, map_location="cpu", weights_only=False)
        residual = (ActorCritic.load(args.residual, "cuda").eval(), ckpt["scale"])
    for name in args.only.split(","):
        task, stage, n = SUITE[name]
        r = evaluate(policy, stats, task, stage, n, args.envs, residual=residual)
        print(name, json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
