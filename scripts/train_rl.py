"""Fine-tune the BC policy with PPO (plus a decaying BC term on the demos).

    python scripts/train_rl.py --init runs/bc/policy.pt --demos data/processed/robot_demos.npz --out runs/ppo
    python scripts/evaluate.py --policy runs/ppo/policy.pt --videos runs/ppo/videos

    # G1 cup pyramid (the task comes from the checkpoint)
    python scripts/train_rl.py --init runs/g1_toss_bc/policy.pt --demos data/processed/g1_toss.npz --out runs/g1_toss_ppo

Simulation runs on CPU worker processes (--num-envs, roughly one per core); the networks train on the GPU.
"""

import argparse
import json
from dataclasses import fields

from robot_arm.ppo import PPOConfig, train_ppo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", default="runs/bc/policy.pt", help="BC checkpoint to start from")
    p.add_argument("--demos", default="data/processed/robot_demos.npz", help="demos for the BC term ('' = none)")
    p.add_argument("--out", default="runs/ppo")
    p.add_argument("--device", default=None)
    for f in fields(PPOConfig):
        p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = p.parse_args()
    cfg = PPOConfig(**{f.name: getattr(args, f.name) for f in fields(PPOConfig)})
    result = train_ppo(args.init, args.demos or None, args.out, cfg, device=args.device)
    print(json.dumps({"initial_eval": result["initial_eval"], "best_eval": result["best_eval"]}))


if __name__ == "__main__":
    main()
