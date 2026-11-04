"""Fine-tune SmolVLA on the G1 image demos (run in .venv-vla).

    G1_HAND=shadow G1_REAL=1 MUJOCO_GL=egl python scripts/train_vla.py --data data/processed/vla --out runs/vla_smolvla

Every --eval-every steps the policy runs closed loop in the sim (each placement stage and the throw) and the best
checkpoint by mean success is kept in --out; the latest is in --out/last.
"""

import argparse
import json
import math
import time

import numpy as np
import torch

from robot_arm.humanoid.vla_model import (IMAGE_KEYS, PRETRAINED, Tokens, VLAData, build_policy, evaluate,
                                          make_config, save, to_batch)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="VLA shard folder(s), comma-separated")
    p.add_argument("--out", required=True)
    p.add_argument("--init", default=PRETRAINED)
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--n-exec", type=int, default=8)
    p.add_argument("--res", type=int, default=384, help="image size for the vision encoder")
    p.add_argument("--state", default="robot", choices=["robot", "perceived"],
                   help="robot: joints and hand only; perceived: also hand-to-object and object-to-target")
    p.add_argument("--train-vlm", action="store_true", help="also train the VLM (default: action expert only)")
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--eval-envs", type=int, default=24)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"

    data = VLAData(args.data, args.chunk, state=args.state)
    eps = np.unique(data.episode)
    rng = np.random.default_rng(args.seed)
    val_eps = set(rng.choice(eps, max(1, len(eps) // 20), replace=False).tolist())
    is_val = np.isin(data.episode, list(val_eps))
    train_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    # Every stage gets an equal share of the batches, so the few top-cup and throw demos aren't drowned out.
    stage = data.d["stage"][train_idx]
    weights = np.zeros(len(train_idx))
    for s in np.unique(stage):
        weights[stage == s] = 1.0 / (stage == s).sum()
    print(f"{len(data)} samples, {len(eps)} episodes; train by stage "
          f"{ {int(s): int((stage == s).sum()) for s in np.unique(stage)} }", flush=True)
    sampler = torch.utils.data.WeightedRandomSampler(weights, args.steps * args.batch, generator=torch.Generator()
                                                     .manual_seed(args.seed))
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, train_idx), batch_size=args.batch,
                                         sampler=sampler, num_workers=args.workers, pin_memory=True,
                                         persistent_workers=True, prefetch_factor=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, val_idx[rng.permutation(len(val_idx))
                                                                                   [:2048]]),
                                             batch_size=args.batch, num_workers=4)

    cfg, _ = make_config(args.chunk, args.n_exec, args.res, args.state, train_expert_only=not args.train_vlm)
    policy = build_policy(cfg, args.init).to(dev)
    policy.train()
    tokens = Tokens(cfg, dev)
    params = [q for q in policy.parameters() if q.requires_grad]
    print(f"trainable {sum(q.numel() for q in params) / 1e6:.1f}M of "
          f"{sum(q.numel() for q in policy.parameters()) / 1e6:.1f}M parameters", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * (
        0.025 + 0.975 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    def val_loss():
        policy.eval()
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in val_loader:
                losses.append(policy.forward(to_batch(b, tokens, dev))[0].item())
        policy.train()
        return float(np.mean(losses))

    best, log, t0, run = -1.0, [], time.time(), []
    for step, b in enumerate(loader, 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(to_batch(b, tokens, dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if step % 200 == 0:
            print(f"step {step} loss {np.mean(run):.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / step:.3f}s/step", flush=True)
            run = []
        if step % args.eval_every == 0 or step == args.steps:
            entry = {"step": step, "val_loss": val_loss()}
            policy.eval()
            scores = {}
            for name, task, st in (("cup0", "place", 0), ("cup1", "place", 1), ("cup2", "place", 2),
                                   ("throw", "toss", None)):
                r = evaluate(policy, {**data.stats, "state": args.state}, task, st, args.eval_episodes, args.eval_envs, seed=50_000)
                scores[name] = r["success_rate"]
                entry[name] = r
            policy.train()
            entry["mean"] = float(np.mean(list(scores.values())))
            log.append(entry)
            print(f"EVAL step {step} val {entry['val_loss']:.4f} " +
                  " ".join(f"{k} {v:.2f}" for k, v in scores.items()), flush=True)
            save(policy, data.stats, f"{args.out}/last", args.state, step=step, scores=scores)
            if entry["mean"] > best:
                best = entry["mean"]
                save(policy, data.stats, args.out, args.state, step=step, scores=scores)
            with open(f"{args.out}/train_log.json", "w") as f:
                json.dump(log, f, indent=1, default=str)
        if step >= args.steps:
            break


if __name__ == "__main__":
    main()
