"""RL on top of the fine-tuned VLA (run in .venv-vla): PPO trains a small residual policy whose output is added to
the VLA's action, final = clip(vla + scale * residual). The VLA stays frozen and keeps planning chunks from the
cameras; the residual sees the robot's observation (object positions as the robot's perception reports them, see
real.py) and the VLA's current action. It starts at zero, so iteration 0 is the VLA alone.

    G1_HAND=shadow G1_REAL=1 MUJOCO_GL=egl python scripts/train_vla_rl.py --vla runs/vla_smolvla --task full \
        --out runs/vla_rl_full
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from robot_arm.humanoid.envs import OBS_STAGE
from robot_arm.humanoid.vla_model import EnvPool, VLAController, load
from robot_arm.policy import ActorCritic


class Residual(ActorCritic):
    """ActorCritic over [observation, VLA action]; the actor's output is a correction to the VLA action."""

    def __init__(self, obs_dim: int, scale: float, hidden: int = 256):
        super().__init__(hidden=hidden, layers=3, obs_dim=obs_dim + 7, act_dim=7, task="g1-vla-residual")
        self.scale = scale


def run_iteration(pool, ctl, model, cur, T, seed_counter, args, device, stats, eval_mode=False):
    """T steps of every env; returns the rollout buffers. `cur` holds each env's latest obs and views."""
    N = len(pool.conns)
    bufs = {k: [] for k in ("x", "res", "logp", "val", "rew", "done")}
    for _ in range(T):
        ids = list(range(N))
        stages = [int(np.argmax(cur[i]["obs"][OBS_STAGE])) for i in ids]
        base = ctl.act(ids, [cur[i]["obs"] for i in ids], [cur[i]["views"] for i in ids], stages)
        base = np.stack([base[i] for i in ids])
        x = torch.as_tensor(np.concatenate([np.stack([cur[i]["obs"] for i in ids]), base], 1), dtype=torch.float32,
                            device=device)
        with torch.no_grad():
            dist = model.dist(x)
            res = dist.mean if eval_mode else dist.sample()
            logp = dist.log_prob(res).sum(-1)
            val = model.value(x)
        act = np.clip(base + model.scale * res.cpu().numpy(), -1, 1)
        for i in ids:
            pool.step(i, act[i])
        rew, done = np.zeros(N, np.float32), np.zeros(N, np.float32)
        for i in ids:
            obs, views, r, term, trunc, info = pool.recv(i)
            # Penalties for disturbing a placed cup: each step the hand touches one, each metre it moves (counted
            # as it happens, so the total is the final shift), and knocking it out of place.
            r -= args.touch_penalty * info["placed_touch"]
            r -= args.shift_penalty * (info["placed_shift"] - cur[i].get("shift", 0.0))
            r -= args.knock_penalty * (info["fail_reason"] == "knocked a placed cup")
            cur[i]["shift"] = info["placed_shift"]
            cur[i]["ret"] += r
            rew[i] = r * args.reward_scale
            if term or trunc:
                if trunc and not term:  # a time limit, not an ending: bootstrap from the last state
                    xf = torch.as_tensor(np.r_[obs, base[i]], dtype=torch.float32, device=device)[None]
                    with torch.no_grad():
                        rew[i] += args.gamma * model.value(xf).item()
                done[i] = 1.0
                stats.append({"success": bool(info["is_success"]), "stage": int(info["stage"]),
                              "reason": info["fail_reason"] or ("ok" if info["is_success"] else "timeout"),
                              "return": cur[i]["ret"], "knocked": info["fail_reason"] == "knocked a placed cup"})
                seed_counter[0] += 1
                pool.reset(i, seed_counter[0])
                obs, views, _ = pool.recv(i)
                ctl.reset(i)
                cur[i]["ret"], cur[i]["shift"] = 0.0, 0.0
            cur[i]["obs"], cur[i]["views"] = obs, views
        for k, v in zip(bufs, (x, res, logp, val, torch.as_tensor(rew, device=device),
                               torch.as_tensor(done, device=device))):
            bufs[k].append(v)
    last_x = torch.as_tensor(np.concatenate([np.stack([cur[i]["obs"] for i in range(N)]),
                                             np.zeros((N, 7), np.float32)], 1), dtype=torch.float32, device=device)
    return {k: torch.stack(v) for k, v in bufs.items()}, last_x


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vla", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--task", default="full", choices=["place", "toss", "full"])
    p.add_argument("--stage", type=int, default=None)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--rollout", type=int, default=256, help="steps per env per iteration")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--scale", type=float, default=0.3, help="largest residual, in action units")
    p.add_argument("--init-std", type=float, default=0.3, help="exploration std of the residual (before scaling)")
    p.add_argument("--min-log-std", type=float, default=-3.0)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--critic-warmup", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatch", type=int, default=1024)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--touch-penalty", type=float, default=0.0, help="per step the hand touches a placed cup")
    p.add_argument("--shift-penalty", type=float, default=0.0, help="per metre a placed cup moves")
    p.add_argument("--knock-penalty", type=float, default=0.0, help="once, for knocking a placed cup out of place")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda"

    vla, stats_vla = load(args.vla, device)
    pool = EnvPool(args.envs, args.task, args.stage)
    N = args.envs
    ctl = VLAController(vla, stats_vla, N, device)
    seed_counter = [1_000_000 + 100_000 * args.seed]
    cur = {}
    for i in range(N):
        seed_counter[0] += 1
        pool.reset(i, seed_counter[0])
    for i in range(N):
        obs, views, _ = pool.recv(i)
        cur[i] = {"obs": obs, "views": views, "ret": 0.0}
    obs_dim = len(cur[0]["obs"])

    model = Residual(obs_dim, args.scale).to(device)
    # Observation statistics from a first rollout of the VLA alone (the residual is zero at the start).
    model.log_std.data.fill_(np.log(args.init_std))
    actor_params = list(model.actor.parameters()) + [model.log_std]
    actor_opt = torch.optim.Adam(actor_params, lr=args.actor_lr)
    critic_opt = torch.optim.Adam(model.critic.parameters(), lr=args.critic_lr)

    episodes, history, best, t0 = [], [], -1.0, time.time()
    for it in range(1, args.iterations + 1):
        b, last_x = run_iteration(pool, ctl, model, cur, args.rollout, seed_counter, args, device, episodes)
        if it == 1:
            x = b["x"].reshape(-1, obs_dim + 7)
            model.set_obs_stats(x.mean(0).cpu().numpy(), x.std(0).cpu().numpy())
            with torch.no_grad():  # recompute what the old policy said under the new normalisation
                d = model.dist(b["x"])
                b["logp"], b["val"] = d.log_prob(b["res"]).sum(-1), model.value(b["x"])
        T = args.rollout
        with torch.no_grad():
            next_v = model.value(last_x)
            adv = torch.zeros_like(b["rew"])
            last = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                nv = next_v if t == T - 1 else b["val"][t + 1]
                nonterm = 1.0 - b["done"][t]
                delta = b["rew"][t] + args.gamma * nv * nonterm - b["val"][t]
                last = delta + args.gamma * args.lam * nonterm * last
                adv[t] = last
            ret = adv + b["val"]
        fx, fres, flogp = b["x"].reshape(T * N, -1), b["res"].reshape(T * N, -1), b["logp"].reshape(-1)
        fadv, fret = adv.reshape(-1), ret.reshape(-1)
        train_actor = it > args.critic_warmup
        st = {"v_loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "clipfrac": 0.0}
        n_up, stop = 0, False
        for _ in range(args.epochs):
            if stop:
                break
            perm = torch.randperm(T * N, device=device)
            for s in range(0, T * N, args.minibatch):
                idx = perm[s:s + args.minibatch]
                v_loss = 0.5 * ((model.value(fx[idx]) - fret[idx]) ** 2).mean()
                critic_opt.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)
                critic_opt.step()
                st["v_loss"] += v_loss.item()
                if train_actor:
                    d = model.dist(fx[idx])
                    logp = d.log_prob(fres[idx]).sum(-1)
                    ratio = (logp - flogp[idx]).exp()
                    a = fadv[idx]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                    actor_opt.zero_grad()
                    pg.backward()
                    torch.nn.utils.clip_grad_norm_(actor_params, 0.5)
                    actor_opt.step()
                    with torch.no_grad():
                        model.log_std.clamp_(min=args.min_log_std)
                        kl = ((ratio - 1) - (logp - flogp[idx])).mean().item()
                    st["pg_loss"] += pg.item()
                    st["kl"] += kl
                    st["clipfrac"] += ((ratio - 1).abs() > args.clip).float().mean().item()
                    stop = kl > args.target_kl
                n_up += 1
                if stop:
                    break
        recent = episodes[-200:]
        row = {"iter": it, "steps": it * T * N, "time_s": round(time.time() - t0),
               "episodes": len(episodes),
               "success": float(np.mean([e["success"] for e in recent])) if recent else None,
               "mean_stage": float(np.mean([e["stage"] for e in recent])) if recent else None,
               "return": float(np.mean([e["return"] for e in recent])) if recent else None,
               "knocked": float(np.mean([e["knocked"] for e in recent])) if recent else None,
               "std": model.log_std.exp().mean().item(), **{k: v / max(n_up, 1) for k, v in st.items()}}
        history.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        model.save(out / "last.pt", vla=str(args.vla), scale=args.scale, task=args.task)
        # Keep the residual whose recent training episodes did best (only after the actor started moving).
        if train_actor and recent and len(recent) >= 50 and row["mean_stage"] + row["success"] > best:
            best = row["mean_stage"] + row["success"]
            model.save(out / "policy.pt", vla=str(args.vla), scale=args.scale, task=args.task, iter=it)
        (out / "history.json").write_text(json.dumps(history, indent=1))
    pool.close()


if __name__ == "__main__":
    main()
