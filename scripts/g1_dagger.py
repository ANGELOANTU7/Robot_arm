"""DAgger for the G1 pyramid: run the learned policy, label every state it visits with the retargeted expert (your
demos and throw styles, as in g1_retarget_demos.py), add those to the data and train again.

    python scripts/g1_dagger.py --task g1-place --human data/processed/human_pyramid \
        --data data/processed/shadow/g1_place.npz --init runs/shadow_place_bc/policy.pt --out runs/shadow_place_dagger

Behaviour cloning only sees the states the expert visits; once the learned policy drifts a little (a cup held a few
millimetres off, a hand a little late) it is somewhere it was never shown what to do, and the errors add up. DAgger
shows it exactly those states. Each round: `--episodes` rollouts of the current policy (at each step the expert's
action instead with probability beta, which decays), then BC on all the data so far, keeping the checkpoint that
succeeds most often.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from robot_arm.bc import train_bc
from robot_arm.humanoid.dataset import generate_dagger, load_skill_demos, throw_styles


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="g1-place", choices=["g1-place", "g1-toss"])
    p.add_argument("--human", type=Path, default=Path("data/processed/human_pyramid"))
    p.add_argument("--kept", type=Path, help="g1_kept_demos.json of the datasets (default: next to --data)")
    p.add_argument("--styles", type=Path, help="g1_styles.json (default: next to --data)")
    p.add_argument("--data", required=True, help="the BC dataset(s) to start from, comma-separated")
    p.add_argument("--init", required=True, help="the policy for the first round of rollouts")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--beta", type=float, default=0.3, help="probability of the expert's action at a step, round 1")
    p.add_argument("--beta-decay", type=float, default=0.5)
    p.add_argument("--task-mix", default="full=0.5,place=0.5")
    p.add_argument("--place-only", action="store_true", help="end full episodes once the pyramid stands (no throw)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args()

    data_dir = Path(args.data.split(",")[0]).parent
    demos = load_skill_demos(args.human)
    demos.pop("rejected")
    kept = args.kept or data_dir / "g1_kept_demos.json"
    if kept.exists():
        names = {n for v in json.loads(kept.read_text()).values() for n in v}
        demos = {k: [d for d in v if d.info["name"] in names] for k, v in demos.items()}
    styles_path = args.styles or data_dir / "g1_styles.json"
    styles = json.loads(styles_path.read_text()) if styles_path.exists() else throw_styles(demos["toss"],
                                                                                          calibrate=False)
    mix = tuple((k, float(v)) for k, v in (item.split("=") for item in args.task_mix.split(",")))
    key = "place" if args.task == "g1-place" else "toss"
    args.out.mkdir(parents=True, exist_ok=True)
    datasets, policy, beta = args.data.split(","), args.init, args.beta
    for r in range(1, args.rounds + 1):
        ckpts = (policy, None) if key == "place" else (None, policy)
        data = generate_dagger(demos, styles, *ckpts, args.episodes, beta, mix, seed=args.seed + r,
                               workers=args.workers, place_only=args.place_only)
        reached = np.bincount(data["reached"], minlength=6)
        print(f"round {r}: beta {beta:.2f}, rollouts reached stage (0-4, 5 = done): {reached.tolist()}", flush=True)
        path = args.out / f"round{r}_{key}.npz"
        np.savez_compressed(path, **data[key])
        datasets.append(str(path))
        result = train_bc(",".join(datasets), args.out / f"round{r}", epochs=args.epochs, task=args.task,
                          eval_every=args.eval_every, eval_episodes=args.eval_episodes, eval_workers=args.workers,
                          seed=r)
        print(f"round {r}: best {result['best']}", flush=True)
        policy, beta = str(args.out / f"round{r}" / "policy.pt"), beta * args.beta_decay
    print("final policy:", policy)


if __name__ == "__main__":
    main()
