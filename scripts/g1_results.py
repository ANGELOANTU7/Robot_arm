"""Results table for the G1 cup pyramid: every policy on each place stage, the toss (re-throws after misses, then
the thumbs-up) and the full task.

    python scripts/g1_results.py --replay data/processed/human_pyramid_synthetic \
        --bc runs/g1_place_bc/policy.pt runs/g1_toss_bc/policy.pt \
        --ppo runs/g1_place_ppo/policy.pt runs/g1_toss_ppo/policy.pt --episodes 50

Writes runs/g1_results.json and runs/g1_results.md. All policies see the same layouts (fixed seeds, disjoint
from the ones used for training and checkpoint selection).
"""

import argparse
import json
from pathlib import Path

from robot_arm.humanoid.evaluate import evaluate_g1

COLUMNS = [("place A", "place", 0), ("place B", "place", 1), ("place top", "place", 2), ("toss", "toss", None),
           ("full task", "full", None)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay", type=Path, help="folder of human skill demos (retargeted replay, no learning)")
    p.add_argument("--bc", nargs=2, metavar=("PLACE", "TOSS"), help="BC checkpoints")
    p.add_argument("--ppo", nargs=2, metavar=("PLACE", "TOSS"), help="BC+PPO checkpoints")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--no-scripted", action="store_true", help="leave out the scripted experts' row")
    p.add_argument("--append", action="store_true", help="add the rows to the existing g1_results.json")
    args = p.parse_args()

    policies = [] if args.no_scripted else [("scripted experts", "scripted")]
    if args.replay:
        policies.append(("retargeted human demos (replay)", f"replay:{args.replay}"))
    if args.bc:
        policies.append(("BC", f"place={args.bc[0]},toss={args.bc[1]}"))
    if args.ppo:
        policies.append(("BC + PPO", f"place={args.ppo[0]},toss={args.ppo[1]}"))

    results = {}
    if args.append and (args.out / "g1_results.json").exists():
        results = json.loads((args.out / "g1_results.json").read_text())
    for name, spec in policies:
        row = {}
        for col, task, stage in COLUMNS:
            stats = evaluate_g1(task, spec, args.episodes, stage, seed=90_000, workers=args.workers)
            row[col] = stats
            print(f"{name:34s} {col:10s} {stats['success_rate']:.2f} {stats['outcomes']}", flush=True)
        results[name] = row

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "g1_results.json").write_text(json.dumps(results, indent=1))
    lines = ["| Policy | " + " | ".join(c for c, _, _ in COLUMNS) + " |",
             "|---|" + "---|" * len(COLUMNS)]
    for name, row in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{100 * row[c]['success_rate']:.0f}%" for c, _, _ in COLUMNS)
                     + " |")
    table = "\n".join(lines) + f"\n\n{args.episodes} episodes per cell. Missed throws per toss episode (each followed by" \
        " a re-throw): " + ", ".join(f"{name} {row['toss']['misses_per_episode']:.2f}" for name, row in results.items()) + ".\n"
    (args.out / "g1_results.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
