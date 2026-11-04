"""Image demos for the VLA: the retargeted experts on random layouts, filmed by the robot's head and wrist cameras.

    G1_HAND=shadow G1_REAL=1 MUJOCO_GL=egl python scripts/g1_vla_data.py --out data/processed/vla \
        --kept data/processed/real3/g1_kept_demos.json --styles data/processed/real_toss_styles.json \
        --place-episodes 8000 --toss-episodes 12000 --workers 26

Place episodes (whole runs that stop once the pyramid stands, and single cups) and throw episodes (from a scripted
pyramid) are written as shards of completed stages; see robot_arm/humanoid/vla.py.
"""

import argparse
import json
from pathlib import Path

from robot_arm.humanoid.dataset import g1_jobs, load_skill_demos
from robot_arm.humanoid.vla import generate_vla, save_summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--human", type=Path, default=Path("data/processed/dataset2"))
    p.add_argument("--kept", type=Path, required=True, help="g1_kept_demos.json of an earlier run (reproducible demos)")
    p.add_argument("--styles", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/processed/vla"))
    p.add_argument("--place-episodes", type=int, default=8000)
    p.add_argument("--toss-episodes", type=int, default=12000)
    p.add_argument("--place-noise", type=float, default=0.15)
    p.add_argument("--toss-noise", type=float, default=0.08)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    demos = load_skill_demos(args.human)
    kept = json.loads(args.kept.read_text())
    demos = {k: [d for d in v if d.info["name"] in kept.get(k, [])] for k, v in demos.items() if k != "rejected"}
    styles = json.loads(args.styles.read_text())
    print("Demos: " + ", ".join(f"{len(v)} {k}" for k, v in demos.items()) + f"; {len(styles)} throw styles")

    summary = {}
    place, _ = g1_jobs(demos, styles, args.place_episodes, args.place_noise, 0.2, (("full", 0.6), ("place", 0.4)),
                       args.seed, place_only=True)
    toss, _ = g1_jobs(demos, styles, args.toss_episodes, args.toss_noise, 0.4, (("toss", 1.0),), args.seed + 1,
                      place_only=False)
    summary["place"] = generate_vla(place, args.out, args.workers)
    summary["toss"] = generate_vla(toss, args.out, args.workers, first_episode=len(place))
    summary.update(human=str(args.human), styles=str(args.styles))
    save_summary(args.out, summary)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
