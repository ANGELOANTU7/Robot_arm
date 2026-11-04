"""Download the trained controllers, and optionally the VLA and the videos, from Hugging Face.

    python scripts/download_models.py               # the ten controllers in the README tables (66 MB) -> runs/
    python scripts/download_models.py --vla         # also the fine-tuned VLA (1.2 GB) -> runs/vla_smolvla3/
    python scripts/download_models.py --videos      # the phone videos and the robot's tries (1.7 GB) -> data/videos/

Weights: https://huggingface.co/angeloantu/robot-arm-g1-weights
Videos:  https://huggingface.co/datasets/angeloantu/robot-arm-g1-videos
Both are public; files already downloaded are skipped.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = "angeloantu/robot-arm-g1-weights"
VIDEOS = "angeloantu/robot-arm-g1-videos"
CONTROLLERS = ["runs/chunk_6k_elbow/policy.pt", "runs/dp_6k_elbow/policy.pt", "runs/toss4k_dp_elbow/policy.pt",
               "runs/toss4k_ppo_small/policy_iter40.pt", "runs/toss4k_bc/policy.pt",
               "runs/real_place_chunk/policy.pt", "runs/real_place_ppo/policy.pt", "runs/real2_toss_dp/policy.pt",
               "runs/real2_toss_ppo/policy.pt", "runs/real2_toss_bc/policy.pt"]
VLA = "runs/vla_smolvla3/*"


def download_controllers(paths: list[str] | None = None, vla: bool = False) -> None:
    """Fetch these checkpoints (default: all ten controllers) into the repo, at the paths the code expects."""
    from huggingface_hub import snapshot_download

    patterns = list(paths or CONTROLLERS) + ([VLA] if vla else [])
    snapshot_download(WEIGHTS, allow_patterns=patterns, local_dir=ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vla", action="store_true", help="also the fine-tuned VLA (1.2 GB)")
    p.add_argument("--videos", action="store_true", help="also the videos, into data/videos/")
    p.add_argument("--only-videos", action="store_true", help="the videos and nothing else")
    args = p.parse_args()
    if not args.only_videos:
        download_controllers(vla=args.vla)
        print(f"Controllers in {ROOT / 'runs'}")
    if args.videos or args.only_videos:
        from huggingface_hub import snapshot_download

        snapshot_download(VIDEOS, repo_type="dataset", local_dir=ROOT / "data" / "videos")
        print(f"Videos in {ROOT / 'data' / 'videos'}")


if __name__ == "__main__":
    main()
