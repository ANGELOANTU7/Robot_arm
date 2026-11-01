"""Track both hands in the clips at full resolution, once, for scripts/g1_process_videos.py --hands.

    python scripts/g1_track_landmarks.py ../dataset_2/videos/*/*.MOV --out data/landmarks --workers 16

Writes <out>/<folder>_<clip>.npz with hand_px (T, 2, 21, 2), in pixels of the clip, and hand_world (T, 2, 21, 3): two
continuous hand tracks, see skill_process.track_two_hands.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from robot_arm.video.process import read_video
from robot_arm.video.skill_process import track_two_hands
from robot_arm.video.tracking import ensure_hand_model


def track_clip(video: Path, out: Path) -> str:
    target = out / f"{video.parent.name}_{video.stem}.npz"
    if not target.exists():
        frames, fps = read_video(video, 100000)
        np.savez_compressed(target, **track_two_hands(frames, fps), fps=fps)
    return video.name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/landmarks"))
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ensure_hand_model()
    with ProcessPoolExecutor(args.workers) as pool:
        for name in pool.map(partial(track_clip, out=args.out), args.videos):
            print(name, flush=True)


if __name__ == "__main__":
    main()
