"""Videos of every try: with the TRY_VIDEOS environment variable set to a folder, each G1 episode run by the
evaluation, the dataset generation and the experiment scripts is saved there as <name>_<seed>_<outcome>.mp4.

The format is that of scripts/g1_record_attempts.py: side and front views side by side (320x240 each), 10 frames
per second of simulated time, with the stage, the time and finally the outcome written on it.
"""

import os
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

STAGE_NAMES = ["place cup A", "place cup B", "turn over + place top cup", "throw the cap", "thumbs-up"]
WIDTH, HEIGHT = 320, 240


def try_dir() -> Path | None:
    folder = os.environ.get("TRY_VIDEOS")
    return Path(folder) if folder else None


def render_kwargs() -> dict:
    """G1PyramidEnv keyword arguments for an env whose tries are recorded (none when TRY_VIDEOS is unset)."""
    return {"render_mode": "rgb_array", "width": WIDTH, "height": HEIGHT} if try_dir() else {}


class TryRecorder:
    """Collects the frames of one episode and writes them when it ends. Does nothing without TRY_VIDEOS."""

    def __init__(self, env, name: str, label: str = "", every: int = 2):
        self.env, self.name, self.label, self.every = env, name, label, every
        self.dir = try_dir()
        self.frames: list[np.ndarray] = []
        if self.dir is not None and env.render_mode is None:
            raise ValueError("Create the env with trylog.render_kwargs() to record its tries.")

    def _frame(self, text: str) -> np.ndarray:
        env, cam = self.env, self.env.camera
        tiles = []
        for view in ("side", "front"):
            env.camera = view
            tiles.append(env.render().copy())
        env.camera = cam
        img = np.concatenate(tiles, axis=1)
        cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def step(self, t: int) -> None:
        if self.dir is None or t % self.every:
            return
        env = self.env
        self.frames.append(self._frame(f"{self.label} | {STAGE_NAMES[env.stage]} | t={t * env.cfg.control_dt:4.1f}s"
                                       f" | misses {env.misses}"))

    def finish(self, info: dict, seed: int) -> Path | None:
        if self.dir is None:
            return None
        outcome = "success" if info.get("is_success") else (info.get("fail_reason") or "timeout").replace(" ", "-")
        last = self._frame(f"{self.label} | {outcome.upper()} (reached: {STAGE_NAMES[self.env.stage]})")
        self.frames += [last] * 10  # the outcome, held for a second
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{self.name}_{seed}_{outcome}.mp4"
        imageio.mimsave(path, self.frames, fps=10, macro_block_size=1, quality=6)
        self.frames = []
        return path
