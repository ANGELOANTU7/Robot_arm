"""Watch the G1's tries live in a browser, and score them as an evaluation at the same time.

    python scripts/g1_live.py serve                     # the page, on http://localhost:8765 (this machine only)
    MUJOCO_GL=egl python scripts/g1_live.py run --name final --jobs place0:32,place1:32,place2:32,toss:32,full:96 \
        --policy place=runs/chunk_2k/policy.pt,toss=runs/toss4k_dp_e80/policy.pt

The two parts are separate processes that share files: each worker keeps its newest frame in /dev/shm/g1live and every
finished try is one line in logs/live/<name>.jsonl. A try already in that file is skipped, so a run can be restarted.
Seeds match g1_evaluate.py (20000 upward). The first few tries of each job are also saved to --videos.

    MUJOCO_GL=egl python scripts/g1_live.py watch --policy place=runs/chunk_2k/policy.pt,toss=runs/x/candidate.pt \\
        --train-log logs/t_ppo_x.log

plays tries at real speed instead (`run` goes as fast as it can, too fast to follow by eye) and takes the checkpoint
again for each try, so during training the tiles show the policy as it improves.
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import get_context
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

SHM = Path("/dev/shm/g1live")
LIVE = Path("logs/live")
STAGE_NAMES = ["place cup A", "place cup B", "turn over + place top cup", "throw the cap", "thumbs-up"]


def parse_jobs(text: str) -> list[tuple[str, int]]:
    return [(a, int(b)) for a, b in (item.split(":") for item in text.split(","))]


def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


# Runner

def try_once(cfg: dict, slot: int, job: str, seed: int, keep_video: bool, pace: float = 0.0, round_: int | None = None) -> dict:
    from robot_arm.humanoid.envs import G1PyramidEnv
    from robot_arm.humanoid.evaluate import make_policy

    task = job.rstrip("012")
    stage = int(job[-1]) if job[-1].isdigit() else None
    env = G1PyramidEnv(task=task, stage=stage, render_mode="rgb_array", camera=cfg["camera"], width=480, height=360)
    policy = make_policy(cfg["policy"])
    obs, _ = env.reset(seed=seed)
    policy.reset()

    def show(t: int, final: dict | None = None) -> np.ndarray:
        img = env.render().copy()
        atomic_write(SHM / f"{slot}.jpg", cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes())
        atomic_write(SHM / f"{slot}.json", json.dumps({
            "job": job, "seed": seed, "stage": STAGE_NAMES[env.stage], "t": round(t * env.cfg.control_dt, 1),
            "limit": round(env.max_steps * env.cfg.control_dt), "misses": env.misses, "final": final,
            "round": round_}).encode())
        return img

    frames = []
    for t in range(env.max_steps):
        began = time.time()
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if t % 2 == 0:
            img = show(t)
            if keep_video:
                frames.append(img)
        if terminated or truncated:
            break
        time.sleep(max(0.0, pace - (time.time() - began)))
    outcome = "ok" if info["is_success"] else info["fail_reason"] or "timeout"
    result = {"job": job, "seed": seed, "ok": bool(info["is_success"]), "outcome": outcome, "reached": int(info["stage"]),
              "misses": int(info["misses"]), "seconds": round(t * env.cfg.control_dt, 1)}
    show(t, final=outcome)
    if keep_video:
        Path(cfg["videos"]).mkdir(parents=True, exist_ok=True)
        imageio.mimsave(Path(cfg["videos"]) / f"{job}_{seed}_{outcome.replace(' ', '-')}.mp4", frames + frames[-1:] * 10,
                        fps=10, macro_block_size=1, quality=6)
    env.close()
    return result


def worker(cfg: dict, slot: int, queue, lock) -> None:
    while True:
        item = queue.get()
        if item is None:
            return
        job, seed, keep_video = item
        result = try_once(cfg, slot, job, seed, keep_video)
        with lock, open(cfg["log"], "a") as f:
            f.write(json.dumps(result) + "\n")


def run(args) -> None:
    SHM.mkdir(exist_ok=True)
    LIVE.mkdir(parents=True, exist_ok=True)
    for old in SHM.iterdir():
        old.unlink()
    log = LIVE / f"{args.name}.jsonl"
    done = {(r["job"], r["seed"]) for r in map(json.loads, log.read_text().splitlines())} if log.exists() else set()
    plan = parse_jobs(args.jobs)
    (LIVE / f"{args.name}.plan.json").write_text(json.dumps({"policy": args.policy, "jobs": plan}))
    log.touch()
    cfg = {"policy": args.policy, "camera": args.camera, "videos": args.videos, "log": str(log)}
    ctx = get_context("spawn")
    queue, lock = ctx.Queue(), ctx.Lock()
    for job, count in plan:
        for i in range(count):
            if (job, args.seed + i) not in done:
                queue.put((job, args.seed + i, i < args.keep))
    for _ in range(args.workers):
        queue.put(None)
    workers = [ctx.Process(target=worker, args=(cfg, slot, queue, lock)) for slot in range(args.workers)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    print(json.dumps(summarise(args.name)["jobs"], indent=1))


def latest_round(train_log: str | None) -> int | None:
    """The newest round with a test, which is the round the checkpoint being followed comes from."""
    if not train_log or not Path(train_log).exists():
        return None
    rows = [json.loads(line) for line in Path(train_log).read_text().splitlines() if '"eval_success"' in line]
    return rows[-1]["iter"] if rows else 0


def watcher(cfg: dict, slot: int) -> None:
    """Plays tries forever at real speed. Every try takes the newest checkpoint, so the tiles improve as training does."""
    from robot_arm.humanoid.envs import G1_CFG

    seed = 30_000 + slot * 100_000
    while True:
        seed += 1
        job = cfg["jobs"][slot % len(cfg["jobs"])]
        try:
            result = try_once(cfg, slot, job, seed, False, G1_CFG.control_dt, latest_round(cfg["train_log"]))
        except (EOFError, RuntimeError, OSError):  # the checkpoint was being rewritten
            time.sleep(2)
            continue
        with open(LIVE / "watch.ndjson", "a") as f:
            f.write(json.dumps({**result, "round": latest_round(cfg["train_log"])}) + "\n")
        time.sleep(2)  # leave the outcome on screen


def watch(args) -> None:
    SHM.mkdir(exist_ok=True)
    LIVE.mkdir(parents=True, exist_ok=True)
    for old in list(SHM.iterdir()) + [LIVE / "watch.ndjson"]:
        old.unlink(missing_ok=True)
    cfg = {"policy": args.policy, "camera": args.camera, "jobs": args.jobs.split(","), "train_log": args.train_log}
    ctx = get_context("spawn")
    workers = [ctx.Process(target=watcher, args=(cfg, slot)) for slot in range(args.workers)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()


# Page

def summarise(name: str) -> dict:
    log, plan = LIVE / f"{name}.jsonl", json.loads((LIVE / f"{name}.plan.json").read_text())
    tries = [json.loads(line) for line in log.read_text().splitlines()]
    jobs = []
    for job, total in plan["jobs"]:
        rows = [r for r in tries if r["job"] == job]
        jobs.append({"job": job, "total": total, "done": len(rows), "ok": sum(r["ok"] for r in rows),
                     "outcomes": Counter(r["outcome"] for r in rows), "reached": Counter(str(r["reached"]) for r in rows),
                     "tries": [[r["seed"], r["outcome"]] for r in rows]})
    return {"name": name, "policy": plan["policy"], "jobs": jobs}


def curves() -> dict:
    """Training curves from every RL log: success while training, and the fixed-layout tests when there was one."""
    out = {}
    for path in sorted(Path("logs").glob("*ppo*.log")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{"iter"')]
        if rows:
            out[path.stem] = {"train": [[r["iter"], r["success"]] for r in rows],
                              "test": [[r["iter"], r["eval_success"]] for r in rows if "eval_success" in r]}
    return out


def scores() -> list:
    path = Path("logs/throw_summary.txt")
    rows = [re.match(r"(.+?)\s+(\d+)/96", line) for line in path.read_text().splitlines()] if path.exists() else []
    return [[m[1], int(m[2])] for m in rows if m]


def practice() -> dict | None:
    path = LIVE / "watch.ndjson"
    tries = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if not tries:
        return None
    last = tries[-40:]
    return {"tries": len(tries), "last": len(last), "ok": sum(r["ok"] for r in last), "round": tries[-1]["round"],
            "recent": [[r["seed"], r["outcome"]] for r in last]}


def status() -> dict:
    slots = []
    for path in sorted(SHM.glob("*.json"), key=lambda p: int(p.stem)) if SHM.exists() else []:
        try:
            slots.append({"slot": int(path.stem), "age": round(time.time() - path.stat().st_mtime, 1),
                          **json.loads(path.read_text())})
        except (OSError, ValueError):  # the worker replaced it mid-read
            pass
    logs = sorted(LIVE.glob("*.jsonl"), key=lambda p: p.stat().st_mtime) if LIVE.exists() else []
    return {"slots": slots, "run": summarise(logs[-1].stem) if logs else None, "practice": practice(), "curves": curves(),
            "scores": scores()}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send(self, body: bytes, kind: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        frame = re.fullmatch(r"/frame/(\d+)", path)
        if path == "/":
            self.send(Path(__file__).with_suffix(".html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/status":
            self.send(json.dumps(status()).encode(), "application/json")
        elif frame and (SHM / f"{frame[1]}.jpg").exists():
            self.send((SHM / f"{frame[1]}.jpg").read_bytes(), "image/jpeg")
        else:
            self.send(b"", "text/plain", 404)

    def log_message(self, *_):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8765)
    r = sub.add_parser("run")
    r.add_argument("--name", default="live")
    r.add_argument("--jobs", default="full:16", help="job:tries,... where job is place0, place1, place2, toss or full")
    r.add_argument("--policy", default="place=runs/chunk_2k/policy.pt,toss=runs/toss4k_dp_e80/policy.pt")
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--seed", type=int, default=20_000)
    r.add_argument("--camera", default="side", choices=["front", "side", "over"])
    r.add_argument("--keep", type=int, default=3, help="tries of each job saved as video")
    r.add_argument("--videos", default="try_videos/26_final_evaluation")
    w = sub.add_parser("watch", help="tries at real speed, not scored as an evaluation")
    w.add_argument("--jobs", default="full,toss", help="comma list, shared out across the tiles")
    w.add_argument("--policy", default="place=runs/chunk_2k/policy.pt,toss=runs/toss4k_dp_e80/policy.pt")
    w.add_argument("--train-log", help="a training log, to show which round the checkpoint in --policy comes from")
    w.add_argument("--workers", type=int, default=6)
    w.add_argument("--camera", default="side", choices=["front", "side", "over"])
    args = p.parse_args()
    if args.command == "serve":
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    elif args.command == "run":
        run(args)
    else:
        watch(args)


if __name__ == "__main__":
    main()
