"""Set up and watch one G1 try in the browser: pick a controller for each task, arrange the cups on the table,
choose the real-world limits, then watch the robot live and move the camera around.

    python scripts/g1_gui.py                       # opens http://localhost:8766
    python scripts/g1_gui.py --host 0.0.0.0 --no-browser --port 8766   # in Docker (publish the port)

The page is scripts/g1_gui.html. Each run is one simulator process, started with the hand (Shadow) and the
real-world limits (G1_REAL, G1_REAL_SET) chosen on the page, since both are read when the simulator is imported.
It streams JPEG frames back, rendered from the camera the page asks for. Trained controllers are loaded from
runs/; a controller whose checkpoint isn't there is shown but can't be picked.
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("G1_HAND", "shadow")  # every trained controller here is for the five-fingered hand
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # no screen: render off-screen (the Docker image sets osmesa)

ROOT = Path(__file__).resolve().parent.parent
TASKS = ["Cup A", "Cup B", "Top cup", "Throw"]
STAGE_NAMES = ["Placing cup A", "Placing cup B", "Turning over and placing the top cup", "Throwing the cap",
               "Thumbs-up"]
W, H = 800, 560

# Success rates from the README's comparison tables (48 tries per cup, 96 per throw).
CONTROLLERS = {
    "perfect": {
        "cups": [("imitation", "Imitation (chunked MLP)", "runs/chunk_6k_elbow/policy.pt", [100, 100, 79]),
                 ("diffusion", "Diffusion policy", "runs/dp_6k_elbow/policy.pt", [100, 85, 44]),
                 ("scripted", "Hand-written", "scripted", [100, 100, 88])],
        "throw": [("diffusion", "Diffusion policy", "runs/toss4k_dp_elbow/policy.pt", 61),
                  ("rl", "RL (PPO)", "runs/toss4k_ppo_small/policy_iter40.pt", 52),
                  ("imitation", "Imitation (BC)", "runs/toss4k_bc/policy.pt", 40),
                  ("scripted", "Hand-written", "scripted", 83)],
        "best": ["imitation", "imitation", "imitation", "diffusion"],
    },
    "real": {
        "cups": [("imitation", "Imitation (chunked MLP)", "runs/real_place_chunk/policy.pt", [98, 69, 38]),
                 ("rl", "RL (PPO)", "runs/real_place_ppo/policy.pt", [100, 52, 35]),
                 ("scripted", "Hand-written", "scripted", [98, 60, 21])],
        "throw": [("diffusion", "Diffusion policy", "runs/real2_toss_dp/policy.pt", 9),
                  ("rl", "RL (PPO)", "runs/real2_toss_ppo/policy.pt", 7),
                  ("imitation", "Imitation (BC)", "runs/real2_toss_bc/policy.pt", 5),
                  ("scripted", "Hand-written", "scripted", 12.5)],
        "best": ["rl", "imitation", "imitation", "diffusion"],
    },
}

# Real-world limits the page can change: (RealSpec field or condition, label, unit, scale from the page's value to
# the spec's, min, max, step). The page sends values in its own units.
REAL_SETTINGS = [
    ("camera_bias", "Camera calibration error", "mm", 0.001, 0, 10, 0.5),
    ("object_bias", "Error per object", "mm", 0.001, 0, 10, 0.5),
    ("camera_jitter", "Jitter per frame", "mm", 0.001, 0, 5, 0.25),
    ("camera_dropout", "Frames dropped", "%", 0.01, 0, 20, 1),
    ("action_delay", "Command delay (up to)", "ms", 0.1, 0, 60, 10),
    ("camera_delay", "Camera image age", "ms", 0.02, 0, 150, 50),
    ("push", "Random pushes on the torso", "N", 1.0, 0, 20, 1),
    ("joint_noise", "Encoder noise", "mrad", 0.001, 0, 10, 0.5),
]
REAL_GROUPS = [("motors", "Motor strength varies (±20%), battery sag, joint friction"),
               ("friction", "Grip and object friction vary"), ("mass", "Cup, cap and hand weight vary")]
REAL_FIXED = ["Real torque limits (e.g. wrist 5 Nm), torque drops near top speed",
              "Joints below 85% of top speed and 8 rad/s", "Motors carry the arm's weight",
              "Motors heat up: weaker above 75 °C, stop at 90 °C",
              "The robot balances on its own feet and can fall", "Hand weighs 1.8 kg (Shadow Super Lite)"]


def available(spec: str) -> bool:
    return spec == "scripted" or (ROOT / spec).exists()


def geometry(real: bool) -> dict:
    """Table, reach and default layout for the page's top view (positions in metres, robot at the origin)."""
    from dataclasses import fields

    old = os.environ.pop("G1_REAL", None)
    if real:
        os.environ["G1_REAL"] = "1"
    try:
        from robot_arm.humanoid.config import task_config
        cfg = task_config(os.environ["G1_HAND"])
    finally:
        os.environ.pop("G1_REAL", None)
        if old is not None:
            os.environ["G1_REAL"] = old
    keys = ["table_center", "table_half", "reach_box", "reach_center", "reach_radius", "reach_bearing",
            "pyramid_center", "pyramid_yaw", "supply_cups", "cap_spot", "cap_max_x", "pyramid_clearance",
            "cup_clearance", "base_gap"]
    g = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name in keys}
    g["cup_radius"] = cfg.cup.mouth_radius + cfg.cup.wall
    g["cap_radius"] = 0.014
    return g


def default_layout() -> dict:
    """A layout the simulator itself drew (test seed 20001), so it passes every reach and clearance check."""
    from robot_arm.humanoid.envs import G1PyramidEnv

    env = G1PyramidEnv(task="full")
    env.reset(seed=20_001)
    xy = lambda p: [round(float(v), 3) for v in p[:2]]
    return {"supply": [xy(env.cup_pos(k)) for k in range(3)], "cap": xy(env.cap_pos),
            "pyramid": xy((env.layout.base_a + env.layout.base_b) / 2)}


def page_config() -> dict:
    ctl = {}
    for mode, c in CONTROLLERS.items():
        ctl[mode] = {"cups": [{"id": i, "label": l, "available": available(s), "scores": sc} for i, l, s, sc in c["cups"]],
                     "throw": [{"id": i, "label": l, "available": available(s), "score": sc}
                               for i, l, s, sc in c["throw"]],
                     "best": c["best"]}
    from robot_arm.humanoid.real import RealSpec

    spec = RealSpec()
    settings = []
    for key, label, unit, scale, lo, hi, step in REAL_SETTINGS:
        v = getattr(spec, key)
        v = v[1] if isinstance(v, tuple) else v
        settings.append({"key": key, "label": label, "unit": unit, "value": round(v / scale, 3), "min": lo, "max": hi,
                         "step": step})
    return {"tasks": TASKS, "controllers": ctl, "geometry": {"perfect": geometry(False), "real": geometry(True)},
            "layout": default_layout(),
            "real": {"settings": settings, "groups": [{"key": k, "label": l} for k, l in REAL_GROUPS],
                     "fixed": REAL_FIXED}}


def spec_for(mode: str, task: int, cid: str) -> str:
    group = CONTROLLERS[mode]["throw" if task == 3 else "cups"]
    for i, _, spec, _ in group:
        if i == cid and available(spec):
            return spec
    return "scripted"


def label_for(mode: str, task: int, cid: str) -> str:
    return next(l for i, l, _, _ in CONTROLLERS[mode]["throw" if task == 3 else "cups"] if i == cid)


def real_env(real: dict) -> tuple[str, dict]:
    """G1_REAL and G1_REAL_SET for the page's choices."""
    groups = [g for g, _ in REAL_GROUPS if g in real.get("groups", [])]
    names = ["clean", *groups, "action_delay", "camera_delay", "camera", "push"]
    scale = {k: s for k, _, _, s, *_ in REAL_SETTINGS}
    over = {}
    for key, value in real.get("values", {}).items():
        v = float(value) * scale[key]
        if key == "action_delay":
            over[key] = [0, int(round(v))]  # 10 ms ticks
        elif key == "camera_delay":
            over[key] = [int(round(v))] * 2  # 50 ms control steps
        else:
            over[key] = v
    return ",".join(names), over


# ---- the simulator process ----

def sim_worker(job: dict, out, cmds):
    import cv2
    import mujoco
    import numpy as np

    from robot_arm.humanoid.envs import OBS_STAGE, G1PyramidEnv
    from robot_arm.humanoid.scripted import PlaceExpert, TossExpert
    from robot_arm.policy import load_checkpoint

    env = G1PyramidEnv(task="full", render_mode="rgb_array", width=W, height=H)
    view = mujoco.Renderer(env.model, H, W)
    cache = {}

    def load(spec, toss):
        key = (spec, toss)
        if key not in cache:
            cache[key] = (TossExpert() if toss else PlaceExpert()) if spec == "scripted" else load_checkpoint(ROOT / spec)
        return cache[key]

    policies = [load(s, k == 3) for k, s in enumerate(job["specs"])]
    lay = job["layout"]
    obs, _ = env.reset(seed=job["seed"], options={"supply": lay["supply"], "cap": lay["cap"],
                                                  "pyramid_center": lay["pyramid"]})
    for p in policies:
        if hasattr(p, "reset"):
            p.reset()

    free = mujoco.MjvCamera()
    free.type = mujoco.mjtCamera.mjCAMERA_FREE
    free.lookat[:] = [0.35, -0.1, env.cfg.table_height + 0.05]
    free.distance, free.azimuth, free.elevation = 1.9, 200.0, -25.0
    camera = "free"

    def command(c):
        nonlocal camera
        if "preset" in c:
            camera = c["preset"]
        if "orbit" in c:
            camera = "free"
            free.azimuth -= 0.3 * c["orbit"][0]
            free.elevation = float(np.clip(free.elevation - 0.3 * c["orbit"][1], -89, -2))
        if "zoom" in c:
            camera = "free"
            free.distance = float(np.clip(free.distance * 1.1 ** c["zoom"], 0.4, 5.0))
        if "pan" in c:
            camera = "free"
            a = np.radians(free.azimuth)
            right, fwd = np.array([np.sin(a), -np.cos(a), 0]), np.array([np.cos(a), np.sin(a), 0])
            free.lookat[:] += 0.002 * free.distance * (-c["pan"][0] * right + c["pan"][1] * fwd)

    dt, steps_per_frame = env.cfg.control_dt, max(1, int(job.get("speed", 1)))
    t, current, done, outcome, reached = 0.0, -1, False, None, {}
    while True:
        tick = time.time()
        while not cmds.empty():
            c = cmds.get()
            if c.get("stop"):
                return
            command(c)
        if not done:
            for _ in range(steps_per_frame):
                stage = int(np.argmax(obs[OBS_STAGE]))
                k = min(stage, 3)
                if k != current:
                    if hasattr(policies[k], "reset"):
                        policies[k].reset()
                    current = k
                    reached.setdefault(stage, round(t, 1))
                obs, _, term, trunc, info = env.step(policies[k](obs))
                t += dt
                if term or trunc:
                    done = True
                    outcome = "Success: pyramid built and cap in the cup" if info["is_success"] else \
                        "Failed: " + (info.get("fail_reason") or "out of time")
                    break
        view.update_scene(env.data, camera=free if camera == "free" else camera)
        img = view.render()[:, :, ::-1].copy()
        stage = int(np.argmax(obs[OBS_STAGE]))
        label = outcome or STAGE_NAMES[min(stage, 4)]
        cv2.rectangle(img, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(img, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        speed = f"{steps_per_frame}x speed"
        (tw, th), _ = cv2.getTextSize(speed, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        cv2.rectangle(img, (W - tw - 26, 38), (W - 8, 50 + th), (0, 0, 0), -1)
        cv2.putText(img, speed, (W - tw - 17, 44 + th), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.send((jpg.tobytes(), {"stage": stage, "stage_name": STAGE_NAMES[min(stage, 4)], "time": round(t, 1),
                                  "controller": job["labels"][min(stage, 3)], "done": done, "outcome": outcome,
                                  "reached": reached, "camera": camera}))
        time.sleep(max(0.0, (dt if not done else 1 / 15) - (time.time() - tick)))


# ---- the server ----

class Run:
    """The current simulator process, its newest frame and status."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = self.cmds = None
        self.frame, self.status, self.seq = b"", {"running": False}, 0

    def start(self, req: dict):
        self.stop()
        mode = "real" if req.get("real", {}).get("on") else "perfect"
        ctl = req["controllers"]
        job = {"specs": [spec_for(mode, k, c) for k, c in enumerate(ctl)],
               "labels": [label_for(mode, k, c) for k, c in enumerate(ctl)],
               "layout": req["layout"], "seed": int(req.get("seed", 0)), "speed": int(req.get("speed", 1))}
        env = {"G1_HAND": os.environ["G1_HAND"]}
        if mode == "real":
            env["G1_REAL"], over = real_env(req["real"])
            env["G1_REAL_SET"] = json.dumps(over)
        ctx = get_context("spawn")
        a, b = ctx.Pipe(duplex=False)
        self.cmds = ctx.Queue()
        saved = {k: os.environ.get(k) for k in ("G1_REAL", "G1_REAL_SET")}
        with self.lock:  # the child inherits the environment at spawn
            for k in saved:
                os.environ.pop(k, None)
            os.environ.update(env)
            self.proc = ctx.Process(target=sim_worker, args=(job, b, self.cmds), daemon=True)
            self.proc.start()
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v
        b.close()
        self.status = {"running": True, "starting": True, "mode": mode, "labels": job["labels"]}
        threading.Thread(target=self._read, args=(a, self.proc), daemon=True).start()

    def _read(self, conn, proc):
        while True:
            try:
                jpg, st = conn.recv()
            except (EOFError, OSError):
                break
            if proc is not self.proc:
                break
            self.frame, self.seq = jpg, self.seq + 1
            self.status = {**self.status, **st, "running": True, "starting": False}
        if proc is self.proc:
            self.status = {**self.status, "running": False}

    def command(self, c: dict):
        if self.cmds is not None:
            self.cmds.put(c)

    def stop(self):
        if self.proc is not None:
            self.cmds.put({"stop": True})
            self.proc.join(timeout=3)
            if self.proc.is_alive():
                self.proc.terminate()
        self.proc = None
        self.status = {"running": False}


RUN = Run()
CONFIG = None


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
        if path == "/":
            self.send(Path(__file__).with_suffix(".html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/config":
            self.send(json.dumps(CONFIG).encode(), "application/json")
        elif path == "/status":
            self.send(json.dumps(RUN.status).encode(), "application/json")
        elif path == "/stream":
            self.stream()
        else:
            self.send(b"", "text/plain", 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        path = self.path.split("?")[0]
        if path == "/run":
            RUN.start(body)
        elif path == "/stop":
            RUN.stop()
        elif path == "/camera":
            RUN.command(body)
        else:
            return self.send(b"", "text/plain", 404)
        self.send(b"{}", "application/json")

    def stream(self):
        """MJPEG: the newest frame whenever there is a new one."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seen = -1
        try:
            while True:
                if RUN.seq != seen and RUN.frame:
                    seen, jpg = RUN.seq, RUN.frame
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                     str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *_):
        pass


def main():
    global CONFIG
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from outside a container")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)
    CONFIG = page_config()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Robot_arm GUI on {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        RUN.stop()


if __name__ == "__main__":
    main()
