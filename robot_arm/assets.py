"""Fetch robot models from MuJoCo Menagerie (only the folders we need)."""

import os
import subprocess
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"
MENAGERIE_COMMIT = "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "third_party" / "mujoco_menagerie"


def _run(args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def ensure_model(folder: str, xml: str, menagerie_dir: Path | None = None) -> Path:
    """Return the path to `<folder>/<xml>` in Menagerie, sparse-cloning or extending the checkout on first use.

    Set MENAGERIE_DIR to point at an existing Menagerie checkout instead.
    """
    menagerie_dir = Path(os.environ.get("MENAGERIE_DIR", menagerie_dir or DEFAULT_DIR))
    path = menagerie_dir / folder / xml
    if path.exists():
        return path

    print(f"Downloading {folder} from MuJoCo Menagerie into {menagerie_dir} ...")
    if not (menagerie_dir / ".git").exists():
        menagerie_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--quiet", "--filter=blob:none", "--sparse", "--no-checkout",
              MENAGERIE_URL, str(menagerie_dir)])
        _run(["git", "sparse-checkout", "set", folder], cwd=menagerie_dir)
    else:
        _run(["git", "sparse-checkout", "add", folder], cwd=menagerie_dir)
    _run(["git", "checkout", "--quiet", MENAGERIE_COMMIT], cwd=menagerie_dir)
    if not path.exists():
        raise FileNotFoundError(f"Menagerie checkout did not produce {path}")
    return path


def ensure_panda(menagerie_dir: Path | None = None) -> Path:
    return ensure_model("franka_emika_panda", "panda.xml", menagerie_dir)


def ensure_g1(menagerie_dir: Path | None = None) -> Path:
    """Unitree G1 with Dex3 three-finger hands."""
    return ensure_model("unitree_g1", "g1_with_hands.xml", menagerie_dir)
