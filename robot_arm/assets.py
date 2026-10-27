"""Fetch the Franka Panda model from MuJoCo Menagerie (only the folder we need)."""

import os
import subprocess
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"
MENAGERIE_COMMIT = "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "third_party" / "mujoco_menagerie"


def _run(args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def ensure_panda(menagerie_dir: Path | None = None) -> Path:
    """Return the path to panda.xml, sparse-cloning Menagerie on first use.

    Set MENAGERIE_DIR to point at an existing Menagerie checkout instead.
    """
    menagerie_dir = Path(os.environ.get("MENAGERIE_DIR", menagerie_dir or DEFAULT_DIR))
    panda_xml = menagerie_dir / "franka_emika_panda" / "panda.xml"
    if panda_xml.exists():
        return panda_xml

    print(f"Downloading the Panda model from MuJoCo Menagerie into {menagerie_dir} ...")
    menagerie_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--quiet", "--filter=blob:none", "--sparse", "--no-checkout",
          MENAGERIE_URL, str(menagerie_dir)])
    _run(["git", "sparse-checkout", "set", "franka_emika_panda"], cwd=menagerie_dir)
    _run(["git", "checkout", "--quiet", MENAGERIE_COMMIT], cwd=menagerie_dir)
    if not panda_xml.exists():
        raise FileNotFoundError(f"Menagerie checkout did not produce {panda_xml}")
    return panda_xml
