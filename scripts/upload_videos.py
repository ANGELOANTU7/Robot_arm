"""Send a folder of recorded clips to the cloud session through a separate branch of your GitHub repository.

Run it on your own computer (Python 3.8+ and git; nothing else to install):

    python3 upload_videos.py ~/Desktop/pyramid_videos

The folder's sub-folders are kept (e.g. part_a/, part_b/, part_c/). The clips are committed to a branch of their own
(`videos-upload`, sharing no history with the code) and pushed in batches, so the code branches stay small. The
cloud session fetches that branch, and the branch can be deleted afterwards.

Uses your normal git login for GitHub. If the push asks for a password, use a personal access token, or run
`gh auth login` once; with an SSH key, pass --repo git@github.com:ANGELOANTU7/Robot_arm.git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
GITHUB_FILE_LIMIT_MB = 95  # GitHub refuses files over 100 MB


def git(args: list[str], cwd: Path, quiet: bool = False) -> subprocess.CompletedProcess:
    # Clips are already compressed: don't spend time compressing or delta-ing them again.
    base = ["git", "-c", "core.bigFileThreshold=1m", "-c", "core.compression=0", "-c", "http.postBuffer=524288000"]
    return subprocess.run(base + args, cwd=cwd, check=True, capture_output=quiet, text=True)


def find_clips(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
                  and not any(part.startswith(".") for part in p.relative_to(folder).parts))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, help="folder with the clips (sub-folders are kept)")
    p.add_argument("--repo", default="https://github.com/ANGELOANTU7/Robot_arm.git")
    p.add_argument("--branch", default="videos-upload")
    p.add_argument("--batch-mb", type=float, default=800, help="push after about this much data")
    p.add_argument("--dry-run", action="store_true", help="only list what would be sent")
    args = p.parse_args()

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")
    clips = find_clips(folder)
    if not clips:
        sys.exit(f"No video files ({', '.join(sorted(VIDEO_EXTENSIONS))}) in {folder}")
    mb = {c: c.stat().st_size / 1e6 for c in clips}
    too_big = [c for c in clips if mb[c] > GITHUB_FILE_LIMIT_MB]
    clips = [c for c in clips if c not in too_big]

    print(f"Clips in {folder}:")
    groups: dict[str, list[Path]] = {}
    for c in clips:
        groups.setdefault(str(c.parent.relative_to(folder)), []).append(c)
    for name, items in groups.items():
        print(f"  {name:20s} {len(items):3d} clips, {sum(mb[c] for c in items):7.1f} MB")
    print(f"  {'total':20s} {len(clips):3d} clips, {sum(mb[c] for c in clips):7.1f} MB")
    if too_big:
        print(f"\nSkipped (over {GITHUB_FILE_LIMIT_MB} MB, GitHub's limit): send these as chat attachments instead, or "
              "trim them:")
        for c in too_big:
            print(f"  {c.relative_to(folder)}  ({mb[c]:.0f} MB)")
    if args.dry_run or not clips:
        return

    if shutil.which("git") is None:
        sys.exit("git is not installed. On a Mac run: xcode-select --install")
    work = Path(tempfile.mkdtemp(prefix="videos-upload-"))
    try:
        git(["init", "-q"], work)
        git(["symbolic-ref", "HEAD", f"refs/heads/{args.branch}"], work)
        if not subprocess.run(["git", "config", "user.email"], cwd=work, capture_output=True, text=True).stdout.strip():
            git(["config", "user.email", "videos@upload.local"], work)
            git(["config", "user.name", "Video upload"], work)

        batches, current, size = [], [], 0.0
        for c in clips:
            if current and size + mb[c] > args.batch_mb:
                batches.append(current)
                current, size = [], 0.0
            current.append(c)
            size += mb[c]
        batches.append(current)

        manifest = []
        for i, batch in enumerate(batches, 1):
            for c in batch:
                rel = c.relative_to(folder)
                dest = work / "videos" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(c, dest)
                manifest.append({"path": f"videos/{rel.as_posix()}", "bytes": c.stat().st_size, "sha256": sha256(c)})
            (work / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
            git(["add", "-A"], work)
            git(["commit", "-q", "-m", f"Clips, batch {i}/{len(batches)}"], work)
            print(f"\nPushing batch {i}/{len(batches)} ({sum(mb[c] for c in batch):.0f} MB, {len(batch)} clips) ...")
            push = ["push", args.repo, f"HEAD:refs/heads/{args.branch}"]
            try:
                git(push + (["--force"] if i == 1 else []), work)
            except subprocess.CalledProcessError:
                sys.exit("\nThe push failed. Check that you can push to the repository: run `gh auth login` (GitHub CLI) "
                         "or use a personal access token as the password; with an SSH key, add\n"
                         "  --repo git@github.com:ANGELOANTU7/Robot_arm.git\n"
                         "Running the script again starts over and replaces the branch.")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"\nDone: {len(clips)} clips on branch '{args.branch}'. Tell Claude the videos are uploaded"
          + (" (and send the skipped ones as attachments)." if too_big else "."))


if __name__ == "__main__":
    main()
