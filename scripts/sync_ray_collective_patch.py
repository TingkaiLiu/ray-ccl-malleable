#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple


def _repo_root() -> Path:
    # This script lives at <portable_dir>/scripts/...
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _resolve_collective_dest_dir(conda_env: str) -> Path:
    code = (
        "import os\n"
        "import ray.util.collective as pkg\n"
        "print(os.path.dirname(pkg.__file__))\n"
    )
    out = _run(["conda", "run", "-n", conda_env, "python", "-c", code])
    return Path(out).resolve()


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_file(
    *,
    src: Path,
    dst: Path,
    dry_run: bool,
    backup: bool,
    backup_suffix: str,
) -> Tuple[bool, str]:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and _sha256(src) == _sha256(dst):
        return False, "unchanged"

    if dry_run:
        return True, "would-copy"

    if backup and dst.exists():
        bak = dst.with_name(dst.name + backup_suffix)
        shutil.copy2(dst, bak)

    shutil.copy2(src, dst)
    return True, "copied"


def _default_mappings(repo: Path) -> Dict[Path, Path]:
    # Source files are vendored flat under <portable_dir>/bundle/
    src_root = repo / "bundle"
    return {
        src_root / "collective.py": Path("collective.py"),
        src_root / "util.py": Path("util.py"),
        src_root / "__init__.py": Path("__init__.py"),
        src_root / "malleable.py": Path("malleable.py"),
        src_root
        / "torch_gloo_collective_group.py": Path("collective_group")
        / "torch_gloo_collective_group.py",
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Sync modified ray.util.collective files from this repo into a conda env's "
            "installed Ray site-packages (for local testing)."
        )
    )
    p.add_argument(
        "--env",
        default="ray",
        help="Conda env name to target (default: ray).",
    )
    p.add_argument(
        "--dest",
        default="",
        help=(
            "Destination collective package dir. If set, bypasses conda lookup. "
            "Example: /path/to/site-packages/ray/util/collective"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without copying.",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak copies of overwritten files.",
    )
    args = p.parse_args(argv)

    repo = _repo_root()
    if args.dest:
        dest_dir = Path(args.dest).expanduser().resolve()
    else:
        dest_dir = _resolve_collective_dest_dir(args.env)

    if not dest_dir.exists():
        print(f"ERROR: destination dir does not exist: {dest_dir}", file=sys.stderr)
        return 2

    normalized = str(dest_dir).replace("\\", "/")
    if not normalized.endswith("/ray/util/collective"):
        print(
            f"ERROR: destination does not look like ray/util/collective: {dest_dir}",
            file=sys.stderr,
        )
        return 2

    backup_suffix = ".bak." + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    mappings = _default_mappings(repo)

    print(f"Repo: {repo}")
    print(f"Dest: {dest_dir}")
    print(f"Mode: {'dry-run' if args.dry_run else 'copy'}")
    print(f"Backup: {'disabled' if args.no_backup else 'enabled'}")
    print()

    any_changes = False
    for src, rel_dst in mappings.items():
        dst = dest_dir / rel_dst
        changed, status = _copy_file(
            src=src,
            dst=dst,
            dry_run=args.dry_run,
            backup=(not args.no_backup),
            backup_suffix=backup_suffix,
        )
        any_changes = any_changes or changed
        print(f"{status:10} {src.relative_to(repo)} -> {dst}")

    print()
    if any_changes:
        print("Done.")
        if args.dry_run:
            print("Re-run without --dry-run to apply.")
    else:
        print("Done. No changes needed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

