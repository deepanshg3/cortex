"""Bundle experiment recovery artifacts for persistent storage.

On Kaggle, ``/kaggle/working`` is erased when a session resets. This tool
collects everything needed to resume or audit a run — checkpoints, configs,
metrics, campaign summary, split manifests, and the code revision — into a
single timestamped ``.tar.gz`` (optionally excluding the very large periodic
``step_*.pt`` files, which are never required to resume), and writes a
SHA-256 manifest so the archive can be verified after copying.

Usage::

    # Everything under experiments/ (excluding step_*.pt) -> persistent dir
    python -m src.export_run --source experiments --dest /kaggle/working/backups

    # A single experiment
    python -m src.export_run --source experiments/campaign_v1/baseline \
        --dest /kaggle/working/backups

    # Verify an existing archive against its embedded manifest
    python -m src.export_run --verify /kaggle/working/backups/cortex_export_*.tar.gz
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.logger import get_logger  # noqa: E402

logger = get_logger("src.export_run")

# Extras copied when present, for provenance even if they live outside --source.
OPTIONAL_EXTRAS = [
    "configs",
    "docs/FINAL_REPORT.md",
    "docs/DATA_AND_EVALUATION.md",
    "pytest.ini",
]


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _git(args: List[str]) -> str:
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.check_output(["git", *args], cwd=cwd,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _collect(source: str, include_steps: bool, extras: bool) -> List[str]:
    files = []
    if os.path.isfile(source):
        files.append(source)
    else:
        for root, _dirs, names in os.walk(source):
            for n in names:
                if not include_steps and n.startswith("step_") and n.endswith(".pt"):
                    continue
                files.append(os.path.join(root, n))
    if extras:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for extra in OPTIONAL_EXTRAS:
            p = os.path.join(repo, extra)
            if os.path.isfile(p):
                files.append(p)
            elif os.path.isdir(p):
                for root, _d, names in os.walk(p):
                    files += [os.path.join(root, n) for n in names]
    return sorted(set(files))


def export(source: str, dest: str, name: Optional[str] = None,
           include_steps: bool = False, extras: bool = True,
           keep_parts: int = 1) -> Dict:
    if not os.path.exists(source):
        raise FileNotFoundError(source)
    os.makedirs(dest, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = name or f"cortex_export_{stamp}"
    files = _collect(source, include_steps, extras)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def arcname_for(path: str) -> str:
        ap = os.path.abspath(path)
        if ap.startswith(repo + os.sep):
            return os.path.relpath(ap, repo)
        return os.path.join("external", os.path.basename(ap))

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": os.path.abspath(source),
        "git_commit": _git(["rev-parse", "--short", "HEAD"]),
        "git_status": _git(["status", "--porcelain"]),
        "include_step_checkpoints": include_steps,
        "files": [
            {"path": arcname_for(f), "size": os.path.getsize(f), "sha256": _sha256(f)}
            for f in files
        ],
    }

    manifest_path = os.path.join(dest, f"{base}.manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    # Pack into one or more parts (useful if a single artifact exceeds a limit).
    arcnames = {f: arcname_for(f) for f in files}
    groups = [files] if keep_parts <= 1 else [files[i::keep_parts] for i in range(keep_parts)]
    parts = []
    for i, group in enumerate(groups):
        suffix = "" if keep_parts <= 1 else f".part{i + 1:02d}"
        archive = os.path.join(dest, f"{base}{suffix}.tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            for f in group:
                tar.add(f, arcname=arcnames[f])
            tar.add(manifest_path, arcname=os.path.basename(manifest_path))
        parts.append(archive)

    result = {"manifest": manifest_path, "archives": parts,
              "n_files": len(manifest["files"]),
              "total_bytes": sum(f["size"] for f in manifest["files"])}
    logger.info("Exported %d files (%.1f MiB) -> %s",
                result["n_files"], result["total_bytes"] / 1e6, parts)
    return result


def verify(archive: str) -> Dict:
    """Verify an archive against the ``*.manifest.json`` sitting beside it."""
    with tarfile.open(archive, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        manifest_name = next((n for n in members if n.endswith(".manifest.json")), None)
        if not manifest_name:
            raise ValueError("archive does not contain a manifest")
        manifest = json.load(tar.extractfile(members[manifest_name]))
        ok, checked = True, 0
        for entry in manifest["files"]:
            if entry["path"] not in members:
                ok = False
                logger.error("Missing file in archive: %s", entry["path"])
                continue
            data = tar.extractfile(members[entry["path"]]).read()
            checked += 1
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                ok = False
                logger.error("Checksum mismatch: %s", entry["path"])
    logger.info("Verify %s: %s (%d files checked)", archive, "OK" if ok else "FAILED", checked)
    return {"ok": ok, "checked": checked}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Export Cortex recovery artifacts")
    ap.add_argument("--source", default="experiments")
    ap.add_argument("--dest", default="backups")
    ap.add_argument("--name", default=None)
    ap.add_argument("--include-steps", action="store_true",
                    help="include periodic step_*.pt checkpoints (large)")
    ap.add_argument("--no-extras", action="store_true")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--verify", default=None, help="verify an existing archive and exit")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.verify:
        return 0 if verify(args.verify)["ok"] else 1
    export(args.source, args.dest, name=args.name, include_steps=args.include_steps,
           extras=not args.no_extras, keep_parts=args.parts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
