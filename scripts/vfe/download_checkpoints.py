#!/usr/bin/env python
"""Download a pinned paper checkpoint release and verify its SHA256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_release(folder: Path, spec: dict) -> None:
    folder = folder.resolve()
    checkpoint = folder / spec["filename"]
    if not checkpoint.is_file() or checkpoint.stat().st_size != spec["size_bytes"]:
        raise ValueError(f"Missing or truncated checkpoint: {checkpoint}")
    verified = {}
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or path == folder:
            raise ValueError(f"Unsafe checksum path: {relative}")
        if relative in verified:
            raise ValueError(f"Duplicate checksum entry: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
        verified[relative] = actual
    if verified.get(spec["filename"]) != spec["sha256"]:
        raise ValueError("Checkpoint checksum does not match the pinned paper release")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["ic_vfe", "ic_vfe_no_context", "recap", "all"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/vfe/checkpoints.json")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints", help="Parent directory; each experiment gets its own subdirectory")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing local files without network access")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    names = list(manifest["models"]) if args.experiment == "all" else [args.experiment]
    for name in names:
        spec = manifest["models"][name]
        destination = args.output / name
        if not args.verify_only:
            if not spec.get("revision"):
                parser.error(f"{name}: the prepared paper package has not been published yet")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"], local_dir=destination,
                              allow_patterns=[spec["filename"], "run_config.json", "README.md", "SHA256SUMS", "paper-release/**"])
        verify_release(destination, spec)
        print(f"Verified {name}: {destination / spec['filename']}")


if __name__ == "__main__":
    main()
