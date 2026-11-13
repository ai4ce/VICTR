import hashlib
from pathlib import Path

import pytest

from download_checkpoints import verify_release


def release_fixture(folder: Path):
    content = b"synthetic checkpoint bytes"
    checksum = hashlib.sha256(content).hexdigest()
    (folder / "checkpoint.pt").write_bytes(content)
    (folder / "SHA256SUMS").write_text(f"{checksum}  checkpoint.pt\n")
    return {"filename": "checkpoint.pt", "size_bytes": len(content), "sha256": checksum}


def test_verify_release_rejects_modified_checkpoint(tmp_path):
    spec = release_fixture(tmp_path)
    verify_release(tmp_path, spec)
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"x" * spec["size_bytes"])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_release(tmp_path, spec)


def test_checksum_manifest_cannot_override_pinned_identity(tmp_path):
    spec = release_fixture(tmp_path)
    spec["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned paper release"):
        verify_release(tmp_path, spec)


def test_checksum_manifest_rejects_parent_paths(tmp_path):
    spec = release_fixture(tmp_path)
    (tmp_path / "SHA256SUMS").write_text("0" * 64 + "  ../outside.pt\n")
    with pytest.raises(ValueError, match="Unsafe checksum path"):
        verify_release(tmp_path, spec)
