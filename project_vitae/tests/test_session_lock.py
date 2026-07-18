import os
import time
from pathlib import Path

from project_vitae.session_lock import acquire_lock, has_lock, release_lock


def test_acquire_and_release_lock(tmp_path):
    sess_dir = tmp_path / "session1"
    assert acquire_lock(sess_dir) is True
    assert has_lock(sess_dir)
    release_lock(sess_dir)
    assert not has_lock(sess_dir)


def test_double_acquire_fails(tmp_path):
    sess_dir = tmp_path / "session2"
    assert acquire_lock(sess_dir) is True
    assert acquire_lock(sess_dir) is False


def test_stale_lock_cleared(tmp_path):
    sess_dir = tmp_path / "session3"
    lock_file = sess_dir / ".lock"
    sess_dir.mkdir(parents=True, exist_ok=True)
    stale_time = time.time() - 4000
    lock_file.write_text(f"{os.getpid()}:{stale_time}", encoding="utf-8")
    assert acquire_lock(sess_dir) is True


def test_no_lock_by_default(tmp_path):
    sess_dir = tmp_path / "session4"
    assert not has_lock(sess_dir)
