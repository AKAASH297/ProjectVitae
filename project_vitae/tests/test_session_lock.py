import json
from pathlib import Path

import pytest

from project_vitae.models import SessionLockError
from project_vitae.session_lock import SessionLock, list_resumable_sessions


def test_acquire_and_release(tmp_path: Path):
    lock = SessionLock(tmp_path)
    lock.acquire()
    assert lock.lock_path.is_file()
    data = json.loads(lock.lock_path.read_text())
    assert "pid" in data
    assert "started_at" in data
    lock.release()
    assert not lock.lock_path.exists()


def test_context_manager(tmp_path: Path):
    with SessionLock(tmp_path) as lock:
        assert lock.lock_path.is_file()
    assert not lock.lock_path.exists()


def test_second_acquire_fails(tmp_path: Path):
    lock1 = SessionLock(tmp_path)
    lock1.acquire()
    lock2 = SessionLock(tmp_path)
    with pytest.raises(SessionLockError, match="locked by PID"):
        lock2.acquire()
    lock1.release()


def test_force_bypasses_lock(tmp_path: Path):
    lock1 = SessionLock(tmp_path)
    lock1.acquire()
    lock2 = SessionLock(tmp_path)
    lock2.acquire(force=True)
    assert lock2.lock_path.is_file()
    lock2.release()


def test_stale_lock_acquireable(tmp_path: Path):
    lock_path = tmp_path / ".lock"
    data = {"pid": 999999, "started_at": "2000-01-01T00:00:00+00:00"}
    lock_path.write_text(json.dumps(data))
    lock = SessionLock(tmp_path)
    lock.acquire()
    assert lock.lock_path.is_file()
    new_data = json.loads(lock.lock_path.read_text())
    assert new_data["pid"] != 999999
    lock.release()


def test_acquire_for_resume(tmp_path: Path):
    lock1 = SessionLock(tmp_path)
    lock1.acquire()
    lock2 = SessionLock(tmp_path)
    lock2.acquire_for_resume()
    assert lock2.lock_path.is_file()
    lock2.release()


def test_list_resumable_sessions(tmp_path: Path):
    s1 = tmp_path / "sessions" / "s1"
    s1.mkdir(parents=True)
    (s1 / ".lock").write_text("{}")
    (s1 / "resume_state.json").write_text("{}")
    s2 = tmp_path / "sessions" / "s2"
    s2.mkdir(parents=True)
    (s2 / ".lock").write_text("{}")
    s3 = tmp_path / "sessions" / "s3"
    s3.mkdir(parents=True)
    (s3 / "resume_state.json").write_text("{}")
    result = list_resumable_sessions(tmp_path)
    assert "s1" in result
    assert "s2" not in result
    assert "s3" not in result


def test_release_safe_when_not_locked(tmp_path: Path):
    lock = SessionLock(tmp_path)
    lock.release()
