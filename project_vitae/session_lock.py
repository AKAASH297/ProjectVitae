import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from project_vitae.io_utils import atomic_write_text, read_text
from project_vitae.models import SessionLockError

logger = logging.getLogger(__name__)

STALE_THRESHOLD_SECONDS = 3600
_lock = threading.Lock()


def _pid_is_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10,
            )
            return str(pid) in result.stdout
        else:
            os.kill(pid, 0)
            return True
    except (OSError, subprocess.TimeoutExpired):
        return False


class SessionLock:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.lock_path = session_dir / ".lock"
        self._pid = os.getpid()
        self._locked = False

    def acquire(self, force: bool = False) -> "SessionLock":
        with _lock:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            if self.lock_path.is_file():
                data = json.loads(read_text(self.lock_path))
                lock_pid = data.get("pid")
                started = data.get("started_at", "")
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds() if started else 0
                if lock_pid and _pid_is_alive(lock_pid) and age < STALE_THRESHOLD_SECONDS and not force:
                    raise SessionLockError(
                        f"session is locked by PID {lock_pid} (age={age:.0f}s); "
                        f"use force=True or wait for the lock to expire ({STALE_THRESHOLD_SECONDS}s)"
                    )
            self._write_lock()
            self._locked = True
        return self

    def acquire_for_resume(self) -> "SessionLock":
        with _lock:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._write_lock()
            self._locked = True
        return self

    def _write_lock(self) -> None:
        data = {"pid": self._pid, "started_at": datetime.now(timezone.utc).isoformat()}
        atomic_write_text(self.lock_path, json.dumps(data, indent=2))

    def release(self) -> None:
        with _lock:
            if self._locked and self.lock_path.is_file():
                try:
                    data = json.loads(read_text(self.lock_path))
                    if data.get("pid") == self._pid:
                        self.lock_path.unlink(missing_ok=True)
                except Exception:
                    logger.warning("failed to remove lock file %s", self.lock_path, exc_info=True)
            self._locked = False

    def __enter__(self) -> "SessionLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def list_resumable_sessions(userprofile_dir: Path | None = None) -> list[str]:
    from project_vitae.io_utils import USERPROFILE_DIR
    base = userprofile_dir or USERPROFILE_DIR
    sessions_dir = base / "sessions"
    if not sessions_dir.is_dir():
        return []
    resumable: list[str] = []
    for child in sorted(sessions_dir.iterdir()):
        if child.is_dir():
            lock_file = child / ".lock"
            state_file = child / "resume_state.json"
            if lock_file.is_file() and state_file.is_file():
                resumable.append(child.name)
    return resumable
