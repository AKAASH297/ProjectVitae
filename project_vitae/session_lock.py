from __future__ import annotations

import os
import time
from pathlib import Path


LOCK_TIMEOUT_SECONDS = 3600


def acquire_lock(session_dir: Path) -> bool:
    lock_file = session_dir / ".lock"
    session_dir.mkdir(parents=True, exist_ok=True)

    if lock_file.exists():
        content = lock_file.read_text(encoding="utf-8").strip()
        parts = content.split(":", 1)
        if len(parts) == 2:
            try:
                timestamp = float(parts[1])
                if time.time() - timestamp < LOCK_TIMEOUT_SECONDS:
                    return False
            except ValueError:
                pass
        lock_file.unlink(missing_ok=True)

    lock_file.write_text(f"{os.getpid()}:{time.time()}", encoding="utf-8")
    return True


def release_lock(session_dir: Path) -> None:
    lock_file = session_dir / ".lock"
    lock_file.unlink(missing_ok=True)


def has_lock(session_dir: Path) -> bool:
    lock_file = session_dir / ".lock"
    return lock_file.exists()
