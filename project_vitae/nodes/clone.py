from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def clone_repo(url: str, clone_dir: Path, timeout: int = 300) -> Path:
    repo_name = _repo_name_from_url(url)
    dest = clone_dir / repo_name
    if dest.exists():
        logger.info("Clone target %s already exists — reusing", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s into %s ...", url, dest)
    subprocess.run(
        ["git", "clone", url, str(dest)],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return dest


def _repo_name_from_url(url: str) -> str:
    name = url.strip().rstrip("/")
    if name.endswith(".git"):
        name = name[:-4]
    parts = name.split("/")
    return parts[-1] if parts else "unknown"
