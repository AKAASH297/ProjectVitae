import logging
import re
import subprocess
from pathlib import Path

import requests

from project_vitae.config import Config
from project_vitae.io_utils import USERPROFILE_DIR, slugify
from project_vitae.models import ProjectVitaeError, SessionState

logger = logging.getLogger(__name__)

GITHUB_URL_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/.]+?)(\.git)?$")
MAX_REPO_SIZE_KB = 200 * 1024


def _check_github_size(url: str) -> int | None:
    m = GITHUB_URL_RE.match(url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        resp = requests.get(api_url, timeout=10)
        if resp.status_code == 200:
            size_kb = resp.json().get("size", 0)
            return size_kb
    except requests.RequestException:
        logger.warning("GitHub API call failed for %s", url, exc_info=True)
    return None


def make_clone(cfg: Config):
    def clone(state: SessionState) -> dict:
        clones_dir = USERPROFILE_DIR / "clones" / slugify(state.session_name)
        clones_dir.mkdir(parents=True, exist_ok=True)
        clone_dirs: list[str] = []
        warnings: list[dict] = []
        skipped: list[str] = list(state.skipped_repos)

        for url in state.github_urls:
            if url in skipped:
                logger.info("skipping previously skipped repo: %s", url)
                continue

            size_kb = _check_github_size(url)
            if size_kb is not None and size_kb > MAX_REPO_SIZE_KB:
                warning = {"url": url, "reason": "too_large", "size_kb": size_kb}
                warnings.append(warning)
                skipped.append(url)
                logger.warning("repo %s is %d KB, skipping (>200 MB)", url, size_kb)
                continue

            repo_name = slugify(url.rstrip("/").split("/")[-1].replace(".git", ""))
            dest = clones_dir / repo_name
            logger.info("cloning %s into %s", url, dest)
            try:
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", url, str(dest)],
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode != 0:
                    stderr = result.stderr[-500:]
                    warnings.append({"url": url, "reason": "clone_failed", "stderr": stderr})
                    skipped.append(url)
                    logger.error("clone failed for %s: %s", url, stderr)
                    continue
                clone_dirs.append(str(dest.relative_to(USERPROFILE_DIR)))
                logger.info("successfully cloned %s", url)
            except subprocess.TimeoutExpired:
                warnings.append({"url": url, "reason": "timeout"})
                skipped.append(url)
                logger.error("clone timed out for %s", url)

        if not clone_dirs and not skipped:
            raise ProjectVitaeError("no repos to clone — all URLs were empty or invalid")

        if not clone_dirs:
            raise ProjectVitaeError("no successful clones — all repos failed or were skipped")

        return {
            "clones_dir": str(clones_dir.relative_to(USERPROFILE_DIR)),
            "clone_dirs": clone_dirs,
            "exploration_warnings": warnings,
            "skipped_repos": skipped,
        }

    return clone
