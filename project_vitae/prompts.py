import logging
from pathlib import Path

from project_vitae.config import SubagentConfig
from project_vitae.io_utils import USERPROFILE_DIR, userprofile_path
from project_vitae.models import PromptError

logger = logging.getLogger(__name__)


def ensure_prompt_path_is_safe(path: str) -> None:
    if ".." in path.split("/") or ".." in path.split("\\"):
        raise PromptError(f"path traversal detected: {path}")
    p = Path(path)
    if p.is_absolute() or path.startswith("/") or path.startswith("\\"):
        raise PromptError(f"absolute path not allowed: {path}")


def load_prompt(prompt_path_rel: str) -> str:
    ensure_prompt_path_is_safe(prompt_path_rel)
    path = userprofile_path([prompt_path_rel])
    if not path.is_file():
        raise PromptError(f"prompt file not found: {prompt_path_rel} (resolved: {path})")
    return path.read_text(encoding="utf-8")


def resolve_prompt(subagent_name: str, cfg: SubagentConfig) -> str:
    if cfg.system_prompt_override:
        logger.info("using system prompt override for %s", subagent_name)
        return cfg.system_prompt_override
    return load_prompt(cfg.prompt_version)
