import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Sequence, TypeVar

import yaml
from pydantic import BaseModel

from project_vitae.models import ProjectVitaeError, ProjectRecord

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def get_userprofile_dir() -> Path:
    return Path(os.environ.get("PROJECTVITAE_USERPROFILE", "./userprofile")).resolve()


USERPROFILE_DIR = get_userprofile_dir()


def _resolve_path(parts: Sequence[str]) -> Path:
    base = get_userprofile_dir()
    raw = base.joinpath(*parts)
    resolved = raw.resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise ProjectVitaeError(f"path traversal denied: {raw}")
    return resolved


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def read_text(path: Path) -> str:
    resolved = path.resolve()
    return resolved.read_text(encoding="utf-8")


def load_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: Any) -> None:
    text = yaml.safe_dump(data, default_flow_style=False, allow_unicode=True)
    atomic_write_text(path, text)


def load_json_model(path: Path, model_cls: type[T]) -> T:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return model_cls.model_validate(data)


def save_json_model(path: Path, model: BaseModel) -> None:
    atomic_write_text(path, model.model_dump_json(indent=2))


def slugify(title: str) -> str:
    s = title.lower().strip()
    s = re.sub(r"[!&%$#_{}~^\\/:*?\"<>|]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    s = s.strip("-")
    return s


def userprofile_path(parts: Sequence[str]) -> Path:
    return _resolve_path(parts)


def parse_userinfo(text: str) -> tuple[dict, str]:
    text = text.lstrip("\n")
    if text.startswith("---"):
        end = text.find("---", 3)
        if end == -1:
            raise ProjectVitaeError("invalid YAML front-matter: no closing ---")
        front_raw = text[3:end].strip()
        body = text[end + 3 :].strip()
        if front_raw:
            try:
                front = yaml.safe_load(front_raw) or {}
            except yaml.YAMLError:
                raise ProjectVitaeError("invalid YAML front-matter: parse error")
            if not isinstance(front, dict):
                raise ProjectVitaeError("invalid YAML front-matter: not a mapping")
        else:
            front = {}
        return front, body
    return {}, text


def serialize_userinfo(front_matter: dict, body: str) -> str:
    parts = []
    if front_matter:
        parts.append("---")
        parts.append(yaml.safe_dump(front_matter, default_flow_style=False, allow_unicode=True).strip())
        parts.append("---")
    parts.append(body)
    return "\n\n".join(p for p in parts if p)


def find_project_dir(title: str) -> Path:
    slug = slugify(title)
    base = get_userprofile_dir()
    candidate = base / "projects" / slug
    if candidate.is_dir():
        return candidate
    projects_dir = base / "projects"
    if not projects_dir.is_dir():
        return candidate
    for child in projects_dir.iterdir():
        if child.is_dir():
            record_path = child / "record.yaml"
            if record_path.is_file():
                try:
                    rec = load_yaml(record_path)
                    if isinstance(rec, dict) and rec.get("title", "").lower() == title.lower():
                        return child
                except Exception:
                    continue
    return candidate


def load_project_records() -> list[ProjectRecord]:
    records: list[ProjectRecord] = []
    base = get_userprofile_dir()
    projects_dir = base / "projects"
    if not projects_dir.is_dir():
        return records
    for child in sorted(projects_dir.iterdir()):
        if child.is_dir():
            record_path = child / "record.yaml"
            if record_path.is_file():
                try:
                    rec = load_yaml(record_path)
                    if isinstance(rec, dict):
                        records.append(ProjectRecord.model_validate(rec))
                except Exception as e:
                    logger.warning("failed to load project record %s: %s", record_path, e)
    return records
