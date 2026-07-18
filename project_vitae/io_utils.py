import os
import tempfile
from pathlib import Path


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_prompt(path: Path) -> str:
    resolved = path.resolve()
    return resolved.read_text(encoding="utf-8")


def slugify(text: str) -> str:
    safe_chars = []
    for ch in text.lower().strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            safe_chars.append(ch)
        elif ch in (" ", "_"):
            safe_chars.append("-")
    result = "".join(safe_chars)
    while "--" in result:
        result = result.replace("--", "-")
    return result.strip("-") or "untitled"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
