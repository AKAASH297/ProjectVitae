import logging
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from project_vitae.io_utils import USERPROFILE_DIR, load_yaml
from project_vitae.models import ConfigError

logger = logging.getLogger(__name__)

REQUIRED_SUBAGENTS = frozenset(
    {"explore", "filter", "writing", "content_critique", "compile_critique"}
)


class SubagentConfig(BaseModel):
    provider: Literal["anthropic", "openai_compatible"]
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    model: str
    prompt_version: str
    temperature: float = 0.3
    max_tokens: int = 4096
    per_repo_token_budget: int | None = None
    system_prompt_override: str | None = None


class RetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_seconds: list[int] = Field(default_factory=lambda: [1, 2, 4])


class CostConfig(BaseModel):
    per_session_cap_usd: float = 5.00
    pricing_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)


class LatexConfig(BaseModel):
    template_path: str = "template.tex"
    compiler: Literal["auto", "tectonic", "pdflatex"] = "auto"


class Config(BaseModel):
    subagents: dict[str, SubagentConfig]
    retry: RetryConfig = RetryConfig()
    cost: CostConfig = CostConfig()
    latex: LatexConfig = LatexConfig()
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    @model_validator(mode="after")
    def _validate_subagent_keys(self) -> "Config":
        given = set(self.subagents.keys())
        missing = REQUIRED_SUBAGENTS - given
        if missing:
            raise ConfigError(f"missing subagent sections: {', '.join(sorted(missing))}")
        extra = given - REQUIRED_SUBAGENTS
        if extra:
            raise ConfigError(f"unknown subagent sections: {', '.join(sorted(extra))}")
        return self

    def subagent(self, name: str) -> SubagentConfig:
        if name not in self.subagents:
            raise KeyError(f"no subagent named '{name}'")
        return self.subagents[name]

    def api_key(self, subagent_name: str) -> str:
        cfg = self.subagent(subagent_name)
        key = cfg.api_key or (os.environ.get(cfg.api_key_env) if cfg.api_key_env else None)
        if not key:
            source = f"environment variable {cfg.api_key_env}" if cfg.api_key_env else "API key"
            raise ConfigError(f"{source} not set for subagent '{subagent_name}'")
        return key


def load_config(path: Path | None = None) -> Config:
    if path is None:
        path = USERPROFILE_DIR / "config.yaml"
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ConfigError(f"config file must contain a YAML mapping, got {type(data).__name__}")
    _check_unknown_keys(data, _CONFIG_ROOT_KEYS, "config root")
    if "subagents" in data and isinstance(data["subagents"], dict):
        for name, sa in data["subagents"].items():
            if isinstance(sa, dict):
                _check_unknown_keys(sa, _SUBAGENT_KEYS, f"subagent '{name}'")
    return Config.model_validate(data)


_CONFIG_ROOT_KEYS = frozenset({"subagents", "retry", "cost", "latex", "log_level"})
_SUBAGENT_KEYS = frozenset(
    {
        "provider",
        "base_url",
        "api_key",
        "api_key_env",
        "model",
        "prompt_version",
        "temperature",
        "max_tokens",
        "per_repo_token_budget",
        "system_prompt_override",
    }
)


def _check_unknown_keys(data: dict[str, Any], allowed: frozenset[str], label: str) -> None:
    extra = set(data.keys()) - allowed
    if extra:
        raise ConfigError(f"unknown keys in {label}: {', '.join(sorted(extra))}")
