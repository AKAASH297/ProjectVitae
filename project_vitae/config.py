from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class SubagentConfig(BaseModel):
    provider: Literal["anthropic", "openai_compatible"] = "anthropic"
    base_url: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    model: str = "claude-sonnet-4-6"
    prompt_version: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    per_repo_token_budget: int | None = None
    system_prompt_override: str | None = None


class RetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_seconds: list[float] = Field(default_factory=lambda: [1, 2, 4])


class CostConfig(BaseModel):
    per_session_cap_usd: float = 5.0
    pricing_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)


class LatexConfig(BaseModel):
    template_path: str = "template.tex"
    compiler: Literal["auto", "tectonic", "pdflatex"] = "auto"


class AppConfig(BaseModel):
    subagents: dict[str, SubagentConfig] = Field(default_factory=dict)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    latex: LatexConfig = Field(default_factory=LatexConfig)
    log_level: str = "info"

    @model_validator(mode="after")
    def validate_api_keys(self):
        for name, sa in self.subagents.items():
            env_val = os.environ.get(sa.api_key_env)
            if not env_val:
                raise ValueError(
                    f"Subagent '{name}': environment variable '{sa.api_key_env}' is not set"
                )
        return self

    @classmethod
    def load(cls, path: Path | str) -> AppConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
