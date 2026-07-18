from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ProjectVitaeError(Exception): ...


class ConfigError(ProjectVitaeError): ...


class TemplateError(ProjectVitaeError): ...


class SessionLockError(ProjectVitaeError): ...


class PromptError(ProjectVitaeError): ...


class LLMCallError(ProjectVitaeError): ...


class CheckpointerError(ProjectVitaeError): ...


class CostCapReached(ProjectVitaeError): ...


class TokenBudgetExceeded(ProjectVitaeError): ...


class ProjectRecord(BaseModel):
    title: str
    summary: str
    tags: list[str]
    source_repo: str
    low_confidence: bool = False


class ExplorationResult(BaseModel):
    action: Literal["new", "update"]
    matched_project: str | None = None
    title: str
    summary: str
    tags: list[str]
    low_confidence: bool = False


class FilterResult(BaseModel):
    selected: list[str]
    rationale: str


class WritingResult(BaseModel):
    section_id: str
    content: str
    rationale: str


class SectionVersion(BaseModel):
    content: str
    feedback_used: str | None = None
    timestamp: datetime
    model: str | None = None
    provider: str
    prompt_version: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    cost_estimate: float | None = None


class ResumeSection(BaseModel):
    id: str
    kind: Literal["experience", "education", "skills", "summary"]
    versions: list[SectionVersion]
    status: Literal["draft", "approved", "needs_review"] = "draft"

    @property
    def current(self) -> SectionVersion:
        return self.versions[-1]


class Issue(BaseModel):
    location: str
    kind: Literal["content_keyword", "formatting"]
    note: str
    keyword_match: bool | None = None
    phase: Literal["content", "compile"]


class CritiqueResult(BaseModel):
    issues: list[Issue]


class SessionState(BaseModel):
    job_description: str = ""
    selected_projects: list[str] = Field(default_factory=list)
    sections: list[ResumeSection] = Field(default_factory=list)
    open_issues: list[Issue] = Field(default_factory=list)
    skipped_repos: list[str] = Field(default_factory=list)
    cost_running_usd: float = 0.0
    session_name: str = ""
    prompts_version: dict[str, str] = Field(default_factory=dict)
    github_urls: list[str] = Field(default_factory=list)
    clones_dir: str = ""
    clone_dirs: list[str] = Field(default_factory=list)
    exploration_warnings: list[dict] = Field(default_factory=list)
    current_repo_url: str | None = None
    current_exploration: ExplorationResult | None = None
    filter_proposal: FilterResult | None = None
    current_section_kind: Literal["experience", "education", "skills", "summary", None] = None
    current_feedback: str | None = None
    generated_sections_cache: dict[str, str] = Field(default_factory=dict)
    latex_compiler: str | None = None
    final_pdf: str | None = None


class LLMCallRecord(BaseModel):
    timestamp: datetime
    subagent: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    duration_ms: int
    prompt_version: str
