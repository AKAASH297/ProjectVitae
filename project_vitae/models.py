from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ProjectRecord(BaseModel):
    title: str
    summary: str
    tags: list[str]
    source_repo: str
    low_confidence: bool = False


class SectionVersion(BaseModel):
    content: str
    feedback_used: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    model: str | None = None
    provider: str = ""
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


class CritiqueResult(BaseModel):
    issues: list[Issue]


class SessionState(BaseModel):
    job_description: str = ""
    selected_projects: list[str] = Field(default_factory=list)
    sections: list[ResumeSection] = Field(default_factory=list)
    open_issues: list[Issue] = Field(default_factory=list)
    skipped_repos: list[str] = Field(default_factory=list)
