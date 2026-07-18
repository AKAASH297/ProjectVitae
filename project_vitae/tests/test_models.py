from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from project_vitae.models import (
    CritiqueResult,
    ExplorationResult,
    FilterResult,
    Issue,
    ProjectRecord,
    ResumeSection,
    SectionVersion,
    SessionState,
    WritingResult,
)


def test_project_record_defaults():
    r = ProjectRecord(
        title="My Project",
        summary="A summary",
        tags=["python"],
        source_repo="https://github.com/user/repo",
    )
    assert r.low_confidence is False
    assert r.title == "My Project"


def test_project_record_round_trip():
    r = ProjectRecord(
        title="T",
        summary="S",
        tags=["a"],
        source_repo="https://github.com/u/r",
        low_confidence=True,
    )
    d = r.model_dump()
    r2 = ProjectRecord.model_validate(d)
    assert r2 == r


def test_exploration_result():
    r = ExplorationResult(action="new", title="P", summary="S", tags=["a"], low_confidence=True)
    assert r.action == "new"
    assert r.matched_project is None
    r2 = ExplorationResult(action="update", matched_project="P", title="P", summary="S", tags=["a"])
    assert r2.matched_project == "P"


def test_filter_result():
    r = FilterResult(selected=["proj1", "proj2"], rationale="best match")
    assert len(r.selected) == 2
    assert r.rationale == "best match"


def test_writing_result():
    r = WritingResult(section_id="exp1", content="Worked on X", rationale="relevant")
    assert r.section_id == "exp1"


def test_section_version_defaults():
    now = datetime.now(timezone.utc)
    v = SectionVersion(content="text", timestamp=now, provider="anthropic")
    assert v.feedback_used is None
    assert v.model is None
    assert v.temperature is None
    assert v.cost_estimate is None
    assert v.provider == "anthropic"


def test_section_version_manual_edit():
    now = datetime.now(timezone.utc)
    v = SectionVersion(content="text", timestamp=now, provider="manual")
    assert v.provider == "manual"
    assert v.model is None


def test_resume_section_current():
    now = datetime.now(timezone.utc)
    v1 = SectionVersion(content="v1", timestamp=now, provider="anthropic")
    v2 = SectionVersion(content="v2", timestamp=now, provider="anthropic")
    s = ResumeSection(id="exp1", kind="experience", versions=[v1, v2])
    assert s.current.content == "v2"
    assert s.status == "draft"


def test_resume_section_status_transitions():
    now = datetime.now(timezone.utc)
    v = SectionVersion(content="c", timestamp=now, provider="anthropic")
    s = ResumeSection(id="s1", kind="summary", versions=[v], status="approved")
    assert s.status == "approved"
    s.status = "needs_review"
    assert s.status == "needs_review"


def test_issue():
    i = Issue(
        location="exp1",
        kind="content_keyword",
        note="missing term",
        keyword_match=False,
        phase="content",
    )
    assert i.keyword_match is False
    assert i.phase == "content"


def test_issue_defaults():
    i = Issue(location="global", kind="formatting", note="bad font", phase="compile")
    assert i.keyword_match is None


def test_critique_result():
    issues = [
        Issue(
            location="exp1", kind="content_keyword", note="x", keyword_match=True, phase="content"
        )
    ]
    r = CritiqueResult(issues=issues)
    assert len(r.issues) == 1


def test_session_state_defaults():
    s = SessionState()
    assert s.job_description == ""
    assert s.selected_projects == []
    assert s.sections == []
    assert s.open_issues == []
    assert s.cost_running_usd == 0.0


def test_session_state_round_trip():
    now = datetime.now(timezone.utc)
    v = SectionVersion(content="c", timestamp=now, provider="anthropic")
    section = ResumeSection(id="exp1", kind="experience", versions=[v])
    state = SessionState(
        job_description="JD",
        selected_projects=["p1"],
        sections=[section],
        session_name="test-session",
    )
    d = state.model_dump()
    state2 = SessionState.model_validate(d)
    assert state2.session_name == "test-session"
    assert len(state2.sections) == 1
    assert state2.sections[0].current.content == "c"


def test_invalid_section_kind():
    now = datetime.now(timezone.utc)
    v = SectionVersion(content="c", timestamp=now, provider="anthropic")
    with pytest.raises(ValidationError):
        ResumeSection(id="x", kind="invalid_kind", versions=[v])


def test_invalid_issue_kind():
    with pytest.raises(ValidationError):
        Issue(location="x", kind="bad_kind", note="x", phase="content")


def test_invalid_phase():
    with pytest.raises(ValidationError):
        Issue(location="x", kind="formatting", note="x", phase="bad_phase")


def test_session_state_serialization_extra_fields():
    state = SessionState(github_urls=["https://github.com/user/repo"])
    d = state.model_dump()
    assert "github_urls" in d
    state2 = SessionState.model_validate(d)
    assert state2.github_urls == ["https://github.com/user/repo"]


def test_section_version_cost_estimate():
    now = datetime.now(timezone.utc)
    v = SectionVersion(content="c", timestamp=now, provider="anthropic", cost_estimate=0.05)
    assert v.cost_estimate == 0.05
