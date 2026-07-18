from project_vitae.models import (
    Issue,
    ProjectRecord,
    ResumeSection,
    SectionVersion,
    SessionState,
)


def test_project_record_roundtrip():
    rec = ProjectRecord(
        title="My Project",
        summary="A test project",
        tags=["python", "fastapi"],
        source_repo="https://github.com/user/project",
        low_confidence=False,
    )
    data = rec.model_dump()
    restored = ProjectRecord.model_validate(data)
    assert restored.title == "My Project"
    assert restored.source_repo == "https://github.com/user/project"
    assert restored.low_confidence is False


def test_resume_section_current_property():
    v1 = SectionVersion(content="v1 content", provider="ai")
    v2 = SectionVersion(content="v2 content", provider="ai")
    section = ResumeSection(id="experience", kind="experience", versions=[v1, v2])
    assert section.current.content == "v2 content"


def test_section_status_transitions_draft():
    section = ResumeSection(id="skills", kind="skills", versions=[])
    assert section.status == "draft"


def test_issue_construction():
    issue = Issue(
        location="experience",
        kind="content_keyword",
        note="Missing 'machine learning'",
        keyword_match=True,
        phase="content",
    )
    assert issue.kind == "content_keyword"
    assert issue.keyword_match is True


def test_session_state_defaults():
    state = SessionState()
    assert state.job_description == ""
    assert state.selected_projects == []
    assert state.sections == []
    assert state.open_issues == []
    assert state.skipped_repos == []


def test_session_state_with_data():
    section = ResumeSection(id="exp", kind="experience", versions=[SectionVersion(content="test", provider="ai")])
    issue = Issue(location="global", kind="formatting", note="test", phase="compile")
    state = SessionState(
        job_description="Software engineer role",
        selected_projects=["proj1"],
        sections=[section],
        open_issues=[issue],
        skipped_repos=["https://github.com/empty/repo"],
    )
    assert len(state.sections) == 1
    assert state.sections[0].current.content == "test"
    assert "proj1" in state.selected_projects


def test_section_version_nullable_fields():
    v = SectionVersion(content="hello", provider="manual")
    assert v.model is None
    assert v.feedback_used is None
    assert v.temperature is None
    assert v.max_tokens is None
    assert v.cost_estimate is None


def test_exploration_result_new():
    from project_vitae.models import ExplorationResult
    r = ExplorationResult(action="new", title="NewProj", summary="desc", tags=["go"])
    assert r.action == "new"
    assert r.low_confidence is False
