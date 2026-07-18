import json
from pathlib import Path

import pytest
import yaml

from project_vitae.io_utils import (
    USERPROFILE_DIR,
    atomic_write_text,
    atomic_write_bytes,
    dump_yaml,
    find_project_dir,
    get_userprofile_dir,
    load_project_records,
    load_yaml,
    parse_userinfo,
    read_text,
    save_json_model,
    serialize_userinfo,
    slugify,
    userprofile_path,
)
from project_vitae.models import ProjectRecord, ProjectVitaeError


def test_atomic_write_creates_file(tmp_path: Path):
    p = tmp_path / "sub" / "test.txt"
    atomic_write_text(p, "hello")
    assert p.read_text() == "hello"


def test_atomic_write_overwrites(tmp_path: Path):
    p = tmp_path / "test.txt"
    p.write_text("old")
    atomic_write_text(p, "new")
    assert p.read_text() == "new"


def test_atomic_write_bytes(tmp_path: Path):
    p = tmp_path / "data.bin"
    atomic_write_bytes(p, b"\x00\x01\x02")
    assert p.read_bytes() == b"\x00\x01\x02"


def test_atomic_write_creates_parent(tmp_path: Path):
    p = tmp_path / "a" / "b" / "c.txt"
    atomic_write_text(p, "x")
    assert p.exists()


def test_read_text(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_text("abc")
    assert read_text(p) == "abc"


def test_dump_yaml_and_load_yaml(tmp_path: Path):
    p = tmp_path / "data.yaml"
    data = {"key": "value", "num": 42}
    dump_yaml(p, data)
    loaded = load_yaml(p)
    assert loaded == data


def test_save_json_model(tmp_path: Path):
    from project_vitae.models import ProjectRecord
    rec = ProjectRecord(title="T", summary="S", tags=["a"], source_repo="https://github.com/u/r")
    p = tmp_path / "rec.json"
    save_json_model(p, rec)
    with open(p) as f:
        d = json.load(f)
    assert d["title"] == "T"


def test_slugify():
    assert slugify("My Project! 2") == "my-project-2"
    assert slugify("Hello World") == "hello-world"
    assert slugify("Special & % $ # _ { } ~ ^ \\ chars") == "special-chars"
    assert slugify("  spaces  ") == "spaces"
    assert slugify("") == ""


def test_userprofile_path_resolves():
    p = userprofile_path(["projects", "test"])
    assert str(p).startswith(str(USERPROFILE_DIR))
    assert str(p).endswith("test")


def test_userprofile_path_traversal_raises():
    with pytest.raises(ProjectVitaeError, match="path traversal"):
        userprofile_path(["..", "..", "etc"])


def test_parse_userinfo_no_front_matter():
    body = "Just text"
    front, b = parse_userinfo(body)
    assert front == {}
    assert b == "Just text"


def test_parse_userinfo_with_front_matter():
    text = "---\nname: John\nage: 30\n---\nBody here"
    front, b = parse_userinfo(text)
    assert front == {"name": "John", "age": 30}
    assert b == "Body here"


def test_parse_userinfo_empty_front_matter():
    text = "---\n---\nBody only"
    front, b = parse_userinfo(text)
    assert front == {}
    assert b == "Body only"


def test_parse_userinfo_malformed_yaml():
    with pytest.raises(ProjectVitaeError, match="invalid YAML front-matter"):
        parse_userinfo("---\n  [bad yaml\n---\nbody")


def test_serialize_userinfo_round_trip():
    front = {"name": "Alice", "links": ["https://example.com"]}
    body = "Some context\nabout me"
    serialized = serialize_userinfo(front, body)
    front2, body2 = parse_userinfo(serialized)
    assert front2 == front
    assert body2 == body


def test_serialize_userinfo_no_front():
    result = serialize_userinfo({}, "just body")
    assert "---" not in result


def test_find_project_dir_new(tmp_path: Path, monkeypatch):
    from project_vitae.io_utils import USERPROFILE_DIR
    monkeypatch.setattr("project_vitae.io_utils.USERPROFILE_DIR", tmp_path)
    (tmp_path / "projects").mkdir()
    result = find_project_dir("New Project")
    assert result.name == "new-project"


def test_find_project_dir_match_by_slug(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("project_vitae.io_utils.USERPROFILE_DIR", tmp_path)
    proj = tmp_path / "projects" / "existing-project"
    proj.mkdir(parents=True)
    rec = proj / "record.yaml"
    rec.write_text(yaml.safe_dump({"title": "Existing Project", "summary": "s", "tags": [], "source_repo": "https://github.com/u/r"}))
    result = find_project_dir("Existing Project")
    assert result.name == "existing-project"


def test_find_project_dir_match_by_title_case_insensitive(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("project_vitae.io_utils.USERPROFILE_DIR", tmp_path)
    proj = tmp_path / "projects" / "old-proj"
    proj.mkdir(parents=True)
    rec = proj / "record.yaml"
    rec.write_text(yaml.safe_dump({"title": "Old Proj", "summary": "s", "tags": [], "source_repo": "https://github.com/u/r"}))
    result = find_project_dir("old proj")
    assert result.name == "old-proj"


def test_load_project_records(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("project_vitae.io_utils.get_userprofile_dir", lambda: tmp_path)
    (tmp_path / "projects" / "p1").mkdir(parents=True)
    (tmp_path / "projects" / "p1" / "record.yaml").write_text(
        yaml.safe_dump({"title": "P1", "summary": "S1", "tags": ["a"], "source_repo": "https://github.com/u/r1"})
    )
    (tmp_path / "projects" / "p2").mkdir()
    (tmp_path / "projects" / "p2" / "record.yaml").write_text(
        yaml.safe_dump({"title": "P2", "summary": "S2", "tags": ["b"], "source_repo": "https://github.com/u/r2"})
    )
    records = load_project_records()
    assert len(records) == 2
    assert records[0].title == "P1"
    assert records[1].source_repo == "https://github.com/u/r2"


def test_load_project_records_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("project_vitae.io_utils.get_userprofile_dir", lambda: tmp_path)
    (tmp_path / "projects").mkdir()
    assert load_project_records() == []
