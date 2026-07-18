from pathlib import Path

from project_vitae.io_utils import atomic_write, load_prompt, read_text, slugify


def test_atomic_write_creates_file(tmp_path):
    path = tmp_path / "test.txt"
    atomic_write(path, "hello world")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "hello world"


def test_atomic_write_overwrites(tmp_path):
    path = tmp_path / "test.txt"
    atomic_write(path, "first")
    atomic_write(path, "second")
    assert path.read_text(encoding="utf-8") == "second"


def test_atomic_write_creates_parent_dirs(tmp_path):
    path = tmp_path / "a" / "b" / "c" / "test.txt"
    atomic_write(path, "nested")
    assert path.exists()


def test_read_text(tmp_path):
    path = tmp_path / "foo.txt"
    path.write_text("bar", encoding="utf-8")
    assert read_text(path) == "bar"


def test_load_prompt(tmp_path):
    prompt_dir = tmp_path / "prompts" / "test"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = prompt_dir / "v1.md"
    path.write_text("You are a helpful assistant.", encoding="utf-8")
    assert load_prompt(path) == "You are a helpful assistant."


def test_slugify_simple():
    assert slugify("Hello World") == "hello-world"


def test_slugify_special_chars():
    assert slugify("My Project! @#$") == "my-project"


def test_slugify_multiple_spaces():
    assert slugify("  a  b  ") == "a-b"


def test_slugify_empty():
    assert slugify("") == "untitled"


def test_slugify_leading_trailing():
    assert slugify("  --hello--  ") == "hello"
