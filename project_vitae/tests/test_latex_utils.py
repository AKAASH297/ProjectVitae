from project_vitae.latex_utils import (
    fill_template,
    find_placeholders,
    sanitize_latex,
    validate_placeholders,
)


def test_sanitize_ampersand():
    assert sanitize_latex("A & B") == r"A \& B"


def test_sanitize_percent():
    assert sanitize_latex("100%") == r"100\%"


def test_sanitize_dollar():
    assert sanitize_latex("$10") == r"\$10"


def test_sanitize_hash():
    assert sanitize_latex("#1") == r"\#1"


def test_sanitize_underscore():
    assert sanitize_latex("a_b") == r"a\_b"


def test_sanitize_braces():
    assert sanitize_latex("{hello}") == r"\{hello\}"


def test_sanitize_tilde():
    assert sanitize_latex("~") == r"\textasciitilde{}"


def test_sanitize_caret():
    assert sanitize_latex("^") == r"\textasciicircum{}"


def test_sanitize_backslash():
    assert sanitize_latex("\\") == r"\textbackslash{}"


def test_sanitize_empty_string():
    assert sanitize_latex("") == ""


def test_sanitize_unicode():
    assert sanitize_latex("héllo wörld") == "héllo wörld"


def test_sanitize_already_escaped():
    assert "\\$" in sanitize_latex("$")


def test_find_placeholders():
    tmpl = r"\section*{Summary} \VAR{summary} \VAR{experience}"
    assert find_placeholders(tmpl) == {"summary", "experience"}


def test_find_placeholders_no_matches():
    assert find_placeholders(r"\section*{Hello}") == set()


def test_validate_placeholders_all_present():
    tmpl = r"\VAR{experience} \VAR{education} \VAR{skills} \VAR{summary}"
    assert validate_placeholders(tmpl) == []


def test_validate_placeholders_missing():
    tmpl = r"\VAR{experience} \VAR{skills}"
    missing = validate_placeholders(tmpl)
    assert "education" in missing
    assert "summary" in missing


def test_fill_template():
    tmpl = r"\VAR{greeting}, \VAR{name}!"
    result = fill_template(tmpl, {"greeting": "Hello", "name": "World"})
    assert result == "Hello, World!"


def test_fill_template_unknown_placeholder_preserved():
    tmpl = r"\VAR{known} and \VAR{unknown}"
    result = fill_template(tmpl, {"known": "foo"})
    assert result == r"foo and \VAR{unknown}"
