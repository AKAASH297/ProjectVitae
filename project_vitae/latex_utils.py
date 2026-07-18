import re

LATEX_SPECIAL = {"&", "%", "$", "#", "_", "{", "}", "~", "^", "\\"}


def sanitize_latex(text: str) -> str:
    chars = []
    for ch in text:
        if ch == "\\":
            chars.append("\\textbackslash{}")
        elif ch == "~":
            chars.append("\\textasciitilde{}")
        elif ch == "^":
            chars.append("\\textasciicircum{}")
        elif ch in LATEX_SPECIAL:
            chars.append("\\" + ch)
        else:
            chars.append(ch)
    return "".join(chars)


PLACEHOLDER_RE = re.compile(r"\\VAR\{(\w+)\}")


def find_placeholders(template: str) -> set[str]:
    return set(PLACEHOLDER_RE.findall(template))


REQUIRED_PLACEHOLDERS = {"experience", "education", "skills", "summary"}


def validate_placeholders(template: str) -> list[str]:
    found = find_placeholders(template)
    missing = REQUIRED_PLACEHOLDERS - found
    return sorted(missing)


def fill_template(template: str, sections: dict[str, str]) -> str:
    def replacer(m: re.Match) -> str:
        key = m.group(1)
        return sections.get(key, m.group(0))
    return PLACEHOLDER_RE.sub(replacer, template)
