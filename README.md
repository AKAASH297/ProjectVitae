# ProjectVitae

Agent-orchestrated resume builder — clone GitHub repos, explore projects, and generate tailored resumes from a LaTeX template.

## Pipeline

```
GitHub URLs → Clone → Explore (per repo) → Filter → Writing (4 sections)
→ Content Critique → User Review → Export (PDF) → Compile Critique → Done
```

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Set API key
set ANTHROPIC_API_KEY=sk-...

# Run headless
python -m project_vitae run https://github.com/user/repo --jd path/to/job-description.md

# Or with TUI
python -m project_vitae setup
```

## Usage

**`run`** — full pipeline, no interactive UI:

```bash
python -m project_vitae run <url1> [<url2> ...] --jd <path> [--session <name>] [--config <path>]
```

**`setup`** — Textual TUI with screens for setup, exploration progress, filter confirmation, section review, critique, and export.

## Configuration

Edit `userprofile/config.yaml` to set per-subagent providers, models, and cost caps:

```yaml
subagents:
  explore:
    provider: anthropic
    model: claude-sonnet-4-6
    api_key_env: ANTHROPIC_API_KEY
    per_repo_token_budget: 32000
  filter:
    provider: anthropic
    model: claude-sonnet-4-6
  writing:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.5
  content_critique:
    provider: anthropic
    model: claude-sonnet-4-6
  compile_critique:
    provider: anthropic
    model: claude-sonnet-4-6

cost:
  per_session_cap_usd: 5.00

retry:
  max_attempts: 3
  backoff_seconds: [1, 2, 4]

latex:
  template_path: template.tex
  compiler: auto  # tectonic > pdflatex
```

Each subagent can use a different provider (Anthropic-native or any OpenAI-compatible endpoint).

## Directory layout

```
userprofile/
├── config.yaml              # per-subagent provider config
├── userinfo.md              # YAML front-matter + Markdown body
├── template.example.tex     # example LaTeX template with \VAR{} placeholders
├── template.tex             # your customized template (copy from example)
├── projects/                # explored project records (cross-session)
│   └── <title>/
│       ├── record.yaml      # authoritative ProjectRecord
│       ├── summary.md
│       └── tags.md
├── sessions/                # per-run state
│   ├── <name>/
│   │   ├── job_description.md
│   │   ├── resume_state.json
│   │   ├── resume.tex
│   │   ├── llm_log.jsonl
│   │   └── output/resume.pdf
│   └── .lock (per session)
├── clones/                  # cloned repos (gitignored, cleaned on success)
├── prompts/                 # versioned prompt templates
│   ├── explore/v1.md
│   ├── filter/v1.md
│   ├── writing/v1.md
│   ├── content_critique/v1.md
│   └── compile_critique/v1.md
├── sessions.db              # SQLite checkpointer (LangGraph state)
└── template.tex
```

## Required placeholders

Your LaTeX template must include these `\VAR{...}` placeholders:

| Placeholder | Section |
|---|---|
| `\VAR{experience}` | Experience entries from project records |
| `\VAR{education}` | Education from userinfo.md |
| `\VAR{skills}` | Skills from project tags + userinfo |
| `\VAR{summary}` | Career summary (generated last) |

## Project structure

```
project_vitae/
├── models.py           # Pydantic state models
├── config.py           # YAML config parser
├── io_utils.py         # Atomic I/O, prompt loader
├── latex_utils.py      # LaTeX sanitizer, placeholder validator
├── cost.py             # Cost estimation with hardcoded prices
├── llm_call.py         # Shared LLM call envelope with retry/backoff
├── session_lock.py     # PID-based session locking
├── graph.py            # LangGraph pipeline definition
├── __main__.py         # CLI entry point
├── nodes/
│   ├── preflight.py    # Config & environment validation
│   ├── clone.py        # Git clone (sequential)
│   ├── explore.py      # Repo analysis subagent
│   ├── filter_node.py  # Project selection subagent
│   ├── writing.py      # Section generation subagent
│   ├── content_critique.py  # Keyword + semantic match review
│   ├── compile_critique.py  # ATS formatting review
│   └── export_node.py  # LaTeX fill + PDF compilation
├── tui/
│   ├── app.py
│   └── screens/        # 7 Textual screens
└── tests/              # 50 pytest tests
```

## Testing

```bash
pytest
```

## Assumptions

- Public GitHub repos only (v1)
- English-language resumes (v1)
- LaTeX toolchain installed (tectonic or pdflatex)
- Anthropic API key (or OpenAI-compatible endpoint) for LLM subagents
