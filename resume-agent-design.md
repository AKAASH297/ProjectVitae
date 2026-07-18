# Agent-Orchestration Resume Tool — Design Spec

Status: v3 — all gaps identified during design review have been resolved.
Everything is architecture as designed.

**Revision note (v3):** This version closes remaining gaps from the v2
review. Changes: `ProjectRecord` gains on-disk `record.yaml` sidecar with
`source_repo` dedup key and `low_confidence` flag; pre-flight node restricted
to template-structure validation (content-binding deferred to Export); cost
estimation strategy defined (hardcoded price table, zero-cost for local
providers); `per_repo_token_budget` added to Explore config; `.gitignore`
expanded to cover checkpointer DB, session state, and JD files; `Exploration
Result.action` typed as `Literal["new","update"]`; `SectionVersion` allows
`null` fields for manual edits; `ResumeSection.status` transitions defined;
Compile Critique re-pass made explicit (includes re-Export); Content Critique
runs on draft content with deterministic keyword-overlap computed in-node
before LLM semantic pass; retry distinguishes recoverable (429/5xx) from
non-recoverable errors; session lock includes carve-out for resumed sessions;
`userinfo.md` YAML front-matter schema defined; Compile Critique input
mechanism specified (filled `.tex`, not PDF); `skipped_repos` tracking added
to `SessionState`; many low-severity edge cases closed. See diff for full
change set.

---

## 1. Pipeline (end to end)

```
GitHub URLs (user input)
  │
  ▼
Clone repos — sequential, one at a time, plain git, no LLM
  │
  ▼
For each repo (one Explore Subagent at a time, spawn → work → die):
  - reads: cloned repo (list_dir, read_file, grep) + userprofile/projects/
  - dedup: exact match on source_repo URL only, no fuzzy fallback
  - decides: new project, or update to existing one (by URL match)
  - writes summary.md + tags.md directly into userprofile/projects/<title>/
  - no version history kept on project records — new write replaces old
  - tool-call errors: subagent retries itself (3 attempts,
    exponential backoff 1s/2s/4s), then reports failure up to
    Orchestration Agent; retry logic applied uniformly via the
    shared LLMCall envelope (§2.7)
  - unrecoverable failure (bad clone, unreadable repo, exhausted retries):
    Orchestration Agent flags the repo, logs the error, and continues
    to the next repo. An end-of-phase summary lists all failures with
    per-repo skip/abort choices the user enters as they occur.
  - empty / docs-only repos: Explore returns low_confidence: true
    on ExplorationResult; user sees a warning on the filter screen
    and can skip the project
  │
  ▼
Orchestration Agent hands job description (JD) to Filter Subagent
  │
  ▼
Filter Subagent — hard gate: user must confirm selection before proceeding
  - input: JD + all ProjectRecords (title, summary, tags) from userprofile/
  - output: selected subset of projects to include (include/exclude only,
    no ranking/ordering in v1)
  │
  ▼
Writing Subagent — invoked per section
  - input: filtered projects + JD + userinfo.md + (on regenerate) previous
    version + user's free-text feedback
  - output: proposed section content + short rationale
  - each invocation appends a new SectionVersion; nothing is overwritten
  - produces all four section kinds (§2.2 producer table); LaTeX-special
    characters in generated content are escaped by a deterministic sanitizer
    at export time, not by the subagent
  │
  ▼
Content Critique Subagent (pre-export)
  - runs on current draft content (before the user review loop); if the
    user re-enters this phase via a re-pass, Critique re-evaluates the
    updated drafts
  - input: all section current draft content + JD
  - output: keyword-match report — the node first computes a deterministic
    keyword-overlap list (plain string matching of JD terms against section
    text), then runs the LLM semantic-match pass; both are combined into a
    single report with separate labeled sections
  - on flagged issue: user picks per-issue — dismiss, or trigger a
    Writing Subagent re-pass using the critique's note as feedback
  │
  ▼
User review loop (batch style, via TUI)
  - per section: approve / regenerate with feedback / manual edit
  - full diff view against previous version available
  - Content Critique issues shown per-section alongside action buttons
  - Content Critique runs on draft content before this loop;
    approved sections may still have unresolved issues (user decides)
  │
  ▼
Export
  - content-to-binding validation: template placeholders checked against
    the compiled-section set; LaTeX-special chars sanitized (see §2.9)
  - fills user-provided .tex template with approved section content
  - compiles to PDF via local LaTeX toolchain (tectonic preferred, fallback
    to pdflatex — detected at startup, cached in session state)
  - purely deterministic templating step, no LLM involved
  │
  ▼
Compile Critique Subagent (post-export)
  - input: filled `.tex` file (text extraction from PDF is unreliable with
    local provider setups) + JD
  - formatting check: LLM reads the rendered output and judges
    ATS-friendliness (no deterministic linter in v1)
  - on flagged issue: user picks per-issue — dismiss, or trigger a
    Writing Subagent re-pass (loops back through Content Critique →
    review loop → **re-Export** → Compile Critique; re-Export is explicit
    since changed content must be re-sanitized and re-compiled)
  │
  ▼
Done — final PDF at sessions/<session-name>/output/resume.pdf
```

---

## 2. Components

### 2.1 Orchestration Agent
- Not a general-purpose agent — a coded control-flow layer that dispatches
  subagents in sequence, owns session/profile state, and is the only thing
  that talks to the TUI.
- Tools it exposes to itself are really just internal function calls:
  `spawn_subagent(kind, task, context)`, `read_session_state()`,
  `write_session_state()`, `request_user_input(prompt, options)`.
- No model call is required for the Orchestration Agent's own control flow
  — it's plain Python logic branching on subagent output (implemented as
  LangGraph conditional edges), not an LLM deciding "what to do next."
  This design decision is confirmed: a coded state machine is simpler,
  cheaper, and fully predictable for a pipeline this well-defined.

### 2.2 Subagents (all one-shot: spawn → run tool loop → return structured result → die)

| Subagent | Runs | Tools | Output |
|---|---|---|---|
| Explore | once per repo, sequential | `list_dir`, `read_file`, `grep` (repo + userprofile), `write_file` (own project folder only) | `ExplorationResult{action: Literal["new","update"], matched_project, title, summary, tags, low_confidence: bool}` |
| Filter | once per session | `read_project_records` | `FilterResult{selected: list[str], rationale}` |
| Writing | once per section per (re)generation | `read_section`, `read_selected_projects`, `read_job_description`, `read_userinfo` | `WritingResult{section_id, content, rationale}` |
| Content Critique | once per review pass (pre-export, on current draft content before user review loop) | `read_section`, `read_job_description` | `CritiqueResult{issues: list[Issue]}` where `Issue{location, kind, note, keyword_match: bool, phase: "content"}` — the node computes the deterministic keyword-overlap list itself (plain string matching on JD terms vs section text) and uses the LLM only for the semantic-match pass; both are combined into a single `CritiqueResult`. |
| Compile Critique | once per export pass (post-export) | `read_rendered_output` (reads the filled `.tex` file — text extraction from PDF would be unreliable with local provider setups that lack vision), `read_job_description` | `CritiqueResult{issues: list[Issue]}` where `Issue{location, kind, note, phase: "compile"}` |

**Section-kind → producer mapping.** All four section kinds are produced
by the Writing Subagent:
- **experience** — grounded in `ProjectRecord` (summary + tags per project);
  one or more projects produce one experience section (multiple entries).
- **education** — drawn from `userinfo.md`.
- **skills** — compiled from `ProjectRecord.tags` across all selected projects
  plus any explicitly listed skills in `userinfo.md`.
- **summary** — synthesized from `userinfo.md` + overall project context;
  produced last so it can reference the experience/skills content.

Generation order: experience → education → skills → summary.

None of the subagents call each other or know the TUI exists. All context
passed in is a deliberately narrow slice (e.g. Writing Subagent never sees
the raw `userprofile/` — only what Filter selected).

### 2.3 State model

**Durable, cross-session — `userprofile/`**
```python
class ProjectRecord(BaseModel):
    title: str
    summary: str
    tags: list[str]
    source_repo: str   # used as the exact-match dedup key
    low_confidence: bool = False  # flagged by Explore for empty/docs-only repos
```
No version history — current file state is the only state.
Persisted as `userprofile/projects/<title>/record.yaml` (YAML serialization of
the `ProjectRecord` fields above). The `summary.md` and `tags.md` files are a
human-readable mirror of the structured data; `record.yaml` is the authoritative
source for `source_repo` (the dedup key) and `low_confidence`.

**Per-session — `sessions/<name>/resume_state.json`**
```python
class SectionVersion(BaseModel):
    content: str
    feedback_used: str | None
    timestamp: datetime
    model: str | None               # model used, e.g. "claude-sonnet-4-6";
                                    # null for manual edits
    provider: str                   # e.g. "anthropic", "openai_compatible",
                                    # or "manual" for user-driven edits
    prompt_version: str | None      # prompt registry path/version;
                                    # null for manual edits
    temperature: float | None       # null for manual edits
    max_tokens: int | None          # null for manual edits
    cost_estimate: float | None     # estimated USD cost of this call

class ResumeSection(BaseModel):
    id: str
    kind: Literal["experience", "education", "skills", "summary"]
    versions: list[SectionVersion]
    status: Literal["draft", "approved", "needs_review"]
    # status transitions:
    # - new section → "draft"
    # - user approves → "approved"
    # - regenerate + new version appended → resets to "draft"
    # - manual edit + new version appended → resets to "draft"
    # - Content Critique flags an issue → "needs_review"
    # - "current" in @property below refers to `versions[-1]`,
    #   which is the latest version regardless of which was approved

    @property
    def current(self) -> SectionVersion:
        return self.versions[-1]

class Issue(BaseModel):
    location: str          # section id or "global"
    kind: Literal["content_keyword", "formatting"]
    note: str
    keyword_match: bool | None   # only meaningful for kind == "content_keyword"
    phase: Literal["content", "compile"]

class SessionState(BaseModel):
    job_description: str           # canonical copy — read from
                                    # job_description.md at session start,
                                    # written back on edit (the .md file is
                                    # the human-editable source; this field
                                    # is the in-memory canonical version)
    selected_projects: list[str]
    sections: list[ResumeSection]
    open_issues: list[Issue]
    skipped_repos: list[str]       # repos the user chose to skip during
                                    # exploration; logged so the end-of-phase
                                    # summary can display them and re-runs
                                    # know which were skipped
```

### 2.4 UserInfo

- `userprofile/userinfo.md` holds resume-relevant facts about the user
  that have nothing to do with any GitHub repo: name, email, phone,
  location, links (LinkedIn/portfolio/etc.), education, certifications,
  and anything else the user wants available as context.
- **Dual storage: YAML front-matter + Markdown body.** Structured fields
  are stored as YAML front-matter between `---` delimiters at the top of
  the file. Unstructured context the user wants to provide (career narrative,
  personal projects, volunteer work, etc.) goes in the Markdown body below.
- **YAML front-matter schema** (hardcoded fields the TUI form knows about):
  `name`, `email`, `phone`, `location`, `links` (list of URLs),
  `education` (list of entries with `institution`, `degree`, `year`),
  `certifications` (list). New fields can be added to the YAML by
  hand-editing the file; the form will preserve unknown keys on save.
- **Collected via TUI form** — the Orchestration Agent renders a form that
  edits the YAML front-matter fields directly (a setup-time or on-demand
  form screen). The Markdown body is presented as a free-text editor area.
  The round-trip is reliable because the YAML front-matter is parseable.
- Nodes that need this context (Writing Subagent, Content Critique) read
  the whole file as plain text; the front-matter is not expected to read
  naturally as prose — it is treated as raw text context by the subagent,
  and parsed by the TUI form. No Pydantic model for this file's contents
  — it's context, not state the pipeline branches on.
- No versioning — same rationale as ProjectRecord (current state only,
  overwritten on edit via the TUI form).
- Editable later — TUI should offer a way to re-open/update the form
  without re-running the whole pipeline.

### 2.5 Directory layout
```
userprofile/
  ├── userinfo.md          # YAML front-matter + Markdown body; form-editable
  │                        # via TUI; read as plain context by Writing Subagent
  │                        # and Content Critique; never generated/inferred.
  ├── projects/
  │   └── <project-title>/  # directory name = title slugified (spaces → hyphens,
  │                        # path-illegal chars removed); collisions disambiguated
  │                        # by appending a numeric suffix since dedup is on
  │                        # source_repo, not title
  │       ├── record.yaml       # authoritative ProjectRecord (title, summary, tags,
  │       │                     # source_repo, low_confidence) — the dedup key
  │       │                     # source_repo lives here; summary.md + tags.md are
  │       │                     # human-readable mirrors
  │       ├── summary.md        # freeform Markdown — ExplorationResult.summary
  │       └── tags.md           # one tag per line — ExplorationResult.tags
  ├── config.yaml
  ├── sessions/
  │   └── <session-name>/
  │       ├── job_description.md
  │       ├── resume_state.json
  │       ├── resume.tex          # filled from user template
  │       ├── llm_log.jsonl       # structured log of every LLM call
  │       └── output/
  │           └── resume.pdf
  ├── clones/              # cloned repos (gitignored), per-repo subdirectory
  │   └── <repo-name>/     # cleaned up after successful exploration; kept
  │                        # on failure for debugging
  └── prompts/             # versioned prompt templates (§2.7)
      ├── explore/
      │   └── v1.md
      ├── filter/
      │   └── v1.md
      ├── writing/
      │   └── v1.md
      ├── content_critique/
      │   └── v1.md
      └── compile_critique/
          └── v1.md
```

**.gitignore entries for `userprofile/`:**
- `clones/` — cloned repo contents are ephemeral build artifacts.
- `sessions/*/output/` — compiled PDFs / aux LaTeX files.
- `sessions.db` — SQLite checkpointer database (contains JD text and
  section content via LangGraph checkpoint state).
- `sessions/<name>/resume_state.json` — contains `SessionState.job_description`
  and all section content.
- `sessions/<name>/job_description.md` — the raw job description text.
- `userinfo.md` — contains PII (name, email, phone). Users should commit
  a `.example` version instead if they version-control `userprofile/`.

### 2.6 Config — per-subagent provider

Each subagent can point at a **different provider, base URL, and model** —
not a single global setting. Providers supported: Anthropic-native API, or
any OpenAI-compatible endpoint (custom `base_url` — covers vLLM, Ollama,
OpenRouter, Together, etc.). API keys are referenced by **environment
variable name**, never stored as raw values in config.

Each subagent also specifies its **prompt version** (path under `prompts/`),
**temperature**, **max_tokens**, and (optionally) a **system_prompt_override**
for hot-fixing prompts without a full version bump.

Top-level settings include a **per-session cost cap** (hard stop if total LLM
spend exceeds this amount) and a **log_level** for the Orchestration Agent.

```yaml
# userprofile/config.yaml
subagents:
  explore:
    provider: anthropic            # or "openai_compatible"
    base_url: null                  # null = provider default
    api_key_env: ANTHROPIC_API_KEY
    model: claude-sonnet-4-6
    prompt_version: prompts/explore/v1.md
    temperature: 0.3
    max_tokens: 4096
    per_repo_token_budget: 32000   # hard cap on total tokens (input + output)
                                   # across all tool-call rounds for one repo;
                                   # prevents runaway cost on large repos
  filter:
    provider: anthropic
    base_url: null
    api_key_env: ANTHROPIC_API_KEY
    model: claude-sonnet-4-6
    prompt_version: prompts/filter/v1.md
    temperature: 0.1
    max_tokens: 2048
  writing:
    provider: anthropic
    base_url: null
    api_key_env: ANTHROPIC_API_KEY
    model: claude-sonnet-4-6
    prompt_version: prompts/writing/v1.md
    temperature: 0.5
    max_tokens: 8192
  content_critique:
    provider: anthropic
    base_url: null
    api_key_env: ANTHROPIC_API_KEY
    model: claude-sonnet-4-6
    prompt_version: prompts/content_critique/v1.md
    temperature: 0.2
    max_tokens: 4096
  compile_critique:
    provider: anthropic
    base_url: null
    api_key_env: ANTHROPIC_API_KEY
    model: claude-sonnet-4-6
    prompt_version: prompts/compile_critique/v1.md
    temperature: 0.2
    max_tokens: 4096

retry:
  max_attempts: 3            # applied via shared LLMCall envelope; distinguishes
                             # recoverable errors (429, 5xx, network timeouts) from
                             # non-recoverable (4xx auth/quota, schema mismatch):
                             # non-recoverable errors fail immediately without retry
  backoff_seconds: [1, 2, 4]

cost:
  per_session_cap_usd: 5.00   # hard stop; user adjusts in config
  # Pricing sources: Anthropic and OpenAI models use published per-token rates
  # hardcoded in the LLMCall wrapper (model → {input_price, output_price} map).
  # OpenAI-compatible endpoints (vLLM/Ollama custom base_url) have no real USD
  # cost; their cost is reported as 0 and excluded from the cap. Unknown models
  # in the hardcoded table log a warning and are treated as cost 0. Users can
  # override prices via a `pricing.overrides` map in config if needed.

latex:
  template_path: template.tex  # relative to userprofile/; user customizes
                               # from template.example.tex
  compiler: auto              # auto-detect tectonic, fallback to pdflatex

log_level: info               # debug | info | warning | error
```

### 2.7 Orchestration framework: LangGraph

Build on **LangGraph** with a **SQLite-based checkpointer** for persistent
state across TUI restarts. Rationale — the pipeline is a fixed, fully-specified
sequence (not dynamic/planner-driven), which maps cleanly onto LangGraph's
graph model: each pipeline stage is a node, the sequence in §1 is the edge
path, and the skip/abort and dismiss/re-pass decisions are conditional edges.

#### What LangGraph provides directly
- **Tool-calling loop** — model call → execute tool calls → feed results
  back → repeat; no need to hand-roll per subagent.
- **Multi-provider model binding** via LangChain's chat model interface
  (`ChatAnthropic`, `ChatOpenAI` with custom `base_url`). Per-subagent
  config (§2.6) is bound at the node level.
- **Structured output** via `.with_structured_output(PydanticModel)` —
  handles Anthropic-vs-OpenAI-compatible differences internally.
- **Human-in-the-loop** via `interrupt()` / checkpointing — graph pauses
  at the failure-contingency, filter-confirmation, section-review, and
  critique-dismiss/re-pass points, then resumes once the TUI collects
  an answer. **Checkpointer = `SqliteSaver`** at `userprofile/sessions.db`,
  enabling resume across TUI restarts (abandoned/paused sessions are
  listed on the setup screen with a "resume" or "discard" option;
  cleanup via `sessions.db` entry removal + `sessions/<name>/` directory
  deletion). **A single active session lock** is enforced — on startup,
  check if any session has an unexpired `.lock` file before allowing a run.

#### Graph structure

1. **Pre-flight node** (graph entry, no LLM call): validates config file
   parses, all `api_key_env` env vars are set, `latex.template_path` exists
   and passes a **template-structure** sanity check (the template contains
   each of the 4 known section-kind placeholders; additional user-defined
   `\VAR{...}` placeholders are allowed — they pass through unchanged and
   appear in the final PDF as-is, which the user can fill or remove from
   their template). LaTeX toolchain is available (tectonic or pdflatex, cached
   after startup). Content-to-placeholder binding is NOT checked here (no
   sections compiled yet); that belongs in the Export node (§2.9). Fail
   fast with a descriptive error message — do not spend any tokens yet.
2. **Clone node** (deterministic, no LLM): sequential plain `git clone` into
   `userprofile/clones/<repo-name>/`. Repo size is checked via GitHub REST
   API (`/repos/{owner}/{repo}` returns `size` in KB) before cloning when
   the URL is a GitHub-hosted public repo; for non-GitHub URLs or when the
   API call fails, the size check is best-effort (repos >200 MB guessed
   during clone may be aborted partway). Repos known or suspected >200 MB
   emit a warning row on the exploration screen. Public-URL clones only
   in v1.
3. **Explore nodes** — one per repo, sequential. Each is a subgraph with:
   tool-calling loop (Explore Subagent), tool allow-list restricting file
   reads to `<clone-dir>/` and `userprofile/projects/`, and file writes
   to `userprofile/projects/<title>/` only.
4. **Filter node** (LLM call) → **hard-gate interrupt** (user confirms).
5. **Writing nodes** — one per section kind, in order: experience →
   education → skills → summary.
6. **Content Critique node** — reads current draft section content + JD,
   produces keyword and semantic-match issues (keyword-overlap is computed
   deterministically in-node before the LLM semantic pass).
7. **Review loop interrupt** — per-section approve / regenerate / manual edit.
8. **Export node** (no LLM): placeholder-validation (content-to-binding
   check, separate from pre-flight's template-structure check), LaTeX
   sanitizer, template fill, compile (shell out to tectonic or pdflatex).
9. **Compile Critique node** — reads the filled `.tex` (not the PDF; see
   §2.2 tool column), produces formatting issues. If issues exist →
   conditional edge back to Writing node (re-generate affected sections),
   then through Content Critique → review loop → **re-Export** → Compile
   Critique. The re-export step is explicit: changed content must be
   re-sanitized and re-compiled before the critique can re-evaluate.

#### Shared LLMCall envelope (all subagents)

Rather than ad-hoc retry code per node, all LLM-invoking nodes go through
a uniform `LLMCall` wrapper:

- **Retry/backoff** — §2.6 retry config applied identically to Explore,
  Filter, Writing, Content Critique, and Compile Critique subagents.
- **Token budget** — each subagent's `max_tokens` from config enforced at
  the call level; Explore additionally has a per-repo total-token budget
  (sum of input + output across all tool loops) to prevent runaway cost
  on a large repo.
- **Cost cap** — a running USD total (tracked in graph state) is checked
  before each LLM call; if `cost.per_session_cap_usd` is exceeded, the
  graph pauses and alerts the user (option to increase cap or abort).
- **Observability** — every call writes a structured JSONL record to
  `sessions/<session-name>/llm_log.jsonl` with fields: `{timestamp,
  subagent, model, input_tokens, output_tokens, cost, duration_ms,
  prompt_version}`. The running USD total is persisted in
  `SessionState`.
- **Structured output** — `.with_structured_output()` is used for all
  subagents. The wrapper detects provider type and uses the correct mode
  (tool-based extraction for Anthropic, `json_schema` mode for OpenAI-
  compatible).

#### Security

- **Explore tool sandbox** — file-read tools (`list_dir`, `read_file`, `grep`)
  are restricted to an explicit allow-list: `<clone-dir>/` and
  `userprofile/projects/`. Any attempt to read outside these paths returns
  a permission-denied error. File-write tools are restricted to
  `userprofile/projects/<title>/` only.
- **Prompt injection** — repo `README`, comments, and source code are
  untrusted. The Explore subagent's system prompt includes a directive
  that `summary.md` and `tags.md` must reflect the subagent's own analysis
  (not verbatim repo content), and that repo instructions asking the agent
  to perform arbitrary actions are to be ignored. No subagent downstream
  of Explore reads raw repo files — only the curated `summary.md`/`tags.md`.
  The `summary.md`/`tags.md` files are treated as model output (subject to
  sanity review) rather than direct passthrough.

#### I/O conventions

- **Atomic file writes** — all JSON, YAML, and Markdown state files
  (`resume_state.json`, `record.yaml`, `summary.md`, `tags.md`) are
  written via a helper:
  write to a `.tmp` sibling first, then `os.replace()` (atomic on same
  filesystem on Windows). This prevents partial reads by other nodes.
- **Single-session lock** — `sessions/<name>/.lock` containing PID +
  timestamp prevents concurrent runs against the same session. Stale locks
  (>1 hour) can be cleared manually via the TUI setup screen.
  **Carve-out for resume:** when the TUI starts and the user selects a
  paused/resumed session from the setup-screen list, the lock check is
  skipped for that session (resuming is the unlock path). The lock is
  re-acquired after resume completes.
- **Clones cleanup** — after successful exploration of all repos, the
  `clones/` directory is removed. If any clone failed or was skipped by
  the user, the clone is retained in `clones/` for debugging and noted
  in the end-of-exploration summary. Skipped repos that haven't been
  cloned at all (pre-clone skip or >200 MB abort) leave no residue.
- **Skip state persistence** — when a user skips or aborts a repo during
  exploration, the skip decision is logged in `SessionState` (as a
  `skipped_repos` list) so the end-of-phase summary can display it and
  subsequent re-runs know which repos were skipped rather than silently
  re-cloning without context.

#### Prompt registry

Prompts are versioned files under `userprofile/prompts/<subagent>/<vN>.md`.
Each subagent's config points to its current prompt path; `SectionVersion`
records the exact prompt version used for provenance. Prompts live next
to the user's other data so they can be hand-edited without a code deploy.
A change in prompt path constitutes a version change — the old path stays
in the filesystem for reproducibility of past sessions.

#### I/O boundary

- `userprofile/` (durable, cross-session) and `userinfo.md` (§2.4) live
  **outside** the graph's own state — they're plain file I/O performed
  inside node functions, not part of LangGraph's per-run state object.
  Only session-scoped data (§2.3's `SessionState`) belongs in graph state.
- The graph definition + conditional-edge logic **is** the Orchestration
  Agent now — the coded control-flow layer described in §2.1. The graph
  wiring replaces the informal "internal tool calls" sketch shown in §2.1;
  §2.1's intent (orchestration owns state, dispatches in sequence, is the
  only thing the TUI talks to) is preserved, implemented via LangGraph
  nodes and edges rather than imperative Python.

### 2.8 TUI (Textual)
1. **Setup screen** — paste GitHub URLs, paste/select JD, choose or
   confirm session name; also list abandoned/paused sessions with
   resume/discard options. If the chosen session name already has a
   `sessions/<name>/` directory, the user confirms overwrite or
   picks a different name (existing projects in `userprofile/` are
   shared and unaffected).
2. **Exploration screen** — one row per repo, sequential progress
   (spinner → ✅ / ❌ / ⚠️ for low-confidence / ❌ for failed),
   inline modal if a repo fails asking skip/abort.
3. **Filter result screen** — shows selected projects + rationale;
   **hard gate** — user must explicitly click "Confirm & Continue"
   before the pipeline proceeds. Projects with `low_confidence: true`
   are highlighted with a warning icon.
4. **Section review screen** — list of sections with status; per-section
   actions: approve, regenerate (opens feedback text input), manual edit,
   diff-against-previous.
5. **Content Critique screen** — list of flagged issues (keyword-overlap
   and semantic-match labeled separately), per-issue action: dismiss /
   send to Writing Subagent with note.
6. **Export screen** — shows compilation progress, final PDF path, and
   LaTeX log excerpt on failure; option to re-open review loop.
7. **Compile Critique screen** — formatting-only issues on rendered
   output; per-issue action: dismiss / trigger re-pass (loops back
   through Content Critique and review).

### 2.9 Export
- Requires `latex.template_path` set in config (path to the user's
  `.tex` template file) — this is the one hard external dependency
  still pending from you. Before Export runs, the template is copied
  into `sessions/<session-name>/resume.tex`; placeholder filling
  and compilation operate on that copy.
- Template is expected to expose named placeholders per section,
  one per section kind: `\VAR{experience}`, `\VAR{education}`,
  `\VAR{skills}`, `\VAR{summary}` (Jinja-style to avoid collision
  with LaTeX `{}`/`\` syntax).
- **Pre-fill placeholder validation** — before any content is injected,
  the template is scanned for placeholders and matched against the set
  of compiled sections. Missing expected placeholders (any of the 4
  section kinds) produce a hard error. Unknown (user-defined)
  `\VAR{...}` placeholders pass through unchanged; they are not errors.
- **LaTeX sanitizer** — a deterministic function escapes characters
  that would break LaTeX compilation: `& % $ # _ { } ~ ^ \`. Applied
  to every section's content before placeholder substitution. This is
  a separate pipeline step (not the Writing Subagent's concern).
- **Compilation**: shell out to `tectonic` if present, else `pdflatex`,
  detected once at startup and cached in session state.
- **Post-compile validation** — the LaTeX log is scanned for errors
  and warnings. If compilation fails, an excerpt of the log (last 30
  lines) is shown on the export screen. Auxiliary files are cleaned up
  on success; retained on failure for debugging.

---

## 3. Explicitly deferred (not v1, don't build for)
- Fuzzy/non-URL-based project dedup matching.
- Version history on `ProjectRecord`.
- Filter Subagent ranking/ordering of projects (include/exclude only).
- Deterministic (non-LLM) formatting linter — ATS critique uses LLM
  judgment only in v1.
- Parallel exploration (sequential only in v1).
- Any live/inline editing model beyond batch generate-then-review.
- Forks / multi-URL dedup (same project cloned from multiple remotes).
- Empty / docs-only repos beyond the `low_confidence` flag — no
  automated skip or special handling; user sees the warning and skips
  manually via the filter screen.
- Private repo / token-based authentication support (public URLs only).
- Data-schema migration tooling for `ProjectRecord`, `SessionState`,
  or `SectionVersion`.
- Internationalization (English-only v1).
- Multi-session concurrency (single active session enforced via lock).

---

## 4. Open item requiring an artifact from you
- **A personalized `.tex` template with your styling.** A minimal
  example with the 4 placeholders (`\VAR{experience}`, `\VAR{education}`,
  `\VAR{skills}`, `\VAR{summary}`) is provided at
  `userprofile/template.example.tex` — this validates the placeholder
  scheme. To produce your actual resume PDF, copy it to
  `userprofile/template.tex` and customize the preamble, fonts, layout,
  and section styling to your preference. Set `latex.template_path`
  in `config.yaml` to point to this file. Until that's done, the
  Export node, the pre-flight template-structure check, and the
  Compile Critique formatting pass will fail.

---

## 5. Verification strategy

These are planning guidelines for test implementation, not executable
specifications. Actual test files live under `tests/`.

### 5.1 Unit-testable layers

| Layer | What to test | Framework suggestion |
|---|---|---|
| LaTeX sanitizer | Escapes `& % $ # _ { } ~ ^ \` correctly; handles edge cases (already-escaped sequences, empty strings, Unicode) | `pytest` parameterized |
| Atomic IO helper | Write-temp-then-rename produces correct file; fails on missing parent dir; survives concurrent writes to different paths | `pytest` with `tmp_path` |
| Placeholder validator | Matching template placeholders against section set; missing placeholder detected; unknown placeholder detected | `pytest` |
| State models | `SectionVersion`, `Issue`, `SessionState` construction, serialization round-trip, `ResumeSection.current` property | `pytest` + `pydantic` |
| Config parser | `config.yaml` load + merge defaults; missing fields produce clear error; bad `api_key_env` detected | `pytest` with fixture YAML files |
| Prompt loader | Load prompt from filesystem; missing path → error; path traversal blocked | `pytest` |
| Pre-flight checks | Valid config passes; missing API key fails; missing template fails; invalid template placeholders fail | `pytest` |
| Cost-cap enforcement | Running total exceeds `per_session_cap_usd` → graph pauses with option to increase/abort; cost=0 models excluded from cap | `pytest` with mock LLMCall |
| Concurrent session lock | Second session start with active lock fails; stale lock cleanup works; resume path skips lock | `pytest` with temp lock files |

### 5.2 Integration-testable flows

| Flow | Approach |
|---|---|
| LangGraph wiring (happy path) | Mock all `ChatModel` instances (return structured outputs directly); verify graph traverses all nodes in §1 order; verify `SessionState` accumulates correctly |
| Content Critique flow | Mock Writing output + JD; feed to Content Critique node; assert `Issue` list contains expected keyword matches |
| Export flow | Place real `.tex` template in `tmp_path`; inject mock section content (with special chars); assert PDF is produced and LaTeX sanitizer was applied |
| Error paths | Mock `ChatModel` to raise on nth call; assert retry fires; assert flag-and-continue on exhausted retries |

### 5.3 Test fixtures

- A small real Git repository (`tests/fixtures/sample-repo/`) with a
  README, a few source files, and a license file — used by Explore
  integration tests.
- A minimal `.tex` template with named placeholders for each of the
  four section kinds — used by Export + pre-flight tests.
- Sample `config.yaml` files (valid, missing API key, bad template
  path) — used by config + pre-flight tests.
- Sample `userinfo.md` files (valid YAML front-matter, missing fields)
  — used by setup-screen form-render tests.

### 5.4 What is not tested automatically

- Visual TUI layout (accept manual inspection via `textual-dev` tools).
- Actual LLM output quality (evaluated manually via the review loop).
- LaTeX compilation correctness beyond exit code (the LaTeX toolchain
  is treated as a trusted external dependency).
