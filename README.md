# Phased Spec Discussion Orchestrator (v2)

Multi-agent spec review using Claude Code and Codex CLI. Runs 4 deterministic phases instead of iterative rounds.

## Architecture

| Phase | Agent(s) | Purpose |
|-------|----------|---------|
| 1. Parallel Critique | Claude + Codex (parallel) | Independent issue identification |
| 2. Consolidation | Codex | Merge, deduplicate, assign ISSUE-NNN IDs |
| 3. Resolution | Claude → Codex (up to 2 rounds) | Resolve issues, revise spec |
| 4. Finalization | Claude (optional, `--finalize`) | Rewrite spec files |

**Total agent invocations**: 5–7 (vs 2–12 with v1 iterative rounds).

## Repository layout

```text
agent_discussion/
  discuss.py
  config.example.json
  templates/
    phase1_codex_critique.md
    phase1_claude_critique.md
    phase2_consolidate.md
    phase3_resolve_claude.md
    phase3_resolve_codex.md
```

## Requirements

- Python 3.10+
- Codex CLI installed and authenticated
- Claude CLI installed and authenticated
- Both CLIs usable non-interactively

## Setup

1. Copy the example config:

```bash
cp agent_discussion/config.example.json agent_discussion/config.json
```

2. Update `agent_discussion/config.json`:

- `discussion.specs_root` — where your spec files live
- `discussion.codebase_paths` — directories agents should inspect
- `discussion.require_codebase_compare` — `true` for existing projects, `false` for greenfield
- `discussion.context_files` — extra files to include in context (e.g. `CLAUDE.md`)
- `discussion.max_resolution_rounds` — max resolution rounds (default: 2)
- `discussion.max_new_issues_per_resolution_round` — cap on new issues per round (default: 3)

## Run

`--spec` accepts a file or a directory. If a directory, all `.md` files are recursively assembled (readme/index first, then alphabetical).

```bash
python3 agent_discussion/discuss.py \
  --config agent_discussion/config.json \
  --project-root /absolute/path/to/target-project \
  --spec path/to/spec-directory
```

With `--finalize`, after consensus the orchestrator backs up the original spec and rewrites it with the agreed version:

```bash
python3 agent_discussion/discuss.py \
  --config agent_discussion/config.json \
  --project-root /absolute/path/to/target-project \
  --spec path/to/spec-directory \
  --finalize
```

`--spec` can be relative to `discussion.specs_root` or absolute.

## Optional flags

- `--run-id` — slug identifying the run (auto-generated from timestamp + spec name if omitted)
- `--max-resolution-rounds` — override from config (default: 2)
- `--specs-root` — override `discussion.specs_root` from config
- `--codebase-path` (repeatable) — override `discussion.codebase_paths` from config
- `--resume` — resume an interrupted run (requires `--run-id`)
- `--finalize` — after consensus, back up original spec and rewrite with the agreed version

## Resuming a run

If a run is interrupted (process killed, agent timeout, etc.), resume it:

```bash
python3 agent_discussion/discuss.py \
  --config agent_discussion/config.json \
  --project-root /absolute/path/to/target-project \
  --spec path/to/spec-directory \
  --run-id the-original-run-id \
  --resume
```

Resume detects the exact phase boundary where the run stopped:

- If one critique exists but not the other, runs only the missing agent
- If both critiques exist but no consolidation, resumes at consolidation
- If consolidation exists, resumes at the first incomplete resolution round
- Mid-round recovery: if Claude resolved but Codex hasn't reviewed, resumes at Codex review

## Environment variable unsetting

Agent env vars in config support `null` to unset inherited variables:

```json
"env": {
  "CLAUDECODE": null
}
```

This is necessary when running Claude Code as a subprocess — it refuses to start inside another Claude session if the `CLAUDECODE` env var is set. Setting it to `null` removes it from the child process environment. The example config already includes this.

## Output artifacts

Each run writes:

```text
<project-root>/.agent-discussions/<run-id>/
  input-spec.md
  working-spec.md
  issue-ledger.json
  context-summary.md
  run-meta.json
  phases/
    01-critique-codex.{prompt,response,stderr}.{md,txt}
    01-critique-claude.{prompt,response,stderr}.{md,txt}
    02-consolidate.{prompt,response,stderr}.{md,txt}
    03-resolve-R01-claude.{prompt,response,stderr}.{md,txt}
    03-resolve-R01-codex.{prompt,response,stderr}.{md,txt}
    03-resolve-R02-claude.{prompt,response,stderr}.{md,txt}   (if round 2)
    03-resolve-R02-codex.{prompt,response,stderr}.{md,txt}
  verdict.md
  claude-action-brief.md
  spec-backup/          (when --finalize is used)
```

## Guidance

- Keep `require_codebase_compare=true` for normal projects.
- Use `require_codebase_compare=false` only for true greenfield/pre-code specs.
- Customize prompts in `agent_discussion/templates/` as needed.
- Phase 1 runs both agents in parallel — total time is max(codex, claude) instead of sum.
- Resolution is capped at 2 rounds by default. Most issues resolve in 1.
