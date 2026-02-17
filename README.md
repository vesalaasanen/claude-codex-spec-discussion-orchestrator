# Claude-Codex Spec Discussion Orchestrator

A reusable, project-agnostic automation system where **Codex CLI** and **Claude Code CLI** iteratively review and refine a spec until convergence.

It runs fully unattended, preserves context across rounds, and produces audit-friendly artifacts (`issue-ledger`, round prompts/responses, final verdict, and an implementation brief).

## Why this exists

Manual back-and-forth between agents tends to lose context and creates inconsistent decisions. This orchestrator standardizes the loop:

1. Codex critiques the latest spec draft.
2. Claude responds issue-by-issue and returns a full revised spec.
3. The orchestrator reconciles issue state and decides whether to stop.

## Features

- Fully automated multi-round discussion loop
- Stable issue IDs and durable issue ledger
- Convergence stop conditions based on issue severity/status
- Codebase-aware mode (default) for spec-vs-implementation validation
- Greenfield mode for pre-code projects
- Plain Python stdlib only (no third-party dependencies)

## Repository layout

```text
agent_discussion/
  discuss.py
  config.example.json
  config.greenfield-test.json
  templates/
    codex_prompt.md
    claude_prompt.md
  test_agents/
    codex_greenfield_stub.py
    claude_greenfield_stub.py
specs/
  greenfield/
    threejs-flight-simulator-product-spec.md
    threejs-flight-simulator-technical-spec.md
    threejs-flight-simulator-content-spec.md
```

## Requirements

- Python 3.10+
- `codex` CLI installed + authenticated
- `claude` CLI installed + authenticated
- Both CLIs must support non-interactive prompt execution

## Quick start (normal codebase-aware mode)

1. Copy config template:

```bash
cp agent_discussion/config.example.json agent_discussion/config.json
```

2. Edit `agent_discussion/config.json`:

- `agents.codex.command`
- `agents.claude.command`
- `discussion.specs_root`
- `discussion.codebase_paths`
- keep `discussion.require_codebase_compare=true` for normal projects

3. Run:

```bash
python3 agent_discussion/discuss.py \
  --config agent_discussion/config.json \
  --project-root /absolute/path/to/your/project \
  --spec your-spec.md
```

`--spec` may be relative to `discussion.specs_root` or absolute.

## Greenfield fixture (deterministic local test)

This repo includes a no-codebase fixture for end-to-end validation:

- Config: `agent_discussion/config.greenfield-test.json`
- Stub agents in `agent_discussion/test_agents/`
- Sample specs in `specs/greenfield/`

Run fixture:

```bash
python3 agent_discussion/discuss.py \
  --config agent_discussion/config.greenfield-test.json \
  --project-root /absolute/path/to/this/repo \
  --spec threejs-flight-simulator-product-spec.md \
  --run-id greenfield-e2e-flight-sim
```

## Output artifacts

Each run writes to:

```text
<project-root>/.agent-discussions/<run-id>/
  input-spec.md
  working-spec.md
  issue-ledger.json
  context-summary.md
  rounds/
  verdict.md
  claude-action-brief.md
  run-meta.json
```

## CLI flags

- `--config` (required)
- `--project-root` (required for cross-repo usage)
- `--spec` (required)
- `--specs-root` (optional override)
- `--codebase-path` (repeatable override)
- `--run-id` (optional)
- `--max-rounds` (optional)

## Notes

- For most projects, keep codebase comparison enabled.
- Use greenfield mode (`require_codebase_compare=false`) only when codebase does not exist yet.
- Prompts are customizable via files under `agent_discussion/templates/`.
