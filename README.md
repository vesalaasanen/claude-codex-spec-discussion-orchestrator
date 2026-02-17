# Claude-Codex Spec Discussion Orchestrator

Reusable automation for running iterative spec discussions between:

- **Codex CLI** (critic)
- **Claude Code CLI** (author/reviser)

The loop is fully automated and writes durable run artifacts so it can be imported and reused across many projects.

## What it does

For each round:

1. Codex critiques the current working spec.
2. Claude replies issue-by-issue and returns a full revised spec.
3. The orchestrator updates issue ledger state and stop conditions.

Default convergence rule: stop when there are no unresolved medium/high/critical issues.

## Repository layout

```text
agent_discussion/
  discuss.py
  config.example.json
  templates/
    codex_prompt.md
    claude_prompt.md
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

2. Update `agent_discussion/config.json` for your environment:

- `discussion.specs_root`
- `discussion.codebase_paths`
- `discussion.require_codebase_compare`
- `agents.codex.command`
- `agents.claude.command`

Command placeholder variables supported in `agents.*.command`:

- `{prompt_file}`
- `{output_file}`
- `{project_root}`
- `{run_dir}`
- `{round_number}`

## Run

```bash
python3 agent_discussion/discuss.py \
  --config agent_discussion/config.json \
  --project-root /absolute/path/to/target-project \
  --spec your-spec.md
```

`--spec` can be relative to `discussion.specs_root` or absolute.

## Optional flags

- `--run-id`
- `--max-rounds`
- `--specs-root`
- `--codebase-path` (repeatable)

## Output artifacts

Each run writes:

```text
<project-root>/.agent-discussions/<run-id>/
  input-spec.md
  working-spec.md
  issue-ledger.json
  context-summary.md
  rounds/
    01-codex.prompt.md
    01-codex.response.md
    01-codex.stderr.txt
    01-claude.prompt.md
    01-claude.response.md
    01-claude.stderr.txt
  verdict.md
  claude-action-brief.md
  run-meta.json
```

## Guidance

- Keep `require_codebase_compare=true` for normal projects.
- Use `require_codebase_compare=false` only for true greenfield/pre-code specs.
- Customize prompts in `agent_discussion/templates/` as needed.
