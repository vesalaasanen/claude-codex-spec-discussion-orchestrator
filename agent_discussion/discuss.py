#!/usr/bin/env python3
"""Fully automated Claude <-> Codex spec discussion orchestrator.

This script is intentionally dependency-free (stdlib only) so it can be reused
in any repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CODEX_TEMPLATE = """You are Codex CLI acting as a strict spec critic in an automated multi-agent loop.

OBJECTIVE
- Critique the spec with focus on correctness, feasibility, risk, testability, and clarity.
- Re-evaluate existing issues before creating new ones.
- Compare spec requirements against the existing implementation within the Codebase Scope from CONTEXT_PACKET.

RULES
- Use existing issue IDs from ISSUE_LEDGER_JSON for updates.
- Add new issues only when they are genuinely new.
- Use severity: low | medium | high | critical.
- Use status: open | agreed | rejected-with-rationale | needs-evidence | escalated.
- For each major claim, include evidence from code paths listed in Codebase Scope (or state that evidence is missing).
- IMPORTANT: Respond with exactly ONE JSON code block and NOTHING else.

JSON SCHEMA
{
  "issue_updates": [
    {
      "id": "ISSUE-001",
      "status": "open",
      "severity": "high",
      "notes": "why status changed"
    }
  ],
  "new_issues": [
    {
      "title": "...",
      "severity": "medium",
      "rationale": "...",
      "proposed_change": "..."
    }
  ],
  "spec_edit_suggestions": [
    {
      "section": "...",
      "replace": "...",
      "with": "..."
    }
  ],
  "codebase_evidence": [
    {
      "path": "src/module/file.ext",
      "finding": "What exists or is missing relative to the spec"
    }
  ],
  "round_digest": "short summary for future context packets"
}

ROUND
{{ROUND_NUMBER}} / {{MAX_ROUNDS}}

CONTEXT_PACKET
{{CONTEXT_PACKET}}

ISSUE_LEDGER_JSON
{{ISSUE_LEDGER_JSON}}

WORKING_SPEC_MARKDOWN
{{WORKING_SPEC}}
"""

DEFAULT_CLAUDE_TEMPLATE = """You are Claude Code CLI acting as the spec author/reviser in an automated multi-agent loop.

OBJECTIVE
- Respond to Codex critique issue-by-issue.
- Produce a revised full spec.
- Validate Codex claims against the codebase paths provided in CONTEXT_PACKET when codebase comparison is required.

RULES
- For each referenced issue ID, provide a decision and a resulting status.
- Decision must be one of: accept | partial | reject.
- Resulting status must be one of: agreed | rejected-with-rationale | needs-evidence | open.
- Keep changes high signal and consistent with project intent.
- Include codebase evidence paths for significant accept/reject decisions.
- IMPORTANT: Respond with exactly TWO blocks and NOTHING else:
  1) one JSON code block (metadata)
  2) one markdown code block containing the FULL updated spec

JSON SCHEMA
{
  "issue_responses": [
    {
      "id": "ISSUE-001",
      "decision": "accept",
      "new_status": "agreed",
      "reason": "..."
    }
  ],
  "applied_changes_summary": [
    "..."
  ],
  "codebase_evidence": [
    {
      "path": "src/module/file.ext",
      "finding": "How this file supports or contradicts Codex critique"
    }
  ],
  "round_digest": "short summary for future context packets"
}

ROUND
{{ROUND_NUMBER}} / {{MAX_ROUNDS}}

CONTEXT_PACKET
{{CONTEXT_PACKET}}

ISSUE_LEDGER_JSON
{{ISSUE_LEDGER_JSON}}

LATEST_CODEX_CRITIQUE
{{CODEX_RESPONSE}}

CURRENT_WORKING_SPEC_MARKDOWN
{{WORKING_SPEC}}
"""

VALID_STATUSES = {
    "open",
    "agreed",
    "rejected-with-rationale",
    "needs-evidence",
    "escalated",
}
OPEN_STATUSES = {"open", "needs-evidence"}
VALID_SEVERITIES = {"low", "medium", "high", "critical"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def slugify(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    clean = re.sub(r"-+", "-", clean).strip("-")
    return clean or "run"


def find_code_block(content: str, languages: list[str]) -> str | None:
    for language in languages:
        pattern = rf"```{language}\s*(.*?)\s*```"
        match = re.search(pattern, content, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_json_block(content: str) -> dict[str, Any]:
    raw = find_code_block(content, ["json"])
    if raw is None:
        raise ValueError("Expected a JSON code block, but none was found.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc


def extract_updated_spec(content: str) -> str:
    # Prefer a markdown-labelled block. Fall back to first non-json code block.
    markdown = find_code_block(content, ["markdown", "md"])
    if markdown is not None:
        return markdown

    for match in re.finditer(r"```([a-zA-Z0-9_-]*)\s*(.*?)\s*```", content, flags=re.DOTALL):
        lang = (match.group(1) or "").lower().strip()
        body = match.group(2).strip()
        if lang != "json":
            return body

    raise ValueError("Expected a markdown code block containing the updated spec.")


def normalize_status(value: str, *, default: str = "open") -> str:
    if not isinstance(value, str):
        return default
    cleaned = value.strip().lower()
    return cleaned if cleaned in VALID_STATUSES else default


def normalize_severity(value: str, *, default: str = "medium") -> str:
    if not isinstance(value, str):
        return default
    cleaned = value.strip().lower()
    return cleaned if cleaned in VALID_SEVERITIES else default


def resolve_template(template_path: str | None, fallback: str, base_dir: Path) -> str:
    if not template_path:
        return fallback
    path = Path(template_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists():
        return fallback
    return read_text(path)


def render_template(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def substitute_placeholders(raw: str, replacements: dict[str, str]) -> str:
    rendered = raw
    for key, value in replacements.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


class OrchestrationError(RuntimeError):
    pass


class DiscussionOrchestrator:
    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path,
        project_root: Path,
        spec_path: Path,
        specs_root_override: Path | None,
        codebase_paths_override: list[str] | None,
        run_id: str | None,
        max_rounds_override: int | None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.config_dir = config_path.parent
        self.project_root = project_root.resolve()
        self.specs_root = self._resolve_specs_root(specs_root_override)
        self.spec_path = self._resolve_spec_path(spec_path)
        self.max_rounds = max_rounds_override or int(config.get("discussion", {}).get("max_rounds", 6))
        if self.max_rounds < 1:
            raise OrchestrationError("max_rounds must be >= 1")

        self.spec_glob = str(config.get("discussion", {}).get("spec_glob", "*.md"))
        self.specs_index_max_files = int(config.get("discussion", {}).get("specs_index_max_files", 200))
        self.require_codebase_compare = bool(
            config.get("discussion", {}).get("require_codebase_compare", True)
        )
        configured_codebase_paths = config.get("discussion", {}).get("codebase_paths", ["."])
        selected_codebase_paths = (
            codebase_paths_override if codebase_paths_override is not None else configured_codebase_paths
        )
        self.codebase_paths = self._resolve_codebase_paths(selected_codebase_paths)

        output_root = config.get("output_root", ".agent-discussions")
        output_root_path = Path(output_root)
        if not output_root_path.is_absolute():
            output_root_path = self.project_root / output_root_path
        self.output_root = output_root_path.resolve()

        inferred_run_id = run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slugify(self.spec_path.stem)}"
        self.run_id = slugify(inferred_run_id)
        self.run_dir = self.output_root / self.run_id
        self.rounds_dir = self.run_dir / "rounds"
        self.input_spec_file = self.run_dir / "input-spec.md"
        self.working_spec_file = self.run_dir / "working-spec.md"
        self.ledger_file = self.run_dir / "issue-ledger.json"
        self.context_summary_file = self.run_dir / "context-summary.md"
        self.verdict_file = self.run_dir / "verdict.md"
        self.action_brief_file = self.run_dir / "claude-action-brief.md"
        self.run_meta_file = self.run_dir / "run-meta.json"

        self.codex_template = resolve_template(
            config.get("templates", {}).get("codex"),
            DEFAULT_CODEX_TEMPLATE,
            self.config_dir,
        )
        self.claude_template = resolve_template(
            config.get("templates", {}).get("claude"),
            DEFAULT_CLAUDE_TEMPLATE,
            self.config_dir,
        )

        self.high_severity_labels = {
            normalize_severity(s) for s in config.get("discussion", {}).get("high_severity_labels", ["high", "critical"])
        }
        if not self.high_severity_labels:
            self.high_severity_labels = {"high", "critical"}

        self._validate_agents_config()

    def _validate_agents_config(self) -> None:
        agents = self.config.get("agents", {})
        for name in ("codex", "claude"):
            if name not in agents:
                raise OrchestrationError(f"Missing config.agents.{name}")
            if "command" not in agents[name]:
                raise OrchestrationError(f"Missing config.agents.{name}.command")

    def _resolve_specs_root(self, override: Path | None) -> Path:
        if override is not None:
            root = override
        else:
            configured = self.config.get("discussion", {}).get("specs_root", ".")
            root = Path(configured)

        if not root.is_absolute():
            root = (self.project_root / root).resolve()
        return root

    def _resolve_spec_path(self, raw_spec_path: Path) -> Path:
        if raw_spec_path.is_absolute():
            return raw_spec_path.resolve()

        from_specs_root = (self.specs_root / raw_spec_path).resolve()
        if from_specs_root.exists():
            return from_specs_root

        from_project_root = (self.project_root / raw_spec_path).resolve()
        if from_project_root.exists():
            return from_project_root

        return from_specs_root

    def _resolve_codebase_paths(self, raw_paths: list[str] | None) -> list[Path]:
        if raw_paths is None:
            raw_paths = ["."]
        if len(raw_paths) == 0:
            return []

        resolved: list[Path] = []
        for raw in raw_paths:
            path = Path(raw)
            if not path.is_absolute():
                path = (self.project_root / path).resolve()
            else:
                path = path.resolve()
            resolved.append(path)

        return resolved

    def _relative_display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)

    def _build_specs_index(self) -> str:
        if not self.specs_root.exists() or not self.specs_root.is_dir():
            return f"- Specs root missing: {self.specs_root}"

        matches = sorted(p for p in self.specs_root.rglob(self.spec_glob) if p.is_file())
        if not matches:
            return f"- No files matching '{self.spec_glob}' under {self.specs_root}"

        limited = matches[: self.specs_index_max_files]
        lines = [f"- {self._relative_display_path(path)}" for path in limited]
        if len(matches) > len(limited):
            lines.append(f"- ... ({len(matches) - len(limited)} more files omitted)")
        return "\n".join(lines)

    def _build_codebase_scope(self) -> str:
        if not self.codebase_paths:
            if self.require_codebase_compare:
                return "- No codebase paths configured (this is invalid when require_codebase_compare=true)."
            return "- No codebase paths configured. Greenfield mode enabled."

        lines = [f"- {self._relative_display_path(path)}" for path in self.codebase_paths]
        if self.require_codebase_compare:
            lines.append(
                "- Agents must inspect relevant files in these paths before concluding whether spec requirements are implemented."
            )
        else:
            lines.append(
                "- Codebase comparison is optional for this run (greenfield mode)."
            )
        return "\n".join(lines)

    def bootstrap(self) -> None:
        if not self.spec_path.exists():
            raise OrchestrationError(f"Spec file not found: {self.spec_path}")
        if not self.specs_root.exists() or not self.specs_root.is_dir():
            raise OrchestrationError(f"Specs root not found or not a directory: {self.specs_root}")

        if self.require_codebase_compare and not self.codebase_paths:
            raise OrchestrationError(
                "No codebase_paths configured but discussion.require_codebase_compare=true. "
                "Set discussion.codebase_paths or disable require_codebase_compare for greenfield specs."
            )

        missing_codebase = [path for path in self.codebase_paths if not path.exists()]
        if missing_codebase:
            joined = ", ".join(str(path) for path in missing_codebase)
            raise OrchestrationError(f"Configured codebase path(s) not found: {joined}")

        if self.run_dir.exists():
            raise OrchestrationError(
                f"Run directory already exists: {self.run_dir}. Use a different --run-id."
            )

        self.rounds_dir.mkdir(parents=True, exist_ok=False)
        spec_content = read_text(self.spec_path)
        write_text(self.input_spec_file, spec_content)
        write_text(self.working_spec_file, spec_content)

        initial_ledger = {
            "meta": {
                "created_at": utc_now(),
                "next_issue_number": 1,
            },
            "issues": [],
            "rounds": [],
        }
        save_json(self.ledger_file, initial_ledger)
        write_text(self.context_summary_file, "# Context Summary\n\n")

        run_meta = {
            "created_at": utc_now(),
            "project_root": str(self.project_root),
            "specs_root": str(self.specs_root),
            "spec_source": str(self.spec_path),
            "codebase_paths": [str(path) for path in self.codebase_paths],
            "require_codebase_compare": self.require_codebase_compare,
            "run_id": self.run_id,
            "max_rounds": self.max_rounds,
            "config_path": str(self.config_path),
        }
        save_json(self.run_meta_file, run_meta)

    def load_ledger(self) -> dict[str, Any]:
        return load_json(self.ledger_file)

    def save_ledger(self, ledger: dict[str, Any]) -> None:
        save_json(self.ledger_file, ledger)

    def next_issue_id(self, ledger: dict[str, Any]) -> str:
        n = int(ledger["meta"].get("next_issue_number", 1))
        ledger["meta"]["next_issue_number"] = n + 1
        return f"ISSUE-{n:03d}"

    def build_context_packet(self, round_number: int) -> str:
        ledger = self.load_ledger()
        open_issues = [
            issue for issue in ledger["issues"] if normalize_status(issue.get("status", "open")) in OPEN_STATUSES
        ]
        open_issues_md = "\n".join(
            f"- {i['id']} [{i.get('severity', 'medium')}] {i.get('title', '')} (status: {i.get('status', 'open')})"
            for i in open_issues
        )
        if not open_issues_md:
            open_issues_md = "- None"

        summary = read_text(self.context_summary_file).strip()
        if not summary:
            summary = "# Context Summary\n\n- No prior rounds."

        extra_context = self._read_extra_context_files()
        specs_index = self._build_specs_index()
        codebase_scope = self._build_codebase_scope()

        return (
            f"Round: {round_number}/{self.max_rounds}\n"
            f"Project Root: {self.project_root}\n\n"
            f"## Spec Location\n"
            f"- Active Spec: {self.spec_path}\n"
            f"- Specs Root: {self.specs_root}\n\n"
            f"## Specs Folder Index\n{specs_index}\n\n"
            f"## Codebase Scope\n{codebase_scope}\n\n"
            f"## Open Issues\n{open_issues_md}\n\n"
            f"## Historical Summary\n{summary}\n\n"
            f"## Additional Context\n{extra_context}\n"
        )

    def _read_extra_context_files(self) -> str:
        paths = self.config.get("discussion", {}).get("context_files", [])
        if not paths:
            return "- None"
        chunks: list[str] = []
        for raw in paths:
            path = Path(raw)
            if not path.is_absolute():
                path = (self.project_root / path).resolve()
            if not path.exists() or not path.is_file():
                chunks.append(f"- Missing: {path}")
                continue
            chunks.append(f"### {path}\n")
            chunks.append(read_text(path))
            chunks.append("\n")
        return "\n".join(chunks).strip() or "- None"

    def run(self) -> None:
        self.bootstrap()
        stop_reason = "Reached max rounds"

        for round_number in range(1, self.max_rounds + 1):
            codex_payload, codex_raw = self._run_codex_round(round_number)
            ledger = self.load_ledger()
            self._apply_codex_payload(ledger, codex_payload, round_number)
            self.save_ledger(ledger)

            claude_payload, claude_spec, _ = self._run_claude_round(
                round_number,
                codex_raw,
            )

            ledger = self.load_ledger()
            self._apply_claude_payload(ledger, claude_payload, round_number)
            ledger["rounds"].append(
                {
                    "round": round_number,
                    "at": utc_now(),
                    "codex_digest": codex_payload.get("round_digest", ""),
                    "claude_digest": claude_payload.get("round_digest", ""),
                }
            )
            self.save_ledger(ledger)
            write_text(self.working_spec_file, claude_spec)
            self._append_round_summary(round_number, codex_payload, claude_payload)

            should_stop, reason = self._should_stop(ledger)
            if should_stop:
                stop_reason = reason
                break

        self._generate_verdict(stop_reason)
        self._generate_action_brief(stop_reason)

    def _run_codex_round(self, round_number: int) -> tuple[dict[str, Any], str]:
        context_packet = self.build_context_packet(round_number)
        prompt = render_template(
            self.codex_template,
            {
                "ROUND_NUMBER": str(round_number),
                "MAX_ROUNDS": str(self.max_rounds),
                "CONTEXT_PACKET": context_packet,
                "ISSUE_LEDGER_JSON": json.dumps(self.load_ledger(), indent=2, ensure_ascii=False),
                "WORKING_SPEC": read_text(self.working_spec_file),
            },
        )

        prompt_path = self.rounds_dir / f"{round_number:02d}-codex.prompt.md"
        response_path = self.rounds_dir / f"{round_number:02d}-codex.response.md"
        write_text(prompt_path, prompt)

        raw = self._invoke_agent("codex", prompt, prompt_path, response_path, round_number)
        payload = extract_json_block(raw)
        return payload, raw

    def _run_claude_round(
        self,
        round_number: int,
        codex_response_raw: str,
    ) -> tuple[dict[str, Any], str, str]:
        context_packet = self.build_context_packet(round_number)
        prompt = render_template(
            self.claude_template,
            {
                "ROUND_NUMBER": str(round_number),
                "MAX_ROUNDS": str(self.max_rounds),
                "CONTEXT_PACKET": context_packet,
                "ISSUE_LEDGER_JSON": json.dumps(self.load_ledger(), indent=2, ensure_ascii=False),
                "WORKING_SPEC": read_text(self.working_spec_file),
                "CODEX_RESPONSE": codex_response_raw,
            },
        )

        prompt_path = self.rounds_dir / f"{round_number:02d}-claude.prompt.md"
        response_path = self.rounds_dir / f"{round_number:02d}-claude.response.md"
        write_text(prompt_path, prompt)

        raw = self._invoke_agent("claude", prompt, prompt_path, response_path, round_number)
        payload = extract_json_block(raw)
        updated_spec = extract_updated_spec(raw)
        return payload, updated_spec, raw

    def _invoke_agent(
        self,
        agent_name: str,
        prompt_text: str,
        prompt_path: Path,
        response_path: Path,
        round_number: int,
    ) -> str:
        agent_cfg = self.config["agents"][agent_name]
        raw_command = agent_cfg["command"]
        timeout = int(agent_cfg.get("timeout_seconds", 900))
        pass_stdin = bool(agent_cfg.get("pass_prompt_via_stdin", False))

        scratch_output = self.rounds_dir / f"{round_number:02d}-{agent_name}.raw-output.txt"
        scratch_stderr = self.rounds_dir / f"{round_number:02d}-{agent_name}.stderr.txt"

        variables = {
            "prompt_file": str(prompt_path),
            "output_file": str(scratch_output),
            "project_root": str(self.project_root),
            "run_dir": str(self.run_dir),
            "round_number": str(round_number),
        }

        env = os.environ.copy()
        for key, value in agent_cfg.get("env", {}).items():
            env[str(key)] = str(value)

        if isinstance(raw_command, list):
            cmd = [substitute_placeholders(str(part), variables) for part in raw_command]
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                input=prompt_text if pass_stdin else None,
                text=True,
                capture_output=True,
                timeout=timeout,
                env=env,
                shell=False,
            )
        elif isinstance(raw_command, str):
            quoted = {k: shlex.quote(v) for k, v in variables.items()}
            cmd = substitute_placeholders(raw_command, quoted)
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                input=prompt_text if pass_stdin else None,
                text=True,
                capture_output=True,
                timeout=timeout,
                env=env,
                shell=True,
            )
        else:
            raise OrchestrationError(f"agents.{agent_name}.command must be a list or string")

        write_text(scratch_stderr, result.stderr or "")
        stdout = (result.stdout or "").strip()
        file_output = read_text(scratch_output).strip() if scratch_output.exists() else ""
        combined = stdout if stdout else file_output

        if result.returncode != 0:
            raise OrchestrationError(
                f"Agent '{agent_name}' failed with code {result.returncode}. See {scratch_stderr}"
            )
        if not combined:
            raise OrchestrationError(
                f"Agent '{agent_name}' returned empty output. Check command and flags in config."
            )

        write_text(response_path, combined + "\n")
        return combined

    def _apply_codex_payload(self, ledger: dict[str, Any], payload: dict[str, Any], round_number: int) -> None:
        issues_by_id = {issue["id"]: issue for issue in ledger["issues"]}

        for update in payload.get("issue_updates", []):
            issue_id = str(update.get("id", "")).strip()
            if not issue_id:
                continue
            issue = issues_by_id.get(issue_id)
            if not issue:
                issue = {
                    "id": issue_id,
                    "title": "Referenced by Codex update",
                    "severity": normalize_severity(update.get("severity", "medium")),
                    "source": "codex",
                    "status": normalize_status(update.get("status", "open")),
                    "rationale": "",
                    "codex_proposal": "",
                    "claude_response": "",
                    "resolution_notes": "",
                    "history": [],
                }
                ledger["issues"].append(issue)
                issues_by_id[issue_id] = issue

            issue["status"] = normalize_status(update.get("status", issue.get("status", "open")))
            issue["severity"] = normalize_severity(update.get("severity", issue.get("severity", "medium")))
            note = str(update.get("notes", "")).strip()
            if note:
                issue["resolution_notes"] = note
            issue.setdefault("history", []).append({"round": round_number, "by": "codex", "event": "update", "payload": update})

        for new_issue in payload.get("new_issues", []):
            issue_id = self.next_issue_id(ledger)
            issue = {
                "id": issue_id,
                "title": str(new_issue.get("title", "Untitled issue")).strip(),
                "severity": normalize_severity(new_issue.get("severity", "medium")),
                "source": "codex",
                "status": "open",
                "rationale": str(new_issue.get("rationale", "")).strip(),
                "codex_proposal": str(new_issue.get("proposed_change", "")).strip(),
                "claude_response": "",
                "resolution_notes": "",
                "history": [{"round": round_number, "by": "codex", "event": "new", "payload": new_issue}],
            }
            ledger["issues"].append(issue)

    def _apply_claude_payload(self, ledger: dict[str, Any], payload: dict[str, Any], round_number: int) -> None:
        issues_by_id = {issue["id"]: issue for issue in ledger["issues"]}
        for response in payload.get("issue_responses", []):
            issue_id = str(response.get("id", "")).strip()
            if not issue_id or issue_id not in issues_by_id:
                continue
            issue = issues_by_id[issue_id]

            decision = str(response.get("decision", "")).strip().lower()
            mapped_status = {
                "accept": "agreed",
                "partial": "needs-evidence",
                "reject": "rejected-with-rationale",
            }.get(decision, issue.get("status", "open"))

            issue["status"] = normalize_status(response.get("new_status", mapped_status), default=mapped_status)
            issue["claude_response"] = str(response.get("reason", "")).strip()
            issue.setdefault("history", []).append(
                {"round": round_number, "by": "claude", "event": "response", "payload": response}
            )

    def _append_round_summary(
        self,
        round_number: int,
        codex_payload: dict[str, Any],
        claude_payload: dict[str, Any],
    ) -> None:
        summary = (
            f"## Round {round_number:02d}\n"
            f"- Codex digest: {str(codex_payload.get('round_digest', '')).strip()}\n"
            f"- Claude digest: {str(claude_payload.get('round_digest', '')).strip()}\n\n"
        )
        current = read_text(self.context_summary_file)
        write_text(self.context_summary_file, current + summary)

    def _should_stop(self, ledger: dict[str, Any]) -> tuple[bool, str]:
        unresolved = [issue for issue in ledger["issues"] if normalize_status(issue.get("status", "open")) in OPEN_STATUSES]
        unresolved_high = [i for i in unresolved if normalize_severity(i.get("severity", "medium")) in self.high_severity_labels]
        unresolved_medium = [i for i in unresolved if normalize_severity(i.get("severity", "medium")) == "medium"]

        if not unresolved_high and not unresolved_medium:
            return True, "All high/medium issues are resolved or explicitly rejected."
        return False, "Continue"

    def _generate_verdict(self, stop_reason: str) -> None:
        ledger = self.load_ledger()
        issues = ledger["issues"]

        agreed = [i for i in issues if i.get("status") == "agreed"]
        rejected = [i for i in issues if i.get("status") == "rejected-with-rationale"]
        unresolved = [
            i for i in issues if i.get("status") in {"open", "needs-evidence", "escalated"}
        ]

        def fmt(items: list[dict[str, Any]]) -> str:
            if not items:
                return "- None"
            return "\n".join(
                f"- {i['id']} [{i.get('severity', 'medium')}] {i.get('title', '')}"
                for i in items
            )

        content = (
            "# Verdict\n\n"
            f"- Generated at: {utc_now()}\n"
            f"- Stop reason: {stop_reason}\n"
            f"- Run directory: `{self.run_dir}`\n\n"
            "## Accepted Issues (agreed)\n"
            f"{fmt(agreed)}\n\n"
            "## Rejected Issues\n"
            f"{fmt(rejected)}\n\n"
            "## Unresolved / Escalated\n"
            f"{fmt(unresolved)}\n\n"
            "## Final Spec\n"
            f"- Use `working-spec.md` at `{self.working_spec_file}` as the latest draft.\n"
        )
        write_text(self.verdict_file, content)

    def _generate_action_brief(self, stop_reason: str) -> None:
        ledger = self.load_ledger()
        agreed = [i for i in ledger["issues"] if i.get("status") == "agreed"]
        unresolved = [i for i in ledger["issues"] if i.get("status") in {"open", "needs-evidence", "escalated"}]

        agreed_lines = "\n".join(
            f"- {i['id']}: {i.get('title', '')} -> {i.get('codex_proposal', '')}" for i in agreed
        ) or "- None"
        unresolved_lines = "\n".join(
            f"- {i['id']} [{i.get('status')}]: {i.get('title', '')}" for i in unresolved
        ) or "- None"

        content = (
            "# Claude Action Brief\n\n"
            f"Stop reason: {stop_reason}\n\n"
            "## Apply these accepted changes\n"
            f"{agreed_lines}\n\n"
            "## Unresolved items (human decision required)\n"
            f"{unresolved_lines}\n\n"
            "## Canonical files\n"
            f"- Final working draft: `{self.working_spec_file}`\n"
            f"- Issue ledger: `{self.ledger_file}`\n"
            f"- Verdict: `{self.verdict_file}`\n"
        )
        write_text(self.action_brief_file, content)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fully automated Claude<->Codex spec discussion orchestrator"
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to initial spec markdown file. Relative paths resolve against --specs-root (or config discussion.specs_root).",
    )
    parser.add_argument(
        "--specs-root",
        default=None,
        help="Directory containing spec files. Overrides config discussion.specs_root.",
    )
    parser.add_argument(
        "--codebase-path",
        action="append",
        default=None,
        help="Repeatable. Limit codebase inspection scope for this run (overrides discussion.codebase_paths).",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root where output_root is resolved (default: current directory)",
    )
    parser.add_argument("--run-id", default=None, help="Optional run ID slug")
    parser.add_argument("--max-rounds", type=int, default=None, help="Override max rounds")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    try:
        config = load_json(config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load config JSON: {exc}", file=sys.stderr)
        return 1

    orchestrator = DiscussionOrchestrator(
        config=config,
        config_path=config_path,
        project_root=Path(args.project_root),
        spec_path=Path(args.spec),
        specs_root_override=Path(args.specs_root) if args.specs_root else None,
        codebase_paths_override=args.codebase_path,
        run_id=args.run_id,
        max_rounds_override=args.max_rounds,
    )

    try:
        orchestrator.run()
    except (OrchestrationError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print("Run completed successfully.")
    print(f"Run directory: {orchestrator.run_dir}")
    print(f"Verdict: {orchestrator.verdict_file}")
    print(f"Action brief: {orchestrator.action_brief_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
