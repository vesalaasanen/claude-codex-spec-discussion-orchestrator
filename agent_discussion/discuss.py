#!/usr/bin/env python3
"""Phased spec discussion orchestrator (v2).

Architecture: 4 deterministic phases instead of iterative rounds.
  Phase 1: Parallel Critique — both agents independently identify issues
  Phase 2: Consolidation — Codex merges and deduplicates issues
  Phase 3: Resolution — Claude resolves issues, Codex reviews (max 2 rounds)
  Phase 4: Finalization — (optional) rewrite spec files

This script is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_STATUSES = {
    "open",
    "agreed",
    "rejected-with-rationale",
    "needs-evidence",
    "escalated",
}
OPEN_STATUSES = {"open", "needs-evidence", "escalated"}
VALID_SEVERITIES = {"low", "medium", "high", "critical"}

_SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG: dict[str, Any] = {
    "output_root": ".agent-discussions",
    "discussion": {
        "specs_root": ".",
        "spec_glob": "*.md",
        "specs_index_max_files": 200,
        "codebase_paths": ["."],
        "require_codebase_compare": True,
        "max_resolution_rounds": 2,
        "max_new_issues_per_resolution_round": 3,
        "high_severity_labels": ["high", "critical"],
        "context_files": [],
    },
    "templates": {
        "phase1_codex_critique": "templates/phase1_codex_critique.md",
        "phase1_claude_critique": "templates/phase1_claude_critique.md",
        "phase2_consolidate": "templates/phase2_consolidate.md",
        "phase3_resolve_claude": "templates/phase3_resolve_claude.md",
        "phase3_resolve_codex": "templates/phase3_resolve_codex.md",
    },
    "agents": {
        "codex": {
            "command": ["codex", "exec", "--full-auto", "-"],
            "timeout_seconds": 900,
            "pass_prompt_via_stdin": True,
            "env": {},
        },
        "claude": {
            "command": [
                "claude", "-p", "--allowedTools",
                "Read,Glob,Grep,Bash,WebSearch,WebFetch",
            ],
            "timeout_seconds": 900,
            "pass_prompt_via_stdin": True,
            "env": {"CLAUDECODE": None},
        },
    },
}


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


def _extract_outermost_fenced_block(content: str, languages: list[str]) -> str | None:
    """Extract the outermost fenced code block matching *languages*, handling nested fences.

    Tracks fence depth so that inner ``` blocks (e.g. code examples inside a
    spec) do not prematurely close the outer block.
    """
    lang_alt = "|".join(re.escape(lang) for lang in languages)
    open_re = re.compile(rf"^(`{{3,}})({lang_alt})\b.*$", re.IGNORECASE | re.MULTILINE)

    for open_match in open_re.finditer(content):
        opener_ticks = open_match.group(1)
        min_close_len = len(opener_ticks)
        body_start = open_match.end() + 1  # skip the newline after the opening fence
        depth = 1
        pos = body_start

        for line_match in re.finditer(r"^(`{3,})(\S*).*$", content[pos:], re.MULTILINE):
            ticks = line_match.group(1)
            info = line_match.group(2).strip()

            if info:
                # Opening fence (has info string) — increase depth
                depth += 1
            elif len(ticks) >= min_close_len:
                depth -= 1
                if depth == 0:
                    body_end = pos + line_match.start()
                    return content[body_start:body_end].strip()

        # If we never closed, the block runs to EOF — still return it
        return content[body_start:].strip()

    return None


def find_code_block(content: str, languages: list[str]) -> str | None:
    """Extract a code block for *languages*, using depth-aware parsing."""
    return _extract_outermost_fenced_block(content, languages)


def extract_json_block(content: str) -> dict[str, Any]:
    raw = find_code_block(content, ["json"])
    if raw is None:
        raise ValueError("Expected a JSON code block, but none was found.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc


def _extract_sentinel_spec(content: str) -> str | None:
    """Extract spec between ===SPEC_START=== and ===SPEC_END=== sentinels."""
    start_tag = "===SPEC_START==="
    end_tag = "===SPEC_END==="
    start_idx = content.find(start_tag)
    if start_idx == -1:
        return None
    start_idx += len(start_tag)
    end_idx = content.find(end_tag, start_idx)
    if end_idx == -1:
        return None
    return content[start_idx:end_idx].strip()


def extract_updated_spec(content: str) -> str:
    # 1. Prefer sentinel delimiters (immune to nested fences)
    sentinel = _extract_sentinel_spec(content)
    if sentinel is not None:
        return sentinel

    # 2. Depth-aware fenced block extraction
    markdown = find_code_block(content, ["markdown", "md"])
    if markdown is not None:
        return markdown

    # 3. Fallback: first non-json outermost fenced block
    for open_match in re.finditer(r"^(`{3,})(\S*)", content, re.MULTILINE):
        lang = open_match.group(2).strip().lower()
        if lang == "json":
            continue
        if not lang:
            continue  # skip bare fences without language — can't target them reliably
        block = _extract_outermost_fenced_block(content, [lang])
        if block:
            return block

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
        print(f"[WARN] Template not found at {path}, using embedded default.", file=sys.stderr)
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


class PhasedOrchestrator:
    """Phased spec discussion orchestrator.

    Runs 4 deterministic phases:
      1. Parallel Critique — both agents independently identify issues
      2. Consolidation — Codex merges and deduplicates issues
      3. Resolution — Claude resolves, Codex reviews (up to max_resolution_rounds)
      4. Finalization — (optional) rewrite spec files from working-spec.md
    """

    def __init__(
        self,
        config: dict[str, Any],
        config_dir: Path,
        project_root: Path,
        spec_path: Path,
        specs_root_override: Path | None,
        codebase_paths_override: list[str] | None,
        run_id: str | None,
        max_resolution_rounds_override: int | None,
        context_files_override: list[str] | None = None,
        no_apply: bool = False,
    ) -> None:
        self.config = config
        self.config_dir = config_dir
        self.no_apply = no_apply
        self.project_root = project_root.resolve()
        self.specs_root = self._resolve_specs_root(specs_root_override)
        self.spec_path = self._resolve_spec_path(spec_path)

        disc = config.get("discussion", {})
        self.max_resolution_rounds = max_resolution_rounds_override or int(
            disc.get("max_resolution_rounds", 2)
        )
        if self.max_resolution_rounds < 1:
            raise OrchestrationError("max_resolution_rounds must be >= 1")
        self.max_new_issues_per_round = int(disc.get("max_new_issues_per_resolution_round", 3))

        self.spec_glob = str(disc.get("spec_glob", "*.md"))
        self.specs_index_max_files = int(disc.get("specs_index_max_files", 200))
        self.require_codebase_compare = bool(disc.get("require_codebase_compare", True))
        configured_codebase_paths = disc.get("codebase_paths", ["."])
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
        self.phases_dir = self.run_dir / "phases"
        self.input_spec_file = self.run_dir / "input-spec.md"
        self.working_spec_file = self.run_dir / "working-spec.md"
        self.ledger_file = self.run_dir / "issue-ledger.json"
        self.context_summary_file = self.run_dir / "context-summary.md"
        self.verdict_file = self.run_dir / "verdict.md"
        self.action_brief_file = self.run_dir / "claude-action-brief.md"
        self.run_meta_file = self.run_dir / "run-meta.json"

        self.high_severity_labels = {
            normalize_severity(s) for s in disc.get("high_severity_labels", ["high", "critical"])
        }
        if not self.high_severity_labels:
            self.high_severity_labels = {"high", "critical"}

        self.context_files = (
            context_files_override if context_files_override is not None
            else disc.get("context_files", [])
        )

        # Load all 5 phase templates
        templates = config.get("templates", {})
        self.phase1_codex_template = self._load_template(templates, "phase1_codex_critique")
        self.phase1_claude_template = self._load_template(templates, "phase1_claude_critique")
        self.phase2_consolidate_template = self._load_template(templates, "phase2_consolidate")
        self.phase3_resolve_claude_template = self._load_template(templates, "phase3_resolve_claude")
        self.phase3_resolve_codex_template = self._load_template(templates, "phase3_resolve_codex")

        self._validate_agents_config()

    def _load_template(self, templates: dict[str, Any], key: str) -> str:
        path_str = templates.get(key)
        if not path_str:
            raise OrchestrationError(f"Missing required template config: templates.{key}")
        template = resolve_template(path_str, "", self.config_dir)
        if not template:
            raise OrchestrationError(
                f"Template file is empty or missing for templates.{key}: {path_str}"
            )
        return template

    def _validate_agents_config(self) -> None:
        agents = self.config.get("agents", {})
        for name in ("codex", "claude"):
            if name not in agents:
                raise OrchestrationError(f"Missing config.agents.{name}")
            if "command" not in agents[name]:
                raise OrchestrationError(f"Missing config.agents.{name}.command")

    # --- Path resolution ---

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

    # --- Context building ---

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

    def _read_spec_content(self) -> str:
        """Read spec from a file or directory.

        If spec_path is a file, reads it directly.
        If spec_path is a directory, recursively finds all .md files, orders
        them (readme/index first, then alphabetically by relative path), and
        concatenates them with file-path headers so agents know the source
        structure.
        """
        if self.spec_path.is_file():
            return read_text(self.spec_path)

        if not self.spec_path.is_dir():
            raise OrchestrationError(f"Spec path is neither a file nor directory: {self.spec_path}")

        md_files = sorted(self.spec_path.rglob("*.md"))
        if not md_files:
            raise OrchestrationError(f"No .md files found in spec directory: {self.spec_path}")

        priority_names = {"readme.md", "index.md"}
        top_files: list[Path] = []
        rest_files: list[Path] = []
        for f in md_files:
            if f.name.lower() in priority_names and f.parent == self.spec_path:
                top_files.append(f)
            else:
                rest_files.append(f)
        ordered = top_files + rest_files

        chunks: list[str] = []
        for f in ordered:
            rel = f.relative_to(self.spec_path)
            chunks.append(f"<!-- file: {rel} -->\n")
            chunks.append(read_text(f))
            chunks.append("\n")

        combined = "\n".join(chunks).strip()
        print(
            f"Assembled spec from directory: {len(ordered)} files, {len(combined)} chars",
            file=sys.stderr,
        )
        return combined

    def _read_extra_context_files(self) -> str:
        paths = self.context_files
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

    def build_context_packet(self, phase_label: str = "") -> str:
        ledger = self.load_ledger()
        open_issues = [
            issue for issue in ledger.get("issues", [])
            if normalize_status(issue.get("status", "open")) in OPEN_STATUSES
        ]
        open_issues_md = "\n".join(
            f"- {i.get('id', i.get('temp_id', '?'))} [{i.get('severity', 'medium')}] "
            f"{i.get('title', '')} (status: {i.get('status', 'open')})"
            for i in open_issues
        )
        if not open_issues_md:
            open_issues_md = "- None"

        summary = ""
        if self.context_summary_file.exists():
            summary = read_text(self.context_summary_file).strip()
        if not summary:
            summary = "# Context Summary\n\n- No prior phases."

        extra_context = self._read_extra_context_files()
        specs_index = self._build_specs_index()
        codebase_scope = self._build_codebase_scope()

        return (
            f"Phase: {phase_label}\n"
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

    # --- Bootstrap & ledger ---

    def bootstrap(self) -> None:
        if not self.spec_path.exists():
            raise OrchestrationError(f"Spec path not found: {self.spec_path}")
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

        self.phases_dir.mkdir(parents=True, exist_ok=False)
        spec_content = self._read_spec_content()
        write_text(self.input_spec_file, spec_content)
        write_text(self.working_spec_file, spec_content)

        initial_ledger: dict[str, Any] = {
            "meta": {
                "created_at": utc_now(),
                "orchestrator_version": "2.0",
                "next_issue_number": 1,
            },
            "issues": [],
            "dropped_issues": [],
            "phase_digests": [],
        }
        save_json(self.ledger_file, initial_ledger)
        write_text(self.context_summary_file, "# Context Summary\n\n")

        run_meta = {
            "created_at": utc_now(),
            "orchestrator_version": "2.0",
            "project_root": str(self.project_root),
            "specs_root": str(self.specs_root),
            "spec_source": str(self.spec_path),
            "codebase_paths": [str(path) for path in self.codebase_paths],
            "require_codebase_compare": self.require_codebase_compare,
            "run_id": self.run_id,
            "max_resolution_rounds": self.max_resolution_rounds,
            "config_dir": str(self.config_dir),
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

    # --- Agent invocation ---

    def _invoke_agent(
        self,
        agent_name: str,
        prompt_text: str,
        prompt_path: Path,
        response_path: Path,
    ) -> str:
        agent_cfg = self.config["agents"][agent_name]
        raw_command = agent_cfg["command"]
        timeout = int(agent_cfg.get("timeout_seconds", 900))
        pass_stdin = bool(agent_cfg.get("pass_prompt_via_stdin", False))

        # Derive scratch file paths from response_path
        base_name = response_path.name.removesuffix(".response.md")
        scratch_output = response_path.parent / f"{base_name}.raw-output.txt"
        scratch_stderr = response_path.parent / f"{base_name}.stderr.txt"

        variables = {
            "prompt_file": str(prompt_path),
            "output_file": str(scratch_output),
            "project_root": str(self.project_root),
            "run_dir": str(self.run_dir),
        }

        env = os.environ.copy()
        for key, value in agent_cfg.get("env", {}).items():
            if value is None:
                env.pop(str(key), None)
            else:
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

    # =========================================================================
    # Phase 1: Parallel Critique
    # =========================================================================

    def _run_phase_critique(self) -> None:
        """Phase 1: Both agents independently critique the spec in parallel."""
        print("=== Phase 1: Parallel Critique ===", file=sys.stderr)

        codex_response_path = self.phases_dir / "01-critique-codex.response.md"
        claude_response_path = self.phases_dir / "01-critique-claude.response.md"
        codex_done = codex_response_path.exists()
        claude_done = claude_response_path.exists()

        if codex_done and claude_done:
            print("  Both critiques already exist, skipping.", file=sys.stderr)
            codex_critique = extract_json_block(read_text(codex_response_path))
            claude_critique = extract_json_block(read_text(claude_response_path))
        else:
            def invoke_critique(agent_name: str) -> dict[str, Any]:
                template = (
                    self.phase1_codex_template if agent_name == "codex"
                    else self.phase1_claude_template
                )
                context = self.build_context_packet("1 — Independent Critique")
                prompt = render_template(template, {
                    "WORKING_SPEC_FILE": str(self.working_spec_file),
                    "CONTEXT_PACKET": context,
                })
                prefix = f"01-critique-{agent_name}"
                prompt_path = self.phases_dir / f"{prefix}.prompt.md"
                resp_path = self.phases_dir / f"{prefix}.response.md"
                write_text(prompt_path, prompt)
                raw = self._invoke_agent(agent_name, prompt, prompt_path, resp_path)
                return extract_json_block(raw)

            if not codex_done and not claude_done:
                print("  Running both agents in parallel...", file=sys.stderr)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    codex_future = executor.submit(invoke_critique, "codex")
                    claude_future = executor.submit(invoke_critique, "claude")
                    # Collect results; propagate first exception
                    codex_critique = codex_future.result()
                    claude_critique = claude_future.result()
            elif not codex_done:
                print("  Codex critique missing, running...", file=sys.stderr)
                codex_critique = invoke_critique("codex")
                claude_critique = extract_json_block(read_text(claude_response_path))
            else:
                print("  Claude critique missing, running...", file=sys.stderr)
                codex_critique = extract_json_block(read_text(codex_response_path))
                claude_critique = invoke_critique("claude")

        # Update ledger with raw critique issues
        ledger = self.load_ledger()
        self._apply_critique_issues(ledger, codex_critique, "codex")
        self._apply_critique_issues(ledger, claude_critique, "claude")
        ledger["phase_digests"].append({
            "phase": "critique",
            "codex_digest": codex_critique.get("digest", ""),
            "claude_digest": claude_critique.get("digest", ""),
            "at": utc_now(),
        })
        self.save_ledger(ledger)

        self._append_context(
            f"## Phase 1: Parallel Critique\n"
            f"- Codex found {len(codex_critique.get('issues', []))} issues\n"
            f"- Claude found {len(claude_critique.get('issues', []))} issues\n"
            f"- Codex digest: {codex_critique.get('digest', '')}\n"
            f"- Claude digest: {claude_critique.get('digest', '')}\n\n"
        )

        codex_count = len(codex_critique.get("issues", []))
        claude_count = len(claude_critique.get("issues", []))
        print(f"  Codex: {codex_count} issues, Claude: {claude_count} issues", file=sys.stderr)

    # =========================================================================
    # Phase 2: Consolidation
    # =========================================================================

    def _run_phase_consolidate(self) -> None:
        """Phase 2: Codex merges both critiques into a consolidated issue list."""
        print("=== Phase 2: Consolidation ===", file=sys.stderr)

        codex_raw = read_text(self.phases_dir / "01-critique-codex.response.md")
        claude_raw = read_text(self.phases_dir / "01-critique-claude.response.md")
        codex_critique = extract_json_block(codex_raw)
        claude_critique = extract_json_block(claude_raw)

        context = self.build_context_packet("2 — Consolidation")
        prompt = render_template(self.phase2_consolidate_template, {
            "WORKING_SPEC_FILE": str(self.working_spec_file),
            "CODEX_CRITIQUE_JSON": json.dumps(codex_critique, indent=2, ensure_ascii=False),
            "CLAUDE_CRITIQUE_JSON": json.dumps(claude_critique, indent=2, ensure_ascii=False),
            "CONTEXT_PACKET": context,
        })

        prompt_path = self.phases_dir / "02-consolidate.prompt.md"
        response_path = self.phases_dir / "02-consolidate.response.md"
        write_text(prompt_path, prompt)

        raw = self._invoke_agent("codex", prompt, prompt_path, response_path)
        consolidation = extract_json_block(raw)

        # Rebuild ledger with consolidated issues (replaces temporary critique entries)
        ledger = self.load_ledger()
        self._apply_consolidation(ledger, consolidation)
        ledger["phase_digests"].append({
            "phase": "consolidation",
            "digest": consolidation.get("digest", ""),
            "at": utc_now(),
        })
        self.save_ledger(ledger)

        self._append_context(
            f"## Phase 2: Consolidation\n"
            f"- {len(consolidation.get('consolidated_issues', []))} consolidated issues\n"
            f"- {len(consolidation.get('dropped_issues', []))} dropped\n"
            f"- Digest: {consolidation.get('digest', '')}\n\n"
        )

        n_consolidated = len(consolidation.get("consolidated_issues", []))
        n_dropped = len(consolidation.get("dropped_issues", []))
        print(f"  {n_consolidated} consolidated issues, {n_dropped} dropped", file=sys.stderr)

    # =========================================================================
    # Phase 3: Resolution
    # =========================================================================

    def _run_phase_resolve(self, start_round: int = 1, resume_codex_only: bool = False) -> str:
        """Phase 3: Up to max_resolution_rounds of Claude resolve + Codex review.

        This phase is discussion-only — no spec modifications. Claude analyzes
        issues and produces JSON decisions; Codex reviews them. The actual spec
        rewrite happens in a separate step after resolution completes.
        """
        print("=== Phase 3: Resolution ===", file=sys.stderr)

        for round_num in range(start_round, self.max_resolution_rounds + 1):
            print(f"  Resolution round {round_num}/{self.max_resolution_rounds}", file=sys.stderr)

            # Run Claude resolve (unless resuming mid-round with claude already done)
            if not (round_num == start_round and resume_codex_only):
                claude_payload = self._run_resolve_claude(round_num)
            else:
                # Load existing Claude response for mid-round resume
                claude_resp_path = self.phases_dir / f"03-resolve-R{round_num:02d}-claude.response.md"
                claude_raw = read_text(claude_resp_path)
                try:
                    claude_payload = extract_json_block(claude_raw)
                except ValueError:
                    claude_payload = {"issue_responses": [], "round_digest": "(no structured response)"}

            # Run Codex review
            codex_payload = self._run_resolve_codex(round_num)

            # Apply resolution to ledger
            ledger = self.load_ledger()
            self._apply_resolution(ledger, claude_payload, codex_payload, round_num)
            ledger["phase_digests"].append({
                "phase": "resolve",
                "round": round_num,
                "claude_digest": claude_payload.get("round_digest", ""),
                "codex_digest": codex_payload.get("round_digest", ""),
                "at": utc_now(),
            })
            self.save_ledger(ledger)

            self._append_context(
                f"## Resolution Round {round_num}\n"
                f"- Claude: {claude_payload.get('round_digest', '')}\n"
                f"- Codex: {codex_payload.get('round_digest', '')}\n\n"
            )

            if self._all_issues_resolved(ledger):
                stop = f"All medium+ issues resolved after resolution round {round_num}"
                write_text(self.phases_dir / "03-resolve.done", stop + "\n")
                return stop

        stop = f"Max resolution rounds ({self.max_resolution_rounds}) reached"
        write_text(self.phases_dir / "03-resolve.done", stop + "\n")
        return stop

    def _run_resolve_claude(self, round_num: int) -> dict[str, Any]:
        """Claude responds to each issue with JSON decisions (no spec rewriting)."""
        context = self.build_context_packet(
            f"3 — Resolution, Round {round_num}/{self.max_resolution_rounds}"
        )

        # Optional reference to previous Codex review (round 2+)
        codex_review_line = ""
        if round_num > 1:
            prev_codex = self.phases_dir / f"03-resolve-R{round_num - 1:02d}-codex.response.md"
            if prev_codex.exists():
                codex_review_line = f"- Previous Codex review: {prev_codex}"

        prompt = render_template(self.phase3_resolve_claude_template, {
            "ROUND_NUMBER": str(round_num),
            "MAX_RESOLUTION_ROUNDS": str(self.max_resolution_rounds),
            "MAX_NEW_ISSUES": str(self.max_new_issues_per_round),
            "WORKING_SPEC_FILE": str(self.working_spec_file),
            "LEDGER_FILE": str(self.ledger_file),
            "CONTEXT_SUMMARY_FILE": str(self.context_summary_file),
            "CODEX_REVIEW_FILE_LINE": codex_review_line,
            "CONTEXT_PACKET": context,
        })

        prefix = f"03-resolve-R{round_num:02d}-claude"
        prompt_path = self.phases_dir / f"{prefix}.prompt.md"
        response_path = self.phases_dir / f"{prefix}.response.md"
        write_text(prompt_path, prompt)

        # Use read-only tools for Claude during resolution — no Bash, no Write
        agent_cfg = self.config["agents"]["claude"]
        resolve_cfg = dict(agent_cfg)
        resolve_cfg["command"] = ["claude", "-p", "--allowedTools", "Read,Glob,Grep"]
        original_cfg = self.config["agents"]["claude"]
        self.config["agents"]["claude"] = resolve_cfg
        try:
            raw = self._invoke_agent("claude", prompt, prompt_path, response_path)
        finally:
            self.config["agents"]["claude"] = original_cfg

        try:
            payload = extract_json_block(raw)
        except ValueError:
            print(
                f"[WARN] Resolve round {round_num}: Claude response has no JSON block.",
                file=sys.stderr,
            )
            payload = {"issue_responses": [], "round_digest": "(no structured response)"}

        return payload

    def _run_resolve_codex(self, round_num: int) -> dict[str, Any]:
        """Codex reviews Claude's resolution responses."""
        context = self.build_context_packet(
            f"3 — Resolution Review, Round {round_num}/{self.max_resolution_rounds}"
        )

        claude_response_path = self.phases_dir / f"03-resolve-R{round_num:02d}-claude.response.md"
        prompt = render_template(self.phase3_resolve_codex_template, {
            "ROUND_NUMBER": str(round_num),
            "MAX_RESOLUTION_ROUNDS": str(self.max_resolution_rounds),
            "WORKING_SPEC_FILE": str(self.working_spec_file),
            "LEDGER_FILE": str(self.ledger_file),
            "CONTEXT_SUMMARY_FILE": str(self.context_summary_file),
            "CLAUDE_RESPONSE_FILE": str(claude_response_path),
            "CONTEXT_PACKET": context,
        })

        prefix = f"03-resolve-R{round_num:02d}-codex"
        prompt_path = self.phases_dir / f"{prefix}.prompt.md"
        response_path = self.phases_dir / f"{prefix}.response.md"
        write_text(prompt_path, prompt)

        raw = self._invoke_agent("codex", prompt, prompt_path, response_path)
        return extract_json_block(raw)

    # =========================================================================
    # Spec Rewrite (post-resolution)
    # =========================================================================

    def _build_changes_summary(self) -> tuple[list[dict[str, Any]], str]:
        """Build a human-readable summary of agreed changes from the ledger.

        Returns (agreed_issues, formatted_summary_text).
        """
        ledger = self.load_ledger()
        agreed = [i for i in ledger.get("issues", []) if i.get("status") == "agreed"]

        if not agreed:
            return agreed, ""

        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("AGREED CHANGES TO APPLY")
        lines.append("=" * 70)

        for i, issue in enumerate(agreed, 1):
            severity = issue.get("severity", "medium").upper()
            lines.append(f"\n{i}. [{severity}] {issue.get('id', '?')}: {issue.get('title', '')}")
            lines.append(f"   Rationale: {issue.get('rationale', 'N/A')}")
            lines.append(f"   Proposed change: {issue.get('proposed_change', 'N/A')}")

            # Include specific spec_change from resolution if available
            for h in issue.get("history", []):
                if h.get("event") == "response" and h.get("by") == "claude":
                    sc = h.get("payload", {}).get("spec_change", "")
                    if sc:
                        lines.append(f"   Spec change: {sc}")

        lines.append("\n" + "=" * 70)
        return agreed, "\n".join(lines)

    def _present_changes_and_confirm(self) -> bool:
        """Present agreed changes to the user and ask for confirmation.

        Returns True if the user accepts, False otherwise.
        """
        agreed, summary = self._build_changes_summary()

        if not agreed:
            print("No agreed issues — nothing to apply.", file=sys.stderr)
            return False

        # Print to stdout so user sees it
        print(summary)
        print(f"\n{len(agreed)} changes will be applied to: {self.spec_path}")
        print("The original spec will be backed up before any changes.\n")

        try:
            answer = input("Apply these changes? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            return False

        return answer in ("y", "yes")

    def _apply_changes_to_spec(self) -> None:
        """Apply agreed changes directly to the original spec files using Codex.

        Backs up the spec directory first, then invokes Codex with full-auto
        to edit the original files in place.
        """
        print("=== Applying Changes to Spec ===", file=sys.stderr)

        apply_response = self.phases_dir / "04-apply.response.md"
        if apply_response.exists():
            print("  Changes already applied, skipping.", file=sys.stderr)
            return

        agreed, _ = self._build_changes_summary()
        if not agreed:
            print("  No agreed issues — skipping.", file=sys.stderr)
            return

        # Backup original spec before editing
        backup_path = self.run_dir / "spec-backup"
        if not backup_path.exists():
            if self.spec_path.is_dir():
                shutil.copytree(self.spec_path, backup_path)
            else:
                backup_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.spec_path, backup_path / self.spec_path.name)
            print(f"  Backed up spec to: {backup_path}", file=sys.stderr)
        else:
            print(f"  Backup already exists at: {backup_path}", file=sys.stderr)

        # Build change manifest
        changes_md = "\n".join(
            f"- **{i['id']}** [{i.get('severity', 'medium')}]: {i.get('title', '')}\n"
            f"  Decision rationale: {i.get('claude_response', 'N/A')}\n"
            f"  Proposed change: {i.get('proposed_change', 'N/A')}"
            for i in agreed
        )

        spec_changes: list[str] = []
        for i in agreed:
            for h in i.get("history", []):
                if h.get("event") == "response" and h.get("by") == "claude":
                    payload = h.get("payload", {})
                    sc = payload.get("spec_change", "")
                    if sc:
                        spec_changes.append(f"- **{i['id']}**: {sc}")
        spec_changes_md = "\n".join(spec_changes) if spec_changes else "- See proposed changes above"

        # Build file listing for Codex
        if self.spec_path.is_dir():
            md_files = sorted(self.spec_path.rglob("*.md"))
            file_listing = "\n".join(f"- {f}" for f in md_files)
        else:
            file_listing = f"- {self.spec_path}"

        prompt = (
            "You are applying agreed changes from a multi-agent spec review to the original spec files.\n\n"
            "TASK\n"
            "1. Read each spec file listed below.\n"
            f"2. Read the issue ledger from: {self.ledger_file}\n"
            "3. Apply ALL agreed changes to the appropriate files.\n"
            "4. Edit each file in place using your file editing tools.\n\n"
            "SPEC FILES TO EDIT\n"
            f"{file_listing}\n\n"
            "AGREED CHANGES TO APPLY\n"
            f"{changes_md}\n\n"
            "SPECIFIC SPEC CHANGES\n"
            f"{spec_changes_md}\n\n"
            "RULES\n"
            "- Edit files in place. Do NOT create new files unless a change explicitly requires it.\n"
            "- Preserve all existing content that is not affected by the changes.\n"
            "- Preserve all markdown formatting, code blocks, tables, and lists exactly.\n"
            "- Apply each change to the correct file based on which task/section it refers to.\n"
            "- Do NOT reorganize or restructure the files — only apply the agreed changes.\n"
            "- After editing, briefly confirm which files were modified and what changes were made.\n"
        )

        prefix = "04-apply"
        prompt_path = self.phases_dir / f"{prefix}.prompt.md"
        response_path = self.phases_dir / f"{prefix}.response.md"
        write_text(prompt_path, prompt)

        raw = self._invoke_agent("codex", prompt, prompt_path, response_path)
        print(f"  Changes applied. Response at: {response_path}", file=sys.stderr)

    # =========================================================================
    # Ledger update methods
    # =========================================================================

    def _apply_critique_issues(
        self, ledger: dict[str, Any], critique: dict[str, Any], agent_name: str
    ) -> None:
        """Add raw critique issues to ledger (no ISSUE-NNN IDs — temporary entries)."""
        for i, issue in enumerate(critique.get("issues", [])):
            entry = {
                "temp_id": f"{agent_name}-{i}",
                "title": str(issue.get("title", "Untitled")).strip(),
                "severity": normalize_severity(issue.get("severity", "medium")),
                "category": str(issue.get("category", "")).strip(),
                "source": agent_name,
                "status": "open",
                "rationale": str(issue.get("rationale", "")).strip(),
                "proposed_change": str(issue.get("proposed_change", "")).strip(),
                "spec_sections": issue.get("spec_sections", []),
                "codebase_evidence": issue.get("codebase_evidence", []),
            }
            ledger["issues"].append(entry)

    def _apply_consolidation(
        self, ledger: dict[str, Any], consolidation: dict[str, Any]
    ) -> None:
        """Replace temporary critique issues with consolidated issues bearing ISSUE-NNN IDs."""
        ledger["issues"] = []

        for item in consolidation.get("consolidated_issues", []):
            raw_id = str(item.get("id", "")).strip()
            if raw_id:
                # Parse the number to keep next_issue_number in sync
                try:
                    num = int(raw_id.replace("ISSUE-", ""))
                    if num >= ledger["meta"].get("next_issue_number", 1):
                        ledger["meta"]["next_issue_number"] = num + 1
                except ValueError:
                    pass
                issue_id = raw_id
            else:
                issue_id = self.next_issue_id(ledger)

            entry = {
                "id": issue_id,
                "title": str(item.get("title", "Untitled")).strip(),
                "severity": normalize_severity(item.get("severity", "medium")),
                "category": str(item.get("category", "")).strip(),
                "sources": item.get("sources", []),
                "status": "open",
                "rationale": str(item.get("rationale", "")).strip(),
                "proposed_change": str(item.get("proposed_change", "")).strip(),
                "priority": int(item.get("priority", 99)),
                "claude_response": "",
                "resolution_notes": "",
                "history": [{"phase": "consolidation", "event": "created"}],
            }
            ledger["issues"].append(entry)

        ledger["dropped_issues"] = consolidation.get("dropped_issues", [])

    def _apply_resolution(
        self,
        ledger: dict[str, Any],
        claude_payload: dict[str, Any],
        codex_payload: dict[str, Any],
        round_num: int,
    ) -> None:
        """Process Claude's issue_responses and Codex's reviews."""
        issues_by_id = {issue["id"]: issue for issue in ledger["issues"] if "id" in issue}

        # Apply Claude's responses
        for response in claude_payload.get("issue_responses", []):
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

            issue["status"] = normalize_status(
                response.get("new_status", mapped_status), default=mapped_status
            )
            issue["claude_response"] = str(response.get("reason", "")).strip()
            issue.setdefault("history", []).append({
                "phase": "resolve",
                "round": round_num,
                "by": "claude",
                "event": "response",
                "payload": response,
            })

        # Apply Codex's reviews (escalations)
        for review in codex_payload.get("issue_reviews", []):
            issue_id = str(review.get("id", "")).strip()
            if not issue_id or issue_id not in issues_by_id:
                continue
            issue = issues_by_id[issue_id]

            if review.get("escalate"):
                issue["status"] = "escalated"
                note = str(review.get("notes", "")).strip()
                if note:
                    issue["resolution_notes"] = note

            issue.setdefault("history", []).append({
                "phase": "resolve",
                "round": round_num,
                "by": "codex",
                "event": "review",
                "payload": review,
            })

        # Process new issues from Claude (capped at max_new_issues_per_round)
        new_issues = claude_payload.get("new_issues", [])[:self.max_new_issues_per_round]
        for new_issue in new_issues:
            title = str(new_issue.get("title", "Untitled issue")).strip()

            # Check codex verdict on this new issue
            legitimate = True
            for verdict in codex_payload.get("new_issue_verdicts", []):
                if str(verdict.get("title", "")).strip() == title:
                    legitimate = verdict.get("legitimate", True)
                    break

            if legitimate:
                issue_id = self.next_issue_id(ledger)
                entry = {
                    "id": issue_id,
                    "title": title,
                    "severity": normalize_severity(new_issue.get("severity", "medium")),
                    "sources": [{"agent": "claude", "phase": "resolve", "round": round_num}],
                    "status": "open",
                    "rationale": str(new_issue.get("rationale", "")).strip(),
                    "proposed_change": str(new_issue.get("proposed_change", "")).strip(),
                    "claude_response": "",
                    "resolution_notes": "",
                    "history": [{
                        "phase": "resolve",
                        "round": round_num,
                        "by": "claude",
                        "event": "new",
                        "payload": new_issue,
                    }],
                }
                ledger["issues"].append(entry)

    def _all_issues_resolved(self, ledger: dict[str, Any]) -> bool:
        """Check if all medium+ severity issues are settled (not in OPEN_STATUSES)."""
        for issue in ledger.get("issues", []):
            status = normalize_status(issue.get("status", "open"))
            if status not in OPEN_STATUSES:
                continue
            severity = normalize_severity(issue.get("severity", "medium"))
            if severity in self.high_severity_labels or severity == "medium":
                return False
        return True

    # =========================================================================
    # Resume
    # =========================================================================

    def resume_state(self) -> tuple[str, int]:
        """Scan phases/ for existing artifacts and return (phase_name, sub_round).

        Returns:
            ("critique", 0) — need to run/complete critique phase
            ("consolidate", 0) — both critiques exist, need consolidation
            ("resolve", N) — need to start/resume resolution at round N
            ("resolve-codex", N) — claude done for round N, codex pending
            ("rewrite", 0) — resolution done, spec rewrite pending
            ("done", 0) — all phases complete
        """
        if not self.run_dir.exists():
            raise OrchestrationError(f"Cannot resume: run directory not found: {self.run_dir}")
        if not self.ledger_file.exists():
            raise OrchestrationError(f"Cannot resume: ledger not found: {self.ledger_file}")
        if not self.working_spec_file.exists():
            raise OrchestrationError(f"Cannot resume: working-spec.md not found")

        codex_critique = self.phases_dir / "01-critique-codex.response.md"
        claude_critique = self.phases_dir / "01-critique-claude.response.md"
        if not codex_critique.exists() or not claude_critique.exists():
            return ("critique", 0)

        consolidation = self.phases_dir / "02-consolidate.response.md"
        if not consolidation.exists():
            return ("consolidate", 0)

        # Check if resolution completed (may have finished early)
        resolve_done = self.phases_dir / "03-resolve.done"
        if not resolve_done.exists():
            # Resolution not done yet — find where to resume
            for r in range(1, self.max_resolution_rounds + 1):
                claude_resolve = self.phases_dir / f"03-resolve-R{r:02d}-claude.response.md"
                codex_resolve = self.phases_dir / f"03-resolve-R{r:02d}-codex.response.md"
                if not claude_resolve.exists():
                    return ("resolve", r)
                if not codex_resolve.exists():
                    return ("resolve-codex", r)
            # All round artifacts exist but no .done marker — shouldn't happen, but treat as done
            return ("rewrite", 0)

        # Resolution complete — check if changes were applied
        apply_response = self.phases_dir / "04-apply.response.md"
        if not apply_response.exists():
            return ("apply", 0)

        return ("done", 0)

    # =========================================================================
    # Main run
    # =========================================================================

    def run(self, *, resume: bool = False, auto_apply: bool = False) -> None:
        if resume:
            phase, sub_round = self.resume_state()
            if phase == "done":
                print("All phases already complete. Nothing to resume.", file=sys.stderr)
                self._generate_verdict("Resumed — all phases already complete")
                self._generate_action_brief("Resumed — all phases already complete")
                return
            print(f"Resuming from phase '{phase}' (sub_round={sub_round})", file=sys.stderr)
        else:
            self.bootstrap()
            phase, sub_round = "critique", 0

        stop_reason = "Completed all phases"

        # Phase 1: Parallel Critique
        if phase == "critique":
            self._run_phase_critique()
            phase = "consolidate"

        # Phase 2: Consolidation
        if phase == "consolidate":
            self._run_phase_consolidate()
            phase = "resolve"
            sub_round = 1

        # Phase 3: Resolution (discussion only — no spec modifications)
        if phase in ("resolve", "resolve-codex"):
            stop_reason = self._run_phase_resolve(
                start_round=sub_round,
                resume_codex_only=(phase == "resolve-codex"),
            )
            phase = "rewrite"

        # Phase 3b: Present changes and apply to original spec
        if phase in ("rewrite", "confirm", "apply"):
            # Load stop_reason from marker if resuming
            resolve_done = self.phases_dir / "03-resolve.done"
            if resolve_done.exists():
                stop_reason = read_text(resolve_done).strip() or stop_reason

            # Generate verdict and action brief (always)
            self._generate_verdict(stop_reason)
            self._generate_action_brief(stop_reason)

            # --no-apply: stop after generating verdict/action-brief
            if self.no_apply:
                print(
                    f"--no-apply: stopping after resolution. "
                    f"Use --resume --run-id {self.run_id} --auto-apply to apply changes.",
                    file=sys.stderr,
                )
                return

            # Check if changes were already applied
            apply_response = self.phases_dir / "04-apply.response.md"
            if apply_response.exists():
                print("Changes already applied to spec.", file=sys.stderr)
            else:
                # Present changes and ask for confirmation
                agreed, summary = self._build_changes_summary()
                if not agreed:
                    print("No agreed issues — nothing to apply.", file=sys.stderr)
                elif auto_apply or self._present_changes_and_confirm():
                    if auto_apply and agreed:
                        print(summary)
                        print(f"\n--auto-apply: applying {len(agreed)} changes to {self.spec_path}")
                    self._apply_changes_to_spec()
                else:
                    print("Changes not applied. Working spec and ledger are preserved in the run directory.",
                          file=sys.stderr)
            return

        # Generate verdict and action brief
        self._generate_verdict(stop_reason)
        self._generate_action_brief(stop_reason)

    # =========================================================================
    # Verdict & action brief
    # =========================================================================

    def _generate_verdict(self, stop_reason: str) -> None:
        ledger = self.load_ledger()
        issues = ledger.get("issues", [])

        agreed = [i for i in issues if i.get("status") == "agreed"]
        rejected = [i for i in issues if i.get("status") == "rejected-with-rationale"]
        unresolved = [
            i for i in issues if i.get("status") in {"open", "needs-evidence", "escalated"}
        ]

        def fmt(items: list[dict[str, Any]]) -> str:
            if not items:
                return "- None"
            return "\n".join(
                f"- {i.get('id', '?')} [{i.get('severity', 'medium')}] {i.get('title', '')}"
                for i in items
            )

        content = (
            "# Verdict\n\n"
            f"- Generated at: {utc_now()}\n"
            f"- Stop reason: {stop_reason}\n"
            f"- Run directory: `{self.run_dir}`\n"
            f"- Orchestrator version: 2.0\n\n"
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
        issues = ledger.get("issues", [])
        agreed = [i for i in issues if i.get("status") == "agreed"]
        unresolved = [
            i for i in issues if i.get("status") in {"open", "needs-evidence", "escalated"}
        ]

        agreed_lines = "\n".join(
            f"- {i.get('id', '?')}: {i.get('title', '')} -> {i.get('proposed_change', '')}"
            for i in agreed
        ) or "- None"
        unresolved_lines = "\n".join(
            f"- {i.get('id', '?')} [{i.get('status')}]: {i.get('title', '')}"
            for i in unresolved
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

    # =========================================================================
    # Finalization (Phase 4, optional)
    # =========================================================================

    def finalize_spec(self) -> None:
        """Backup the original spec directory/file and rewrite it with the agreed working spec.

        Uses the Claude agent with full write permissions to decompose the
        single working-spec.md back into the original file structure.
        """
        if not self.working_spec_file.exists():
            raise OrchestrationError("Cannot finalize: working-spec.md not found")

        is_dir = self.spec_path.is_dir()

        # Create backup
        backup_path = self.run_dir / "spec-backup"
        if is_dir:
            shutil.copytree(self.spec_path, backup_path)
            print(f"Backed up spec directory to: {backup_path}", file=sys.stderr)
        else:
            backup_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.spec_path, backup_path / self.spec_path.name)
            print(f"Backed up spec file to: {backup_path / self.spec_path.name}", file=sys.stderr)

        # Build the finalization prompt
        finalize_prompt = self._build_finalize_prompt()
        prompt_path = self.run_dir / "finalize.prompt.md"
        response_path = self.run_dir / "finalize.response.md"
        write_text(prompt_path, finalize_prompt)

        # Use the Claude agent config as base but with full permissions
        agent_cfg = self.config["agents"]["claude"]
        finalize_cfg = dict(agent_cfg)
        finalize_cfg["command"] = ["claude", "-p"]
        finalize_cfg["pass_prompt_via_stdin"] = True

        original_claude_cfg = self.config["agents"]["claude"]
        self.config["agents"]["claude"] = finalize_cfg
        try:
            self._invoke_agent("claude", finalize_prompt, prompt_path, response_path)
        finally:
            self.config["agents"]["claude"] = original_claude_cfg

        print(f"Spec finalization complete. Response at: {response_path}", file=sys.stderr)

    def _build_finalize_prompt(self) -> str:
        is_dir = self.spec_path.is_dir()
        verdict = read_text(self.verdict_file) if self.verdict_file.exists() else ""

        if is_dir:
            file_list = sorted(self.spec_path.rglob("*.md"))
            file_listing = "\n".join(f"- {f.relative_to(self.spec_path)}" for f in file_list)
            target_desc = f"spec directory at: {self.spec_path}\n\nExisting files:\n{file_listing}"
        else:
            target_desc = f"spec file at: {self.spec_path}"

        return (
            "You are rewriting a spec based on an agreed revision from a multi-agent review process.\n\n"
            "TASK\n"
            f"1. Read the agreed working spec from: {self.working_spec_file}\n"
            f"2. Read the issue ledger from: {self.ledger_file}\n"
            f"3. Read the verdict from: {self.verdict_file}\n"
            f"4. Rewrite the {target_desc}\n\n"
            "RULES\n"
            "- Preserve the original file structure. Each `<!-- file: ... -->` comment in the working spec "
            "marks where that file's content begins.\n"
            "- Write each file using the Write tool. Maintain the same filenames and directory structure.\n"
            "- Do NOT change the meaning of any agreed content — only restructure back into files.\n"
            "- If the working spec contains content not attributable to any original file, "
            "add it to the most appropriate existing file.\n"
            "- Preserve all markdown formatting, code blocks, tables, and lists exactly.\n\n"
            f"VERDICT SUMMARY\n{verdict}\n"
        )

    # --- Helpers ---

    def _append_context(self, text: str) -> None:
        current = ""
        if self.context_summary_file.exists():
            current = read_text(self.context_summary_file)
        write_text(self.context_summary_file, current + text)


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phased spec discussion orchestrator (v2) — Claude + Codex"
    )
    parser.add_argument("--config", default=None, help="Path to JSON config file (optional; uses built-in defaults if omitted)")
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to spec markdown file or directory. If a directory, all .md files are "
        "recursively assembled. Relative paths resolve against --specs-root.",
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
        help="Repeatable. Limit codebase inspection scope (overrides discussion.codebase_paths).",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root where output_root is resolved (default: current directory)",
    )
    parser.add_argument("--run-id", default=None, help="Optional run ID slug")
    parser.add_argument(
        "--max-resolution-rounds",
        type=int,
        default=None,
        help="Override max resolution rounds (default: 2)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume an interrupted run. Requires --run-id to identify which run to resume.",
    )
    parser.add_argument(
        "--auto-apply",
        action="store_true",
        default=False,
        help="Skip confirmation prompt and apply agreed changes automatically.",
    )
    parser.add_argument(
        "--context-file",
        action="append",
        default=None,
        help="Repeatable. Extra context files for agents (e.g., CLAUDE.md).",
    )
    parser.add_argument(
        "--no-apply",
        action="store_true",
        default=False,
        help="Run phases 1-3 only. Generate verdict and action brief but do not apply changes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.resume and not args.run_id:
        print("--resume requires --run-id to identify which run to resume.", file=sys.stderr)
        return 1

    if args.config:
        config_path = Path(args.config).resolve()
        if not config_path.exists():
            print(f"Config not found: {config_path}", file=sys.stderr)
            return 1
        try:
            config = load_json(config_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load config JSON: {exc}", file=sys.stderr)
            return 1
        config_dir = config_path.parent
    else:
        config = DEFAULT_CONFIG
        config_dir = _SCRIPT_DIR

    orchestrator = PhasedOrchestrator(
        config=config,
        config_dir=config_dir,
        project_root=Path(args.project_root),
        spec_path=Path(args.spec),
        specs_root_override=Path(args.specs_root) if args.specs_root else None,
        codebase_paths_override=args.codebase_path,
        run_id=args.run_id,
        max_resolution_rounds_override=args.max_resolution_rounds,
        context_files_override=args.context_file,
        no_apply=args.no_apply,
    )

    try:
        orchestrator.run(resume=args.resume, auto_apply=args.auto_apply)
    except (OrchestrationError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print("Run completed successfully.")
    print(f"Run directory: {orchestrator.run_dir}")
    print(f"Run ID: {orchestrator.run_id}")
    print(f"Verdict: {orchestrator.verdict_file}")
    print(f"Action brief: {orchestrator.action_brief_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
