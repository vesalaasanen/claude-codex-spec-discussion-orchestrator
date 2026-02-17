#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def _extract_issue_ids_from_prompt(prompt_text: str) -> set[str]:
    marker = "ISSUE_LEDGER_JSON"
    start = prompt_text.find(marker)
    if start == -1:
        return set()

    section = prompt_text[start + len(marker) :]
    end_marker = "\n\nWORKING_SPEC_MARKDOWN"
    end = section.find(end_marker)
    if end != -1:
        section = section[:end]

    json_start = section.find("{")
    json_end = section.rfind("}")
    if json_start == -1 or json_end == -1 or json_end <= json_start:
        return set()

    try:
        ledger = json.loads(section[json_start : json_end + 1])
    except json.JSONDecodeError:
        return set()

    issue_ids: set[str] = set()
    for issue in ledger.get("issues", []):
        issue_id = str(issue.get("id", "")).strip()
        if issue_id:
            issue_ids.add(issue_id)
    return issue_ids


def main() -> int:
    prompt_text = ""
    if len(sys.argv) > 1:
        prompt_path = Path(sys.argv[1])
        if prompt_path.exists():
            prompt_text = prompt_path.read_text(encoding="utf-8")

    issue_ids = _extract_issue_ids_from_prompt(prompt_text)
    issue_already_known = "ISSUE-001" in issue_ids

    if issue_already_known:
        payload = {
            "issue_updates": [
                {
                    "id": "ISSUE-001",
                    "status": "agreed",
                    "severity": "medium",
                    "notes": "Spec now contains explicit performance budgets and telemetry acceptance checks.",
                }
            ],
            "new_issues": [],
            "spec_edit_suggestions": [],
            "codebase_evidence": [],
            "round_digest": "No additional concerns. Prior issue is resolved.",
        }
    else:
        payload = {
            "issue_updates": [],
            "new_issues": [
                {
                    "title": "Add explicit performance budgets and telemetry thresholds",
                    "severity": "medium",
                    "rationale": "The spec does not define measurable FPS/frame-time/memory limits for an MVP release gate.",
                    "proposed_change": "Add a Performance Budget section with concrete render, simulation, and memory targets plus telemetry pass/fail checks.",
                }
            ],
            "spec_edit_suggestions": [
                {
                    "section": "Release Criteria",
                    "replace": "Performance budget is TBD and will be finalized after first implementation spike.",
                    "with": "Define explicit performance budgets and telemetry thresholds required for MVP release.",
                }
            ],
            "codebase_evidence": [],
            "round_digest": "Raised one medium issue requesting measurable performance targets.",
        }

    print("```json")
    print(json.dumps(payload, indent=2))
    print("```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
