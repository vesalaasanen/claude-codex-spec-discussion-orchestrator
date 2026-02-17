#!/usr/bin/env python3
from __future__ import annotations

import json


REVISED_SPEC = """# Three.js Single-Player Flight Simulator - Product Spec (Revised)

## 1. Vision
Build a browser-based single-player flight simulator that feels fast, readable, and skill-based. The player should be able to take off, navigate checkpoints, complete mission objectives, and land safely in a compact but believable environment.

## 2. Product Goals
1. Deliver an MVP that runs in modern desktop browsers without plugins.
2. Provide a satisfying \"one more run\" gameplay loop with clear scoring and mission outcomes.
3. Keep controls accessible for new players while preserving mastery depth.
4. Establish an extensible architecture for adding aircraft, mission types, and weather systems.

## 3. MVP Scope
- One flyable aircraft with tunable handling profile.
- One island environment with terrain, airstrip, and checkpoint rings.
- Three mission archetypes: time trial, cargo delivery, precision landing challenge.
- HUD for speed, altitude, heading, throttle, objective progress, and warnings.
- End-of-run summary with mission result and score breakdown.

## 4. Core Gameplay Loop
1. Choose mission.
2. Launch with pre-flight countdown.
3. Execute mission objectives while managing aircraft stability.
4. Complete mission, fail, or abort.
5. Review score breakdown and restart quickly.

## 5. Flight and Physics Requirements
- Deterministic update step for core flight calculations.
- Inputs: pitch, roll, yaw/rudder, throttle, brake/flaps (simplified).
- Collision detection against terrain and key world objects.
- Stall-like behavior with recovery window.
- Tuning variables exposed via config.

## 6. Technical Constraints
- Engine/rendering: Three.js.
- Runtime: browser only, no server required for MVP gameplay.
- Packaging: Vite-based build pipeline.
- Save data: localStorage for unlocked medals and settings.

## 7. Telemetry and Instrumentation
Record per-run summary metrics:
- mission id
- completion state
- completion time
- crash count
- score

## 8. Performance Budget (MVP Release Gate)
- Render FPS:
  - >= 60 FPS average on reference desktop profile at 1080p.
  - >= 45 FPS 1st percentile during mission-critical gameplay.
- Frame-time budget:
  - <= 16.7 ms average total frame time.
  - <= 22 ms 95th percentile frame time.
- Simulation cost:
  - <= 4 ms average per fixed simulation step at 60 Hz.
- Memory budget:
  - <= 600 MB peak JS heap during 15-minute session.
- Asset loading budget:
  - Initial mission-ready load <= 8 seconds on reference profile.

## 9. Performance Telemetry Pass/Fail Checks
- Log frame time and FPS histogram once per run.
- Log peak JS heap and simulation-step timing.
- Mark run as release-blocking if any budget threshold fails.
- Include budget pass/fail summary in end-of-run diagnostics export.

## 10. Release Criteria
- All MVP mission archetypes are playable from start to finish.
- No blocker defects in flight controls, mission completion, or save/load settings.
- Performance budgets and telemetry pass/fail checks are implemented and verified.
"""


def main() -> int:
    payload = {
        "issue_responses": [
            {
                "id": "ISSUE-001",
                "decision": "accept",
                "new_status": "agreed",
                "reason": "Added explicit FPS/frame-time/memory budgets and telemetry release-gate checks.",
            }
        ],
        "applied_changes_summary": [
            "Added measurable Performance Budget thresholds.",
            "Added telemetry pass/fail criteria tied to release readiness.",
            "Updated release criteria to require budget compliance.",
        ],
        "codebase_evidence": [],
        "round_digest": "Accepted ISSUE-001 and revised the full spec with measurable performance criteria.",
    }

    print("```json")
    print(json.dumps(payload, indent=2))
    print("```")
    print("```markdown")
    print(REVISED_SPEC)
    print("```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
