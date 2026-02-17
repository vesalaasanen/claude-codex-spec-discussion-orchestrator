# Three.js Single-Player Flight Simulator - Content and Progression Spec (Greenfield)

## 1. Mission Catalog (MVP)

### Mission A: Harbor Sprint
- Type: Time trial
- Goal: pass 10 checkpoint rings and return to runway
- Fail states:
  - crash
  - timeout

### Mission B: Ridge Courier
- Type: Cargo delivery
- Goal: carry virtual cargo through turbulence corridor
- Scoring factors:
  - completion time
  - cargo integrity
  - smoothness penalties

### Mission C: Night Approach
- Type: Precision landing
- Goal: land within target zone under low-light conditions
- Scoring factors:
  - touchdown speed
  - lateral offset
  - runway centerline alignment

## 2. Difficulty Progression
- Bronze/Silver/Gold medal thresholds per mission.
- Difficulty ramps by:
  - tighter checkpoint placement
  - stronger crosswind
  - stricter landing tolerance

## 3. World Points of Interest
- Airstrip and hangar zone.
- Coastal cliffs.
- Industrial harbor.
- Mountain pass.
- Emergency landing field.

## 4. Onboarding
- First-launch tutorial card set:
  - controls overview
  - throttle and lift basics
  - stall warning behavior
- Optional guided practice lap before first mission.

## 5. Audio Direction
- Adaptive engine audio by throttle and RPM band.
- Wind noise scaling by airspeed.
- Distinct mission event cues:
  - checkpoint clear
  - warning
  - mission complete/fail

## 6. Art Direction
- Semi-realistic, high-readability visual style.
- Strong contrast for mission-critical objects.
- Minimal clutter near flight path.

## 7. Accessibility Notes
- Colorblind-safe palette for objectives and warnings.
- Subtitle-style text for mission-critical voice cues.
- Configurable camera inversion and sensitivity.
