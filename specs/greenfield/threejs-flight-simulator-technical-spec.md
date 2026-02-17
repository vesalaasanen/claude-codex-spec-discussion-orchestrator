# Three.js Single-Player Flight Simulator - Technical Specification (Greenfield)

## 1. Runtime Architecture
- Frontend app bootstrapped with Vite + TypeScript.
- Render loop separated from simulation loop.
- Core modules:
  - `engine/` (scene, renderer, camera orchestration)
  - `flight/` (aerodynamics model, control input translation)
  - `missions/` (objective states and transitions)
  - `ui/` (HUD, menu, summary overlays)
  - `persistence/` (local storage adapters)

## 2. Update Loop
- Fixed-timestep simulation target: 60 Hz.
- Variable render loop with interpolation support.
- Input sampled once per render frame and consumed by simulation step.

## 3. Data Model
- Aircraft tuning JSON:
  - mass
  - lift coefficient
  - drag coefficient
  - throttle response curve
  - control authority multipliers
- Mission definition JSON:
  - objective type
  - checkpoints
  - time limits
  - fail states
  - score weights

## 4. Rendering Plan
- Terrain mesh with tiled materials and baked normals.
- Directional light + sky model for readability.
- Optional post-processing kept minimal in MVP.

## 5. Physics Plan
- Gameplay-oriented simplified aerodynamics model.
- Collision system based on broad-phase bounds + narrow-phase checks.
- Ground interaction model for taxiing and landing scoring.

## 6. UI Stack
- HUD rendered with HTML/CSS overlay for rapid iteration.
- In-game menus and results UI as component-based overlays.
- Input rebinding deferred until post-MVP.

## 7. Persistence
- `settings` key for sensitivity, assists, graphics preset.
- `progress` key for medals and best times.
- Backward-compatible migration strategy required once schema evolves.

## 8. Testing Hooks
- Deterministic seed mode for mission replay validation.
- Debug overlay for simulation values:
  - AoA estimate
  - vertical speed
  - target objective state

## 9. Open Technical Questions
- Should wind be sampled from procedural noise or scripted mission curves?
- Should mission data be versioned by semantic id or content hash?
- Final target for frame-time and memory budget is still TBD.
