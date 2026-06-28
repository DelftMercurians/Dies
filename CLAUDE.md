# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Dies is the RoboCup SSL (Small Size League) framework for the Delft Mercurians team. It controls 6v6 soccer robots (~0.18m diameter) on a 6m × 9m field. The system processes vision data, runs strategy logic, and sends commands to robots via an RF basestation.

## Build & Run

Plain Cargo workspace — no `just`/justfile anymore. The CLI manages the
strategy build, and `dies-webui`'s `build.rs` handles TS bindings + frontend.

```bash
# Run dies in sim. How the strategy binary is managed (mutually exclusive flags):
#   --launch = use existing binary, --build = build once (default), --watch = hot-reload
cargo run -- --strategy concerto                      # build once + run (= old `just dev`)
cargo run -- --strategy concerto --watch              # build + hot-reload on rebuild
cargo run -- --strategy concerto --launch             # run existing binary, no build
cargo run -- --auto-start --strategy concerto         # auto-start the executor

mprocs                        # Dev: vite dev server + dies (--watch). Old `just webdev`.

cargo build -p dies-cli       # Build main binary only
cargo build -p concerto       # Build a specific strategy

# Release build of dies-cli + all strategies (build.rs produces the frontend
# bundle; needs `typeshare` + `pnpm` on PATH):
cargo build --release -p dies-cli -p concerto
```

**Strategy hot-reload:** with `--watch`, the CLI watches strategy
sources and runs `cargo build -p <strategy>` on change; the executor notices the
rebuilt binary (mtime) and hot-swaps the strategy process. `dies-webui/build.rs`
regenerates `webui/src/bindings.ts` on every build and runs `vite build` (into
`crates/dies-webui/static`) whenever the hash of the frontend inputs changes (any
profile) — dev iteration normally uses the vite dev server instead.

## Testing & Linting

```bash
cargo test                    # All tests
cargo test -p dies-world      # Single crate
cargo test test_name          # Single test by name
cargo fmt                     # Format
cargo clippy                  # Lint
cd webui && pnpm run tsc      # TypeScript type check
```

For hangs, `tokio-console` (install via `cargo install --locked tokio-console`) attaches to the running process and shows task state.

## Log analysis

Every run records a columnar binary log. This is the primary way to debug
behavior after the fact — what a robot actually did, what the strategy/controller
thought, why a skill fired. Analyze logs in Python with `tools/dieslog.py`.

### Where logs live & the format

- Written under `logs/` (gitignored), named `dies-YYYY-MM-DD_HH-MM-SS`
  (`crates/dies-webui/src/executor_task.rs`). Self-play writes under
  `logs/selfplay/...` via `--log-dir`.
- Two on-disk forms, both readable by `load()`:
  - **directory** `logs/dies-.../` — live recording, one Apache Arrow IPC stream
    per table (`{table}.arrow`) plus `meta.json`.
  - **`.dieslog` zip** — produced on close by compacting each `.arrow` to Zstd
    Parquet and zipping them (STORED). This is the durable artifact; the `.arrow`
    dir may be deleted after compaction.
- Format/schema is defined in `crates/dies-logger/src/schema.rs`. Writer:
  `writer.rs`; compaction: `compact.rs`. `meta.json` holds `field_geom`,
  strategies, `side_assignment`, `is_simulation`, `session_start_unix`.

### Tables (all per-frame tables keyed on `frame_id`)

`frames`, `ball`, `players`, `debug_values`, `debug_shapes`, `debug_tree`,
`settings_changes`, `events`, `markers`, `logs`, `vision`. Key columns:

- **frames**: `frame_id`, `t_received`, `t_capture`, `dt`, `game_state`,
  `operating_team`, `side_assignment`. This table maps `frame_id → time`; the
  reader uses it to build the shared `t` index (seconds from log start).
- **players**: `frame_id`, `team` (`"blue"`/`"yellow"`), `player_id`, pose +
  velocity `x, y, vx, vy, yaw, raw_yaw, angular_speed`, plus telemetry
  `primary_status`, `kicker_cap_voltage`, `pack_voltage_0/1`,
  `breakbeam_ball_detected`, `has_ball`, `handicaps`.
- **ball**: position/velocity of the ball, t-indexed.
- **debug_values**: `frame_id`, `key`, `value` (numeric) **or** `value_str`
  (string). This is the firehose of strategy/controller internals.
- **markers**: `frame_id`, `t`, `label` — user-dropped points of interest
  (double-space in the UI) used to segment a log into phases.

### Debug values — the naming convention

Recorded from Rust via `debug_value/debug_string/debug_cross/...` in
`crates/dies-core/src/debug_info.rs`. Keys are dotted `snake_case`. **Player-scoped
keys are namespaced `team_<Color>.p<id>.<tag>`** (capitalized color), e.g.
`team_Blue.p0.target_vel_x`, `team_Yellow.p3.breakbeam`. Global keys are bare
(`dt`, `game_state`, `ball_on_blue_side_for`). Vec2 values are stored as
`"x y"` strings and split by the reader into `<tag>_x` / `<tag>_y`. Grep the
strategy/executor for `debug_value(`/`debug_string(` to discover available keys.

### Reading logs in Python

Use the repo venv (has pyarrow/pandas/matplotlib): `.venv/bin/python`.
Every accessor returns a DataFrame indexed by `t` (seconds from log start), so
all robots/series share one clock. Window with `df.loc[t0:t1]`, take magnitudes
with `np.hypot`.

```python
import sys; sys.path.insert(0, "tools")
from dieslog import load
import numpy as np

log = load("logs/dies-2026-06-27_14-45-58")   # dir or .dieslog

log.robots()                 # -> [('blue', 0), ..., ('yellow', 5)]
log["players"]               # raw players table, t-indexed
log.ball                     # bare-attribute table access
log.debug("team_Blue.p0.")   # that robot's debug values, pivoted wide, prefix stripped
r = log.robot("blue", 0)     # player frames FUSED with team_Blue.p0.* debug (casing handled)

# commanded vs measured speed, first 15 s
r = log.robot("blue", 0).loc[0:15]
measured = np.hypot(r.vx, r.vy)
cmd      = np.hypot(r.target_vel_x, r.target_vel_y)

log.timeline(team="blue")    # overlaid per-robot speed plot with marker lines
log.seg(df, i)               # i-th segment cut by markers (markers split into len+1 pieces)
```

`tools/match_analytics.py` builds on `dieslog` for match-level metrics
(possession, shots, stoppages, goals, `run_frac`) — `analyze(path)`,
`summarize(glob)`, `report(df)`, `benchmark(...)`.

### Scratch space for analysis scripts

Put one-off/throwaway analysis scripts and plots in `.analysis/` (gitignored;
only its `README.md` is tracked). Don't clutter `tools/` or the repo root with
exploratory scripts. Promote anything genuinely reusable into
`tools/dieslog.py` / `tools/match_analytics.py`. Run with `.venv/bin/python`.

## Architecture

**Cargo workspace** with diamond-shaped dependency graph:

```
dies-core (minimal types: Vector2, Angle, PlayerId, etc.)
    ↓
feature crates (dies-executor, dies-world, dies-simulator, dies-webui, ...)
    ↓
dies-cli (entry point, glues everything together)
```

### Core data flow

1. **dies-ssl-client** receives vision packets + referee messages from SSL software
2. **dies-world** tracks world state using Kalman filtering on noisy position data
3. **dies-executor** runs the main loop: Tracker → TeamController → PlayerControllers → Skills
4. **Strategy process** (separate binary) communicates via IPC (bincode over Unix sockets)
5. **dies-basestation-client** sends commands to physical robots (or dies-simulator in sim mode)

### Strategy system

Strategies are separate binaries that implement the `Strategy` trait from `dies-strategy-api`:

- **dies-strategy-protocol**: IPC message types (HostMessage, StrategyMessage, SkillCommand)
- **dies-strategy-api**: Public interface (`Strategy` trait, `TeamContext`, `PlayerHandle`)
- **dies-strategy-runner**: Spawns strategy processes and manages IPC
- **strategies/**: `concerto` (current main, Formation + Plan-Execute-Replan). IPC uses Unix domain sockets — Linux/macOS only.

**Coordinate frame**: All strategy code uses team-relative coordinates (+x toward opponent goal). The executor handles transformation to/from absolute coordinates. Never expose absolute coordinates to strategies.

### Web UI

React/TypeScript frontend in `webui/` (Vite + Tailwind), using **pnpm** as package manager. Backend is Axum with WebSocket in `dies-webui`. TypeScript bindings are generated via `typeshare`.

## Conventions

- **Commits**: Conventional Commits lite — `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `wip:`, `misc:`
- **Branches**: `<author>/<feature-name>` (e.g. `balint/fix-ersim-env-win`)
- **Error handling**: `anyhow::Result` everywhere, no panics, propagate errors to executor
- **Logging**: `tracing` crate — error (fatal only), warn (recoverable), info (rare, user-facing), debug (everything else). No `trace`, no `println!`
- **Generics**: Use sparingly (YAGNI). Crates are kept granular for compile-time caching.

## Key crates

| Crate | Purpose |
|---|---|
| `dies-core` | Minimal shared types (nalgebra-based Vector2, Angle, PlayerId, FieldGeometry) |
| `dies-executor` | Main loop, TeamController, PlayerController, skill activation |
| `dies-world` | World state tracking, Kalman filter |
| `dies-simulator` | Physics simulator (replaces hardware for testing) |
| `dies-ssl-client` | Vision + referee message handling |
| `dies-protos` | Protobuf definitions (code-generated) |
| `dies-strategy-api` | Strategy trait + context types |
| `dies-strategy-protocol` | IPC message serialization |
| `dies-strategy-runner` | Strategy process spawning + IPC management |
| `dies-webui` | Axum server + WebSocket |
| `dies-logger` | Binary log format (protobuf + msgpack) for recording/replaying sessions |
| `dies-cli` | Binary entry point, CLI args (clap) |

## Key dependencies

Rust: tokio (async runtime), nalgebra (linear algebra), axum (web), serde/bincode (serialization), pyo3 (Python interop, optional mpc feature), clap (CLI), tracing (logging), anyhow/thiserror (errors), typeshare (TS bindings)

Web UI: React 18, Vite, Tailwind, Radix UI, TanStack Query, Jotai, Monaco Editor, Dockview
