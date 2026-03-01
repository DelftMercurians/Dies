# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Dies is the RoboCup SSL (Small Size League) framework for the Delft Mercurians team. It controls 6v6 soccer robots (~0.18m diameter) on a 6m × 9m field. The system processes vision data, runs strategy logic, and sends commands to robots via an RF basestation.

## Build & Run

Uses `just` (justfile) for task automation and Cargo workspaces.

```bash
just dev [strategy]           # Build strategy + run dies in simulation (default: concerto)
just build                    # Release build of dies-cli + all strategies
just webdev [strategy]        # Run Vite dev server + dies
just webbuild                 # Generate TS bindings + build webui

cargo build -p dies-cli       # Build main binary only
cargo build -p concerto       # Build a specific strategy
cargo run -- --auto-start --strategy concerto  # Run with options
```

## Testing & Linting

```bash
cargo test                    # All tests
cargo test -p dies-world      # Single crate
cargo test test_name          # Single test by name
cargo fmt                     # Format
cargo clippy                  # Lint
cd webui && pnpm run tsc      # TypeScript type check
```

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
- **strategies/concerto**: Current main strategy (Formation + Plan-Execute-Replan)

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
| `dies-cli` | Binary entry point, CLI args (clap) |

## Key dependencies

Rust: tokio (async runtime), nalgebra (linear algebra), axum (web), serde/bincode (serialization), pyo3 (Python interop, optional mpc feature), clap (CLI), tracing (logging), anyhow/thiserror (errors), typeshare (TS bindings)

Web UI: React 18, Vite, Tailwind, Radix UI, TanStack Query, Jotai, Monaco Editor, Dockview
