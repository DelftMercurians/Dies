# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dies is the RoboCup AI framework for the Delft Mercurians team, written primarily in Rust with strategy scripting in Rhai. The framework provides a complete robotics control system including physics simulation, vision processing, game state management, and real-time robot control.

## Development Commands

### Building and Running
- **Build the project**: `cargo build`
- **Run Dies locally**: `cargo run -- <options>`
- **Run with MPC**: `just run` (installs MPC dependencies and runs)

### Testing
- **Run MPC tests**: `just mpc` or `cd mpc_jax && uv run pytest -vv -s`
- **Run Rust tests**: `cargo test`

### Web UI Development
- **Build web UI**: `cargo make webui` (requires cargo-make: `cargo install cargo-make`)
- **Install web dependencies**: `cd webui && npm install`

### Debugging
- **Debug hanging issues**: Install `cargo install --locked tokio-console`, then run `tokio-console` while Dies is running

## Architecture Overview

### Core Crate Structure
The system is split into focused crates:

- **dies-core**: Fundamental types and traits (Vector2, WorldData, PlayerCmd, etc.)
- **dies-executor**: The main execution engine containing TeamController, PlayerController, behavior trees, and strategy execution
- **dies-world**: World state tracking, filtering, and ball/player data processing
- **dies-ssl-client**: SSL Vision protocol client for field data
- **dies-basestation-client**: Robot communication and command transmission
- **dies-simulator**: Physics-based simulation for testing and development
- **dies-webui**: Web-based monitoring and control interface
- **dies-protos**: Protocol buffer definitions for SSL and game controller communication

### Behavior Tree System
The AI uses a sophisticated behavior tree system implemented in `dies-executor/src/behavior_tree/`:

- **BT Nodes**: Select, Sequence, Guard, Semaphore, ScoringSelect, Action nodes
- **Skills**: Atomic robot actions (GoToPosition, FetchBall, Kick, etc.) in `dies-executor/src/skills/`
- **Rhai Integration**: Strategy scripts written in Rhai language for high-level decision making

### Strategy Architecture
Strategies are located in `strategies/` and written in Rhai:

- **Role-based design**: Modern approach using dynamic role assignment (striker, waller, harasser, support)
- **Entry point**: `strategies/main.rhai` with `main()` function returning role assignment configuration
- **Modular structure**: Shared utilities in `strategies/shared/`, game mode specific logic

### Control System
Located in `dies-executor/src/control/`:

- **MPC**: Model Predictive Control for robot motion planning
- **MTP**: Motion-to-Position controllers
- **RVO**: Reactive Velocity Obstacles for collision avoidance
- **Player/Team Controllers**: Coordinate multiple robots and execute strategies

## Key Implementation Details

### World State Management
- **WorldTracker** (`dies-world`) processes SSL vision data and maintains filtered world state
- **Kalman filtering** for ball and robot position/velocity estimation
- **Team-agnostic operation**: Supports controlling 0, 1, or 2 teams simultaneously

### Communication Architecture
- **SSL Vision**: Receives field geometry and detection data via multicast UDP
- **Game Controller**: Processes referee commands and game state changes  
- **Basestation**: Sends movement commands to physical robots via radio/ethernet

### Hot Reloading
- Rhai scripts support hot reloading for rapid strategy development
- Web UI provides real-time monitoring and script editing capabilities

## Important File Locations

- **Main executable**: `crates/dies-cli/src/main.rs`
- **Executor core**: `crates/dies-executor/src/lib.rs`
- **Strategy scripts**: `strategies/main.rhai` (entry point)
- **Rhai scripting guide**: `.cursor/rules/rhai-scripting.mdc` (comprehensive strategy writing documentation)
- **Settings**: `dies-settings.json`
- **Build config**: `Cargo.toml` (workspace), `Makefile.toml` (cargo-make), `justfile`

## Strategy Development

When working with Rhai strategies:

1. **Follow the role-based pattern**: Use `AssignRoles()` with role builders for modern strategy design
2. **Keep behavior trees shallow**: Maximum 3-4 layers deep for maintainability
3. **Use scoring functions**: Implement proper scoring for role assignment and ScoringSelect nodes
4. **Leverage the situation API**: Rich `RobotSituation` object provides world queries and geometry calculations
5. **Reference the comprehensive guide**: See `.cursor/rules/rhai-scripting.mdc` for complete API reference

## Development Workflow

1. **Initial setup**: Ensure Rust toolchain installed, run `cargo build` to verify
2. **Strategy development**: Edit scripts in `strategies/`, use hot reloading via web UI
3. **Testing**: Use built-in simulator (`cargo run -- --simulator`) or connect to SSL Vision
4. **Web monitoring**: Access web UI for real-time debugging and visualization
5. **MPC development**: Use `just mpc` for Python-based motion planning tests

The codebase emphasizes clean separation between Rust (performance-critical systems) and Rhai (strategy logic), enabling rapid development of complex multi-robot behaviors.