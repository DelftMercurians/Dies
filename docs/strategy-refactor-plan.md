# Strategy System Refactor - Architecture Plan

This document describes the target architecture for the refactored strategy system. The goal is to decouple strategy logic from the core executor, enabling rapid iteration on strategies without recompiling the main framework.

## Goals

1. **Isolation**: Strategy code runs in a separate process, fully isolated from the executor
2. **Hot Reload**: Strategies can be reloaded without restarting the framework
3. **Multi-Strategy**: Support running different strategies for each team (A/B testing, training)
4. **Skill-Based Control**: Strategies control players through a fixed skill API, no direct commands
5. **Rich World API**: Strategies have comprehensive read-only access to world state
6. **Debug Integration**: Full compatibility with existing debug visualization system
7. **Clean Architecture**: Team controller becomes strategy-agnostic, focused on compliance and control
8. **Team-Agnostic Strategies**: Strategies never know their team color or side assignment

## Non-Goals

- Custom skill definitions from strategies
- Per-robot strategy assignment (always whole-team)
- State persistence across hot reloads
- Backwards compatibility with Rhai scripts (clean break)
- Framework-provided team coordination primitives (strategies handle their own coordination)

---

## Coordinate System & Team Abstraction

A core design principle: **strategies never know their team color or side assignment**. All world data is normalized by the framework before being sent to strategies.

### Normalized Coordinate System

All coordinates sent to strategies use a team-relative reference frame:

- **+x axis**: Always points toward the **opponent's goal** (attacking direction)
- **-x axis**: Always points toward **our own goal** (defending direction)
- **y axis**: Left/right from the team's perspective (unchanged)

This means:

- Our goal is always at negative x
- Opponent's goal is always at positive x
- Strategy code never needs to flip coordinates based on which side we're playing

### Player Categorization

Players are categorized as:

- `own_players`: Our team's robots
- `opp_players`: Opponent's robots

Strategies never see `TeamColor::Blue` or `TeamColor::Yellow` - only "own" and "opponent."

### Framework Responsibilities

The framework (StrategyHost) handles:

1. Transforming raw world coordinates to team-relative coordinates
2. Categorizing players as own/opponent based on team assignment
3. Transforming skill commands back to absolute coordinates for execution
4. All side-switching logic when teams change sides at halftime

### Why This Matters

- Strategy code is completely reusable regardless of team color
- No conditional logic based on "am I blue or yellow?"
- Reduces bugs from coordinate sign errors
- Strategies always think "attack toward +x, defend toward -x"

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              dies-executor                                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                           StrategyHost                                  │  │
│  │  ┌─────────────────────┐    ┌─────────────────────┐                    │  │
│  │  │  StrategyConnection │    │  StrategyConnection │   (one per team)   │  │
│  │  │    (Team 0)         │    │    (Team 1)         │                    │  │
│  │  │  + coord transform  │    │  + coord transform  │                    │  │
│  │  └──────────┬──────────┘    └──────────┬──────────┘                    │  │
│  │             │                          │                                │  │
│  │             │ Unix Socket              │ Unix Socket                    │  │
│  │             ▼                          ▼                                │  │
│  └─────────────┼──────────────────────────┼────────────────────────────────┘  │
│                │                          │                                   │
│  ┌─────────────▼──────────────────────────▼────────────────────────────────┐  │
│  │                         TeamController                                   │  │
│  │   • Receives SkillCommands from StrategyHost                            │  │
│  │   • Applies rule compliance                                              │  │
│  │   • Manages PlayerControllers                                            │  │
│  │   • MPC / motion planning                                                │  │
│  │   • NO strategy logic, NO role assignment, NO behavior trees            │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
         │                                    │
         │ Unix Socket                        │ Unix Socket
         │ (normalized coords)                │ (normalized coords)
         ▼                                    ▼
┌─────────────────────────┐        ┌─────────────────────────┐
│   Strategy Process A    │        │   Strategy Process B    │
│   (e.g., v0-strategy)   │        │   (e.g., test-strat)    │
│                         │        │                         │
│  ┌───────────────────┐  │        │  ┌───────────────────┐  │
│  │  StrategyRunner   │  │        │  │  StrategyRunner   │  │
│  │  (IPC, lifecycle) │  │        │  │  (IPC, lifecycle) │  │
│  └─────────┬─────────┘  │        │  └─────────┬─────────┘  │
│            │            │        │            │            │
│  ┌─────────▼─────────┐  │        │  ┌─────────▼─────────┐  │
│  │   Strategy API    │  │        │  │   Strategy API    │  │
│  │  (world, players) │  │        │  │  (world, players) │  │
│  │  NO TEAM COLOR    │  │        │  │  NO TEAM COLOR    │  │
│  └─────────┬─────────┘  │        │  └─────────┬─────────┘  │
│            │            │        │            │            │
│  ┌─────────▼─────────┐  │        │  ┌─────────▼─────────┐  │
│  │  User Strategy    │  │        │  │  User Strategy    │  │
│  │  Implementation   │  │        │  │  Implementation   │  │
│  └───────────────────┘  │        │  └───────────────────┘  │
└─────────────────────────┘        └─────────────────────────┘
```

---

## Component Details

### 1. StrategyHost (in `dies-executor`)

The StrategyHost manages strategy processes and acts as the bridge between strategies and the team controller.

**Responsibilities:**

- Discover available strategy binaries on startup
- Spawn strategy processes on demand
- Manage Unix domain socket connections to each running strategy
- Send world state snapshots to strategy processes each frame
- Receive skill commands from strategy processes
- Translate skill commands to `PlayerControlInput` for team controller
- Forward debug data from strategies to the debug system
- Handle strategy process crashes and restarts
- Support hot-reload on file change (kill + respawn)

**Configuration:**

- Which strategy binary to use for each team
- Whether each team is active
- Strategy-specific configuration (passed to strategy on init)

**State:**

- Map of team index (0/1) → StrategyConnection
- Available strategy binary paths
- Coordinate transformation state per connection (for normalizing world data)

### 2. StrategyConnection (in `dies-executor`)

Manages IPC with a single strategy process.

**Responsibilities:**

- Spawn the strategy process
- Establish Unix domain socket connection
- Send/receive messages with binary serialization
- Monitor process health
- Handle reconnection after crash/reload
- Buffer messages if strategy is temporarily unavailable

### 3. StrategyRunner (new crate: `dies-strategy-runner`)

The process entrypoint and IPC client for strategy binaries.

**Responsibilities:**

- Parse command-line arguments (socket path, config)
- Establish connection to host
- Run the main update loop
- Provide the Strategy API to user code
- Capture and forward logs
- Serialize outgoing skill commands
- Deserialize incoming world state (already in normalized coordinates)
- Handle graceful shutdown signals

**Note:** The runner does NOT know the team color. All coordinates are pre-normalized by the host.

**Main Loop:**

```
loop:
    1. Receive WorldSnapshot from host (normalized coordinates)
    2. Update Strategy API state (World, PlayerHandles)
    3. Call user strategy's update function
    4. Collect skill commands from PlayerHandles
    5. Collect debug data
    6. Send StrategyOutput to host (coordinates in team-relative frame)
```

### 4. Strategy API (new crate: `dies-strategy-api`)

The public API that strategy implementations link against. Designed for ergonomics and safety.

**Key Design Principles:**

- All coordinates are normalized (see Coordinate System section above)
- No team color or side assignment exposed to strategies
- Skill APIs are tailored per-skill for optimal ergonomics
- Only high-level skill status visible (no internal skill state)
- Team-level coordination is entirely the strategy's responsibility

#### 4.1 World (Read-Only State)

Read-only access to the current world state. All coordinates are in the normalized team-relative frame.

```rust
pub struct World {
    // Field geometry
    pub fn field(&self) -> &FieldGeometry;
    pub fn field_length(&self) -> f64;
    pub fn field_width(&self) -> f64;
    pub fn own_goal_center(&self) -> Vector2;      // Always at -x
    pub fn opp_goal_center(&self) -> Vector2;      // Always at +x
    pub fn own_penalty_area(&self) -> Rect;
    pub fn opp_penalty_area(&self) -> Rect;

    // Ball
    pub fn ball(&self) -> Option<&BallState>;
    pub fn ball_position(&self) -> Option<Vector2>;
    pub fn ball_velocity(&self) -> Option<Vector2>;

    // Players (already categorized as own/opponent)
    pub fn own_players(&self) -> &[PlayerState];
    pub fn opp_players(&self) -> &[PlayerState];
    pub fn own_player(&self, id: PlayerId) -> Option<&PlayerState>;

    // Game state
    pub fn game_state(&self) -> GameState;
    pub fn is_ball_in_play(&self) -> bool;
    pub fn us_operating(&self) -> bool;
    pub fn our_keeper_id(&self) -> Option<PlayerId>;

    // Timing
    pub fn dt(&self) -> f64;
    pub fn timestamp(&self) -> f64;

    // Utilities
    pub fn predict_ball_position(&self, t: f64) -> Option<Vector2>;
}

pub struct BallState {
    pub position: Vector2,
    pub velocity: Vector2,
    pub detected: bool,
}

pub struct PlayerState {
    pub id: PlayerId,
    pub position: Vector2,
    pub velocity: Vector2,
    pub heading: Angle,
    pub angular_velocity: f64,
    pub has_ball: bool,  // breakbeam sensor
    pub handicaps: HashSet<Handicap>,
}
```

#### 4.2 PlayerHandle (Per-Player Control)

Per-player control interface. Strategies access handles for their own players only.

The skill API uses a **hybrid approach** tailored to each skill's nature:

- **Continuous skills** (GoToPos, Dribble): Simple methods with implicit update semantics
- **Discrete skills** (PickupBall, ReflexShoot): Return handles for explicit lifecycle control

```rust
pub struct PlayerHandle {
    // === Read-only state ===

    pub fn id(&self) -> PlayerId;
    pub fn position(&self) -> Vector2;
    pub fn velocity(&self) -> Vector2;
    pub fn heading(&self) -> Angle;
    pub fn has_ball(&self) -> bool;

    /// Current skill execution status
    pub fn skill_status(&self) -> SkillStatus;

    // === Continuous Skills (implicit update) ===
    // Call each frame - if already running same skill type, parameters are updated smoothly

    /// Move to position. Calling repeatedly with changing positions creates smooth motion.
    pub fn go_to(&mut self, position: Vector2) -> GoToBuilder<'_>;

    /// Dribble with ball to position. Updates smoothly if already dribbling.
    pub fn dribble_to(&mut self, position: Vector2, heading: Angle) -> DribbleBuilder<'_>;

    // === Discrete Skills (explicit handle) ===
    // Return a handle for monitoring; typically "fire and wait for completion"

    /// Start approaching and capturing the ball.
    /// Returns handle to monitor progress.
    pub fn pickup_ball(&mut self, target_heading: Angle) -> SkillHandle<PickupBall>;

    /// Start a reflex shot toward target position.
    /// Robot orients and kicks when ready. Returns handle to monitor.
    pub fn reflex_shoot(&mut self, target: Vector2) -> SkillHandle<ReflexShoot>;

    // === Control ===

    /// Stop all motion immediately.
    pub fn stop(&mut self);

    // === Metadata ===

    /// Set role name for debugging/visualization in UI.
    pub fn set_role(&mut self, role: &str);
}
```

#### 4.3 Skill Status

Only high-level status is exposed - no visibility into internal skill state.

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkillStatus {
    /// No skill has been commanded yet
    Idle,

    /// Skill is currently executing
    Running,

    /// Skill completed successfully, robot is stopped
    Succeeded,

    /// Skill failed, robot is stopped
    Failed,
}
```

**Lifecycle:**

- Status persists as `Succeeded` or `Failed` until a new skill command is issued
- When a skill completes, the robot physically stops
- Issuing the same skill type after completion starts a **new** skill instance

#### 4.4 Skill Builders and Handles

**GoToBuilder** - for continuous position tracking:

```rust
pub struct GoToBuilder<'a> { /* ... */ }

impl<'a> GoToBuilder<'a> {
    /// Set target heading (defaults to current heading if not set)
    pub fn with_heading(self, heading: Angle) -> Self;

    /// Compute heading to face the given position
    pub fn facing(self, position: Vector2) -> Self;
}

// Commits on drop - just call player.go_to(pos).with_heading(angle);
impl Drop for GoToBuilder<'_> { /* commits command */ }
```

**DribbleBuilder** - for dribbling with ball:

```rust
pub struct DribbleBuilder<'a> { /* ... */ }

impl<'a> DribbleBuilder<'a> {
    // Dribble parameters are set in the initial call
    // Additional builder methods can be added as needed
}
```

**SkillHandle** - for discrete skills with explicit lifecycle:

```rust
/// Handle to a running discrete skill.
/// Dropping the handle does NOT cancel the skill.
pub struct SkillHandle<S> {
    // ...
}

impl<S: SkillParams> SkillHandle<S> {
    /// Update the skill parameters while it's running
    pub fn update(&mut self, params: S);

    /// Update parameters using a closure
    pub fn update_with(&mut self, f: impl FnOnce(&mut S));
}

// Parameter structs for discrete skills
pub struct PickupBallParams {
    pub target_heading: Angle,
}

pub struct ReflexShootParams {
    pub target: Vector2,
}
```

#### 4.5 TeamContext

Access to world state and all own player handles.

```rust
pub struct TeamContext {
    /// Read-only access to world state
    pub fn world(&self) -> &World;

    /// Iterate over all own player handles
    pub fn players(&mut self) -> impl Iterator<Item = &mut PlayerHandle>;

    /// Get a specific player handle by ID
    pub fn player(&mut self, id: PlayerId) -> Option<&mut PlayerHandle>;

    /// Get list of own player IDs
    pub fn player_ids(&self) -> &[PlayerId];
}
```

**Note:** No `team_color()` method - strategies don't know their team color.

#### 4.6 Debug API

Functions for visualization and debugging, compatible with existing UI.

```rust
pub mod debug {
    /// Draw a cross marker
    pub fn cross(key: &str, position: Vector2);
    pub fn cross_colored(key: &str, position: Vector2, color: DebugColor);

    /// Draw a line
    pub fn line(key: &str, start: Vector2, end: Vector2);
    pub fn line_colored(key: &str, start: Vector2, end: Vector2, color: DebugColor);

    /// Draw a circle
    pub fn circle(key: &str, center: Vector2, radius: f64);
    pub fn circle_filled(key: &str, center: Vector2, radius: f64, color: DebugColor);

    /// Record a numeric value (for plotting)
    pub fn value(key: &str, value: f64);

    /// Record a string value
    pub fn string(key: &str, value: &str);

    /// Remove a debug entry
    pub fn remove(key: &str);

    /// Clear all debug entries for this strategy
    pub fn clear();
}
```

Debug data is collected during the update and sent to the host with skill commands.
Coordinates are automatically transformed back to world frame for display.

#### 4.7 Strategy Trait

The interface that user strategies implement.

```rust
pub trait Strategy: Send + 'static {
    /// Called once when the strategy is loaded
    fn init(&mut self, world: &World) {}

    /// Called each frame to update player skills
    fn update(&mut self, ctx: &mut TeamContext);

    /// Optional: called when the strategy is about to be unloaded
    fn shutdown(&mut self) {}
}
```

#### 4.8 Usage Examples

##### Example 1: Goalkeeper - Continuous Tracking

```rust
impl Strategy for GoalkeeperStrategy {
    fn update(&mut self, ctx: &mut TeamContext) {
        let world = ctx.world();
        let keeper = ctx.player(self.keeper_id).unwrap();

        // Compute block position every frame (dynamic)
        let ball_pos = world.ball_position().unwrap_or_default();
        let goal = world.own_goal_center();
        let block_pos = self.compute_block_position(ball_pos, goal);

        // Call go_to each frame - parameters update smoothly
        keeper.go_to(block_pos).facing(ball_pos);
        keeper.set_role("Goalkeeper");
    }
}
```

##### Example 2: Striker - State Machine with Discrete Skills

```rust
enum StrikerState {
    Idle,
    FetchingBall { handle: SkillHandle<PickupBall> },
    Shooting { handle: SkillHandle<ReflexShoot> },
}

impl Strategy for StrikerStrategy {
    fn update(&mut self, ctx: &mut TeamContext) {
        let world = ctx.world();
        let striker = ctx.player(self.striker_id).unwrap();

        match &mut self.state {
            StrikerState::Idle => {
                // Start fetching ball, aiming toward goal
                let goal = world.opp_goal_center();
                let ball = world.ball_position().unwrap();
                let heading = Angle::between_points(ball, goal);

                let handle = striker.pickup_ball(heading);
                striker.set_role("Striker (fetching)");
                self.state = StrikerState::FetchingBall { handle };
            }

            StrikerState::FetchingBall { handle } => {
                // Update heading while approaching (ball might move)
                let goal = world.opp_goal_center();
                let ball = world.ball_position().unwrap();
                handle.update_with(|p| {
                    p.target_heading = Angle::between_points(ball, goal);
                });

                match striker.skill_status() {
                    SkillStatus::Succeeded => {
                        // Got ball, start shooting
                        let handle = striker.reflex_shoot(goal);
                        striker.set_role("Striker (shooting)");
                        self.state = StrikerState::Shooting { handle };
                    }
                    SkillStatus::Failed => {
                        self.state = StrikerState::Idle;
                    }
                    _ => {}
                }
            }

            StrikerState::Shooting { handle } => {
                match striker.skill_status() {
                    SkillStatus::Succeeded | SkillStatus::Failed => {
                        self.state = StrikerState::Idle;
                    }
                    _ => {}
                }
            }
        }
    }
}
```

##### Example 3: Multi-Robot Coordination (Strategy-Managed)

```rust
impl Strategy for TeamStrategy {
    fn update(&mut self, ctx: &mut TeamContext) {
        let world = ctx.world();

        // Strategy manages coordination - no framework support
        let ball_pos = world.ball_position().unwrap();

        // Find closest player to ball
        let closest_id = world.own_players()
            .iter()
            .min_by_key(|p| OrderedFloat((p.position - ball_pos).norm()))
            .map(|p| p.id);

        for player in ctx.players() {
            if Some(player.id()) == closest_id {
                // This player fetches ball
                player.pickup_ball(Angle::from_radians(0.0));
                player.set_role("Ball Handler");
            } else {
                // Others position for pass
                let support_pos = self.compute_support_position(player.id(), world);
                player.go_to(support_pos).facing(ball_pos);
                player.set_role("Support");
            }
        }
    }
}
```

---

## Skill System

### Design Philosophy

The skill system is intentionally **streamlined** with a small set of well-defined skills. Each skill has a clear purpose and completion condition. Skills are designed to be:

1. **Composable**: Complex behaviors are built by sequencing skills in strategy code
2. **Updatable**: Parameters can be changed while a skill is running for smooth motion
3. **Observable**: Strategies see high-level status (Running/Succeeded/Failed) but not internal state

### Available Skills

| Skill         | Parameters                     | API Style  | Description                                 | Completion                |
| ------------- | ------------------------------ | ---------- | ------------------------------------------- | ------------------------- |
| `GoToPos`     | `position`, `heading`          | Continuous | Move to position, face heading              | Arrives at position       |
| `Dribble`     | `target_pos`, `target_heading` | Continuous | Move with ball (limited accel, dribbler on) | Arrives at position       |
| `PickupBall`  | `target_heading`               | Discrete   | Approach ball from behind, capture it       | Ball captured (breakbeam) |
| `ReflexShoot` | `target_pos`                   | Discrete   | Orient toward target and kick               | Ball kicked               |
| `Stop`        | -                              | Immediate  | Stop all motion                             | Immediate                 |

### Skill Details

#### GoToPos (Continuous)

Move to a target position with optional heading control.

```rust
// Usage: call each frame, parameters update smoothly
player.go_to(position).with_heading(angle);
player.go_to(position).facing(ball_position);
```

**Behavior:**

- Pathfinding and obstacle avoidance handled by motion controller
- Smooth trajectory updates when target changes
- Completes when within position tolerance and velocity near zero

**Completion:** `Succeeded` when robot arrives at position

#### Dribble (Continuous)

Move to a target position while carrying the ball.

```rust
// Usage: call each frame
player.dribble_to(position, target_heading);
```

**Behavior:**

- Dribbler activated
- Limited acceleration to avoid losing ball
- Can rotate to align with target heading while moving
- Fails immediately if robot doesn't have ball (breakbeam not triggered)

**Completion:**

- `Succeeded` when robot arrives at position
- `Failed` if ball is lost during dribble

#### PickupBall (Discrete)

Approach and capture the ball, pre-orienting for a subsequent action.

```rust
// Usage: start once, monitor handle
let handle = player.pickup_ball(target_heading);

// Can update heading while approaching
handle.update_with(|p| p.target_heading = new_heading);
```

**Behavior:**

1. Move to a position behind the ball, facing `target_heading`
2. Slowly approach ball until breakbeam triggers
3. Dribbler activated throughout

**Completion:**

- `Succeeded` when breakbeam detects ball
- `Failed` if ball moves away or timeout

#### ReflexShoot (Discrete)

Orient toward a target and kick. Designed for quick shots/passes.

```rust
// Usage: start once, wait for completion
let handle = player.reflex_shoot(goal_center);
```

**Behavior:**

1. Rotate to face target position
2. Prepare kicker (arm capacitor)
3. Kick when aligned and ready

**Completion:**

- `Succeeded` when ball is kicked
- `Failed` if angle is geometrically impossible or ball lost

#### Stop (Immediate)

Stop all motion immediately.

```rust
player.stop();
```

**Behavior:** Sets velocity to zero, disables dribbler
**Completion:** Immediate, transitions to `Idle`

### Skill Command Protocol (IPC)

```rust
/// Skill command sent from strategy to executor
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SkillCommand {
    /// Move to position (continuous, updateable)
    GoToPos {
        position: Vector2,
        heading: Option<Angle>,
    },

    /// Dribble with ball to position (continuous, updateable)
    Dribble {
        target_pos: Vector2,
        target_heading: Angle,
    },

    /// Approach and capture ball (discrete)
    PickupBall {
        target_heading: Angle,
    },

    /// Orient and kick toward target (discrete)
    ReflexShoot {
        target: Vector2,
    },

    /// Stop all motion
    Stop,
}
```

### Skill Update Semantics

The executor compares incoming commands to the currently running skill:

| Incoming Command     | Current Skill    | Action                                      |
| -------------------- | ---------------- | ------------------------------------------- |
| None (no command)    | Any              | Continue current skill with last parameters |
| Same skill type      | Running          | **Update parameters** on existing skill     |
| Different skill type | Running          | **Interrupt** current, start new skill      |
| Any command          | Succeeded/Failed | Start **new** skill instance                |
| Stop                 | Any              | Interrupt current, robot stops              |

This allows:

- Smooth trajectory updates by calling `go_to()` each frame with new positions
- Fire-and-forget discrete skills that run to completion
- Clean transitions between different skill types

### Skill Execution

Skills are executed in the executor, not in the strategy process. This ensures:

- Low latency control (skills run at full frame rate)
- Access to full world state including opponent predictions
- Consistent behavior regardless of strategy implementation
- Strategies remain responsive even during complex skill execution

The strategy process only sees skill status updates (Idle/Running/Succeeded/Failed).

---

## IPC Protocol

### Transport

- Unix domain sockets
- One socket per strategy connection
- Socket path pattern: `/tmp/dies-strategy-{pid}-{instance}.sock`

Note: Socket path does not include team color - strategies don't know their team.

### Serialization

- Binary format for performance (bincode or MessagePack)
- All message types derive `Serialize`/`Deserialize`

### Messages

#### Host → Strategy

```rust
#[derive(Serialize, Deserialize)]
pub enum HostMessage {
    /// Initial configuration when connecting
    /// Note: No team color sent - strategy doesn't need to know
    Init {
        config: StrategyConfig,
    },

    /// World state update (sent every frame)
    /// All coordinates are pre-transformed to team-relative frame
    WorldUpdate {
        world: WorldSnapshot,
        skill_statuses: HashMap<PlayerId, SkillStatus>,
    },

    /// Request graceful shutdown
    Shutdown,
}

#[derive(Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub timestamp: f64,
    pub dt: f64,
    pub field_geom: FieldGeometry,
    pub ball: Option<BallState>,
    pub own_players: Vec<PlayerState>,   // Our team
    pub opp_players: Vec<PlayerState>,   // Opponent team
    pub game_state: GameState,
    pub us_operating: bool,              // True if it's our turn (free kick, etc.)
    pub our_keeper_id: Option<PlayerId>,
}

#[derive(Serialize, Deserialize)]
pub enum SkillStatus {
    Idle,
    Running,
    Succeeded,
    Failed,
}
```

#### Strategy → Host

```rust
#[derive(Serialize, Deserialize)]
pub enum StrategyMessage {
    /// Ready to receive world updates
    Ready,

    /// Response to world update
    Output {
        /// Skill commands for each player
        /// None = continue previous skill with previous parameters
        skill_commands: HashMap<PlayerId, Option<SkillCommand>>,
        /// Debug visualization data
        debug_data: Vec<DebugEntry>,
        /// Role names for UI display
        player_roles: HashMap<PlayerId, String>,
    },

    /// Log message to forward to host logging system
    Log {
        level: LogLevel,
        message: String,
    },
}

/// Skill command - see Skill System section for details
#[derive(Serialize, Deserialize)]
pub enum SkillCommand {
    GoToPos {
        position: Vector2,
        heading: Option<Angle>,
    },
    Dribble {
        target_pos: Vector2,
        target_heading: Angle,
    },
    PickupBall {
        target_heading: Angle,
    },
    ReflexShoot {
        target: Vector2,
    },
    Stop,
}

#[derive(Serialize, Deserialize)]
pub struct DebugEntry {
    pub key: String,
    pub value: DebugValue,
}
```

### Protocol Flow

```
   Host                              Strategy
     │                                   │
     │──── Init ────────────────────────>│  (config only, no team color)
     │                                   │
     │<───────────────────────── Ready ──│
     │                                   │
     │                                   │
   ┌─┼───────────────────────────────────┼─┐
   │ │                                   │ │  Frame Loop
   │ │──── WorldUpdate ─────────────────>│ │  (normalized coords)
   │ │                                   │ │
   │ │<──────────────────────── Output ──│ │  (skill commands in team frame)
   │ │                                   │ │
   │ │<─────────────────────────── Log ──│ │  (optional, async)
   │ │                                   │ │
   └─┼───────────────────────────────────┼─┘
     │                                   │
     │──── Shutdown ────────────────────>│
     │                                   │
     │                           (process exits)
```

### Coordinate Transformation

The host handles all coordinate transformations:

1. **Inbound (to strategy):** World coordinates → Team-relative coordinates
2. **Outbound (from strategy):** Team-relative coordinates → World coordinates

This is transparent to the strategy - it always works in the normalized frame where +x points toward the opponent's goal.

---

## TeamController Changes

After refactoring, TeamController is simplified significantly:

**Removed:**

- Role assignment logic
- Behavior tree storage and execution
- Strategy function pointer
- RoleAssignmentSolver
- BtContext (semaphores, passing target tracking)
- All coordination logic (now strategy's responsibility)

**Retained:**

- PlayerController management
- Rule compliance (`comply()` function)
- MPC control
- Goal area avoidance
- Yellow card handling (robot removal)
- Manual override handling

**New:**

- Receives `HashMap<PlayerId, Option<SkillCommand>>` from StrategyHost
- Skill execution engine with update semantics
- Skill parameter updates without recreating skill instances
- Reports skill status back to StrategyHost

### Skill Execution in TeamController

Each player has an active skill (or none). The controller handles skill lifecycle:

```
For each player:
    1. Receive skill command (or None) from strategy
    2. Compare to current running skill:
       - None: Continue current skill unchanged
       - Same type: Call skill.update_params(command)
       - Different type: Stop current, create new skill instance
       - After completion: Any command starts new instance
    3. Tick the skill → get PlayerControlInput
    4. Apply compliance transformations (game state rules)
    5. Feed to PlayerController (motion control)
    6. Update skill status (Idle/Running/Succeeded/Failed)
    7. Report status to StrategyHost
```

### Skill Implementation Interface

Skills in the executor implement:

```rust
pub trait ExecutableSkill {
    /// Update parameters while skill is running (same skill type)
    fn update_params(&mut self, command: &SkillCommand);

    /// Execute one tick, returning control input or completion status
    fn tick(&mut self, ctx: SkillContext) -> SkillProgress;

    /// Check if skill has completed
    fn status(&self) -> SkillStatus;
}

pub enum SkillProgress {
    /// Skill continues, apply this control input
    Continue(PlayerControlInput),
    /// Skill completed with given result
    Done(SkillResult),
}

pub enum SkillResult {
    Success,
    Failure,
}
```

### Coordinate Untransformation

Commands from strategy are in team-relative coordinates. The skill executor:

1. Receives commands in team-relative frame
2. Skills operate in team-relative frame internally
3. Output `PlayerControlInput` is untransformed to world frame before PlayerController

---

## Repository Structure

```
dies/
├── crates/
│   ├── dies-core/                    # Shared types (unchanged)
│   ├── dies-executor/                # Main executor
│   │   └── src/
│   │       ├── control/
│   │       │   ├── team_controller.rs    # Simplified
│   │       │   ├── player_controller.rs
│   │       │   └── skill_executor.rs     # NEW: skill execution
│   │       ├── strategy_host/            # NEW
│   │       │   ├── mod.rs
│   │       │   ├── host.rs               # StrategyHost
│   │       │   ├── connection.rs         # StrategyConnection
│   │       │   └── protocol.rs           # IPC message types
│   │       └── ...
│   │
│   ├── dies-strategy-api/            # NEW: Strategy API crate
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── world.rs              # WorldQuery API
│   │       ├── player.rs             # PlayerHandle
│   │       ├── team.rs               # TeamContext
│   │       ├── skill.rs              # Skill enum and types
│   │       ├── debug.rs              # Debug API
│   │       └── strategy.rs           # Strategy trait
│   │
│   ├── dies-strategy-runner/         # NEW: Runner binary template
│   │   └── src/
│   │       ├── main.rs               # Generated entrypoint
│   │       ├── runner.rs             # Main loop logic
│   │       ├── ipc.rs                # IPC client
│   │       └── context.rs            # TeamContext implementation
│   │
│   └── ... (other existing crates)
│
├── strategies/                       # NEW: Strategy crates folder
│   ├── v0-strategy/                  # Ported behavior tree strategy
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                # Strategy implementation
│   │       ├── roles/                # Role logic
│   │       │   ├── goalkeeper.rs
│   │       │   ├── striker.rs
│   │       │   └── ...
│   │       ├── behavior_tree/        # BT implementation (moved)
│   │       └── role_assignment.rs    # Role assignment logic (moved)
│   │
│   ├── test-strategy/                # Simple test strategy
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   │
│   └── [template]/                   # Template for new strategies
│       ├── Cargo.toml.template
│       └── src/lib.rs.template
│
└── target/
    └── strategies/                   # Built strategy binaries
        ├── v0-strategy
        ├── test-strategy
        └── ...
```

### Strategy Crate Template

Each strategy crate has a minimal structure:

**Cargo.toml:**

```toml
[package]
name = "my-strategy"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]  # or rlib, depending on build approach

[dependencies]
dies-strategy-api = { path = "../../crates/dies-strategy-api" }
```

**src/lib.rs:**

```rust
use dies_strategy_api::prelude::*;

pub struct MyStrategy {
    // strategy state
}

impl Strategy for MyStrategy {
    fn update(&mut self, ctx: &mut TeamContext) {
        // Strategy logic here
    }
}

// Required: export factory function
dies_strategy_api::export_strategy!(MyStrategy::new);
```

### Build System

A build script or cargo workspace configuration generates runner binaries:

```
For each strategy crate in strategies/:
    1. Compile strategy crate as library
    2. Link with dies-strategy-runner
    3. Output binary to target/strategies/{strategy-name}
```

This can be done via:

- Cargo workspace with a build.rs that generates runner mains
- A custom build tool/script
- Cargo features to select which strategy to build

---

## Process Lifecycle

### Startup

1. Executor starts
2. StrategyHost initializes, scans `target/strategies/` for available binaries
3. UI can display list of available strategies
4. User selects strategy for each active team
5. StrategyHost spawns selected strategy processes
6. Strategy processes connect via Unix socket
7. Host sends `Init` message with config (no team color - strategy doesn't need it)
8. Strategy responds with `Ready`
9. Frame loop begins

### Runtime

1. Each frame, host sends `WorldUpdate` to active strategies
2. Strategies process and respond with `Output`
3. Host translates skill commands to PlayerControlInputs
4. TeamController executes skills and applies compliance

### Hot Reload

When strategy binary changes (detected via file watcher):

1. Host sends `Shutdown` to strategy process
2. Wait for graceful exit (or kill after timeout)
3. Respawn with new binary
4. Re-establish connection
5. Strategy restarts from fresh state

### Crash Recovery

If strategy process crashes:

1. Host detects connection loss / process exit
2. Log error
3. Respawn strategy process
4. Strategy restarts from fresh state
5. Brief pause in commands (robots hold position via compliance)

### Shutdown

1. Executor shutdown initiated
2. Host sends `Shutdown` to all strategy processes
3. Strategies clean up and exit
4. Host waits for exits or kills after timeout

---

## Strategy Discovery and Selection

### Discovery

On startup and periodically, StrategyHost scans for strategy binaries:

- Look in `target/strategies/`
- Each binary is registered by name
- Optionally read metadata from sidecar file or binary

### Selection UI

The web UI allows:

- Viewing list of available strategies
- Selecting strategy for blue team
- Selecting strategy for yellow team
- Starting/stopping strategy processes
- Triggering hot reload manually

### Configuration

Strategy selection is stored in `ExecutorSettings`:

```rust
pub struct TeamConfiguration {
    pub blue_active: bool,
    pub yellow_active: bool,
    pub blue_strategy: Option<String>,   // Strategy binary name
    pub yellow_strategy: Option<String>,
    pub side_assignment: SideAssignment,
}
```

---

## Porting the v0 Strategy

The current behavior tree strategy in `dies-strategy/src/v0/` will be ported:

### Components to Move

1. **Behavior Tree Core** (`behavior_tree/`)

   - `BehaviorTree`, `BehaviorNode`, `BehaviorStatus`
   - All node types (Select, Sequence, Guard, etc.)
   - Can be simplified since skills now have update semantics

2. **Role Assignment** (`role_assignment.rs`)

   - `Role`, `RoleBuilder`, `RoleAssignmentProblem`
   - `RoleAssignmentSolver`

3. **Coordination Logic** (previously in `BtContext`)

   - Semaphores for exclusive access (e.g., "only one striker")
   - Passing target tracking between players
   - **Note:** This is now entirely the strategy's responsibility

4. **Role Implementations** (`v0/`)
   - All role builders (goalkeeper, striker, etc.)
   - Utility functions

### Adaptations Required

1. **RobotSituation → World + PlayerHandle**

   - Split read-only queries (World) from control (PlayerHandle)
   - Remove `team_color` access - strategies don't know their color
   - Adapt method signatures

2. **ActionNode → New Skill API**

   - Map old skills to new streamlined set:
     - `GoToPosition` → `player.go_to(pos).with_heading(angle)`
     - `FetchBallWithPreshoot` → `player.pickup_ball(heading)` + `player.reflex_shoot(target)`
     - `Shoot` → `player.reflex_shoot(target)`
     - Dribbling behaviors → `player.dribble_to(pos, heading)`
   - Use skill handles for discrete skills, status checks for completion

3. **Coordination**

   - Move `BtContext` semaphores into strategy's own state
   - Implement passing coordination within strategy
   - No framework support - strategy manages all coordination

4. **Debug Calls**

   - Replace `dies_core::debug_*` with `dies_strategy_api::debug::*`
   - Same semantics, different transport
   - Coordinates auto-transformed for display

5. **GameContext**
   - Adapt to new role builder API
   - Works within strategy process

---

## Removed Components

After refactoring, these are removed from the codebase:

### From `dies-executor`

- `behavior_tree/` module entirely
- `behavior_tree_api` re-exports
- Rhai integration (`rhai_host.rs`, `rhai_plugin.rs`, `rhai_types.rs`, `rhai_type_registration.rs`)
- Strategy function pointer from TeamController
- RoleAssignmentSolver from TeamController
- BtContext from TeamController (semaphores, passing coordination)
- Old skill implementations (replaced with streamlined set)

### Dependencies

- `rhai` crate dependency removed from `dies-executor`

### Configuration

- Script path settings (`blue_script_path`, `yellow_script_path`)
- Replaced with strategy binary name selection

### Skills Removed/Replaced

The old skill set is replaced with the streamlined set:

| Old Skill                  | New Skill                    | Notes                                   |
| -------------------------- | ---------------------------- | --------------------------------------- |
| `GoToPosition` (with_ball) | `GoToPos` + `Dribble`        | Split into separate skills              |
| `Face`                     | Integrated into `GoToPos`    | Use `.with_heading()` or `.facing()`    |
| `FetchBall`                | `PickupBall`                 | Renamed, simplified                     |
| `FetchBallWithPreshoot`    | `PickupBall` + `ReflexShoot` | Split into discrete skills              |
| `Shoot`                    | `ReflexShoot`                | Simplified                              |
| `Kick`                     | Part of `ReflexShoot`        | Integrated                              |
| `TryReceive`               | Strategy logic               | Strategy positions player, uses `go_to` |
| `Wait`                     | Strategy logic               | Strategy manages timing                 |

---

## Implementation Plan

This section provides a detailed, phased approach to implementing the strategy refactor. Each phase is designed to be independently testable and builds upon the previous phase.

### Overview of Current State

**Existing Components (to be refactored or removed):**

| Component              | Location                                             | Action                               |
| ---------------------- | ---------------------------------------------------- | ------------------------------------ |
| `TeamController`       | `dies-executor/src/control/team_controller.rs`       | Simplify - remove strategy logic     |
| `BehaviorTree` system  | `dies-executor/src/behavior_tree/`                   | Move to `v0-strategy` crate          |
| `RoleAssignmentSolver` | `dies-executor/src/behavior_tree/role_assignment.rs` | Move to `v0-strategy` crate          |
| `RobotSituation`       | `dies-executor/src/behavior_tree/bt_core.rs`         | Split into World + PlayerHandle      |
| `BtContext`            | `dies-executor/src/behavior_tree/bt_core.rs`         | Move coordination to strategy        |
| Rhai integration       | `dies-executor/src/behavior_tree/rhai_*.rs`          | Remove entirely                      |
| Current skills         | `dies-executor/src/skills/`                          | Refactor to streamlined set          |
| `dies-strategy` crate  | `crates/dies-strategy/`                              | Migrate to `strategies/v0-strategy/` |
| `GameContext`          | `dies-executor/src/behavior_tree/game_context.rs`    | Redesign for new API                 |

**New Components (to be created):**

| Component                | Location                                      | Purpose                                    |
| ------------------------ | --------------------------------------------- | ------------------------------------------ |
| `dies-strategy-api`      | `crates/dies-strategy-api/`                   | Strategy API (World, PlayerHandle, traits) |
| `dies-strategy-runner`   | `crates/dies-strategy-runner/`                | Process runner, IPC client                 |
| `dies-strategy-protocol` | `crates/dies-strategy-protocol/`              | Shared IPC message types                   |
| `StrategyHost`           | `dies-executor/src/strategy_host/`            | Process management, IPC server             |
| `SkillExecutor`          | `dies-executor/src/control/skill_executor.rs` | Skill execution engine                     |
| `v0-strategy`            | `strategies/v0-strategy/`                     | Ported behavior tree strategy              |

---

### Phase 1: Protocol & Shared Types

**Goal:** Define the IPC protocol and shared types that both the executor and strategy processes will use.

**Duration Estimate:** 1-2 days

#### Tasks

1. **Create `dies-strategy-protocol` crate**

   - Location: `crates/dies-strategy-protocol/`
   - Dependencies: `serde`, `bincode`, `dies-core` (for basic types)

2. **Define IPC message types**

   ```
   src/
   ├── lib.rs
   ├── messages.rs      # HostMessage, StrategyMessage
   ├── world.rs         # WorldSnapshot, BallState, PlayerState
   ├── skill.rs         # SkillCommand, SkillStatus
   └── debug.rs         # DebugEntry, DebugValue
   ```

3. **Implement message types**

   - `HostMessage`: `Init`, `WorldUpdate`, `Shutdown`
   - `StrategyMessage`: `Ready`, `Output`, `Log`
   - `SkillCommand`: `GoToPos`, `Dribble`, `PickupBall`, `ReflexShoot`, `Stop`
   - `SkillStatus`: `Idle`, `Running`, `Succeeded`, `Failed`
   - `WorldSnapshot`: Normalized world state
   - `PlayerState`: Position, velocity, heading, has_ball, handicaps
   - `DebugEntry`: Key-value debug data

4. **Add serialization tests**
   - Round-trip serialization for all message types
   - Size benchmarks for world updates

#### Relevant Files

| Action | File                                            |
| ------ | ----------------------------------------------- |
| Create | `crates/dies-strategy-protocol/Cargo.toml`      |
| Create | `crates/dies-strategy-protocol/src/lib.rs`      |
| Create | `crates/dies-strategy-protocol/src/messages.rs` |
| Create | `crates/dies-strategy-protocol/src/world.rs`    |
| Create | `crates/dies-strategy-protocol/src/skill.rs`    |
| Create | `crates/dies-strategy-protocol/src/debug.rs`    |
| Modify | `Cargo.toml` (workspace members)                |

#### Dependencies

- None (first phase)

#### Verification

```bash
# Build the new crate
cargo build -p dies-strategy-protocol

# Run unit tests
cargo test -p dies-strategy-protocol

# Verify serialization round-trips work
cargo test -p dies-strategy-protocol -- --nocapture serialization
```

**Success Criteria:**

- [ ] All message types compile and derive Serialize/Deserialize
- [ ] Unit tests pass for message serialization
- [ ] `cargo doc` generates clean documentation

---

### Phase 2: Strategy API Crate

**Goal:** Create the public API that strategy implementations will use.

**Duration Estimate:** 2-3 days

#### Tasks

1. **Create `dies-strategy-api` crate**

   - Location: `crates/dies-strategy-api/`
   - Dependencies: `dies-strategy-protocol`, `dies-core`

2. **Implement World API** (`src/world.rs`)

   - Read-only access to normalized world state
   - Field geometry helpers (`own_goal_center`, `opp_goal_center`, etc.)
   - Ball state access
   - Player categorization (`own_players`, `opp_players`)
   - Game state queries

3. **Implement PlayerHandle** (`src/player.rs`)

   - Per-player control interface
   - Continuous skills: `go_to()`, `dribble_to()`
   - Discrete skills: `pickup_ball()`, `reflex_shoot()`
   - Skill status querying
   - Role metadata

4. **Implement Skill Builders** (`src/skill_builders.rs`)

   - `GoToBuilder` with `with_heading()`, `facing()`
   - `DribbleBuilder`
   - `SkillHandle<S>` for discrete skills

5. **Implement TeamContext** (`src/team.rs`)

   - Access to World and PlayerHandles
   - Player iteration

6. **Implement Debug API** (`src/debug.rs`)

   - Thread-local debug collector
   - `cross()`, `line()`, `circle()`, `value()`, `string()`
   - Coordinate auto-transformation markers

7. **Define Strategy trait** (`src/strategy.rs`)

   - `init()`, `update()`, `shutdown()` methods

8. **Create prelude module** (`src/lib.rs`)

   - Re-export common types for ergonomic use

9. **Implement `export_strategy!` macro** (`src/macros.rs`)
   - Factory function export for strategy loading

#### Relevant Files

| Action | File                                             |
| ------ | ------------------------------------------------ |
| Create | `crates/dies-strategy-api/Cargo.toml`            |
| Create | `crates/dies-strategy-api/src/lib.rs`            |
| Create | `crates/dies-strategy-api/src/world.rs`          |
| Create | `crates/dies-strategy-api/src/player.rs`         |
| Create | `crates/dies-strategy-api/src/skill_builders.rs` |
| Create | `crates/dies-strategy-api/src/team.rs`           |
| Create | `crates/dies-strategy-api/src/debug.rs`          |
| Create | `crates/dies-strategy-api/src/strategy.rs`       |
| Create | `crates/dies-strategy-api/src/macros.rs`         |
| Modify | `Cargo.toml` (workspace members)                 |

#### Dependencies

- Phase 1 (dies-strategy-protocol)

#### Verification

```bash
# Build the crate
cargo build -p dies-strategy-api

# Run tests
cargo test -p dies-strategy-api

# Check documentation
cargo doc -p dies-strategy-api --open
```

**Success Criteria:**

- [ ] All API types compile
- [ ] Strategy trait is implementable
- [ ] Builder pattern works ergonomically
- [ ] Documentation is complete

---

### Phase 3: Strategy Runner

**Goal:** Create the process runner that strategy binaries will use.

**Duration Estimate:** 2-3 days

#### Tasks

1. **Create `dies-strategy-runner` crate**

   - Location: `crates/dies-strategy-runner/`
   - Dependencies: `dies-strategy-api`, `dies-strategy-protocol`, `tokio`

2. **Implement IPC client** (`src/ipc.rs`)

   - Unix domain socket connection
   - Message framing (length-prefixed)
   - Async send/receive with bincode
   - Reconnection logic

3. **Implement main loop** (`src/runner.rs`)

   - Command-line argument parsing (socket path, config)
   - Connection establishment
   - World update → strategy update → output cycle
   - Graceful shutdown handling

4. **Implement TeamContext runtime** (`src/context.rs`)

   - World state storage
   - PlayerHandle management
   - Skill command collection
   - Debug data collection

5. **Implement log forwarding** (`src/logging.rs`)

   - Capture strategy logs
   - Forward via IPC

6. **Create runner binary template** (`src/main.rs.template`)
   - Entry point that links with strategy crate

#### Relevant Files

| Action | File                                         |
| ------ | -------------------------------------------- |
| Create | `crates/dies-strategy-runner/Cargo.toml`     |
| Create | `crates/dies-strategy-runner/src/lib.rs`     |
| Create | `crates/dies-strategy-runner/src/ipc.rs`     |
| Create | `crates/dies-strategy-runner/src/runner.rs`  |
| Create | `crates/dies-strategy-runner/src/context.rs` |
| Create | `crates/dies-strategy-runner/src/logging.rs` |
| Modify | `Cargo.toml` (workspace members)             |

#### Dependencies

- Phase 1 (dies-strategy-protocol)
- Phase 2 (dies-strategy-api)

#### Verification

```bash
# Build the crate
cargo build -p dies-strategy-runner

# Run unit tests
cargo test -p dies-strategy-runner

# Test IPC client with mock server (integration test)
cargo test -p dies-strategy-runner --test ipc_integration
```

**Success Criteria:**

- [ ] Runner compiles as a library
- [ ] IPC client can connect/send/receive
- [ ] Main loop processes mock updates correctly
- [ ] Graceful shutdown works

---

### Phase 4: Skill Executor in TeamController

**Goal:** Implement the skill execution engine that runs skills in the executor.

**Duration Estimate:** 2-3 days

#### Tasks

1. **Create SkillExecutor module** (`dies-executor/src/control/skill_executor.rs`)

   - `ExecutableSkill` trait: `update_params()`, `tick()`, `status()`
   - `SkillProgress` enum: `Continue(PlayerControlInput)`, `Done(SkillResult)`
   - Skill instance management per player

2. **Implement streamlined skills**

   - Refactor `GoToPosition` → `GoToPosSkill`
   - Create `DribbleSkill` (extracted from GoToPosition with_ball)
   - Create `PickupBallSkill` (derived from FetchBall)
   - Create `ReflexShootSkill` (derived from Shoot)
   - Add parameter update support to all skills

3. **Add skill lifecycle management**

   - Same-type updates → parameter update
   - Different-type → interrupt and start new
   - After completion → new instance on next command
   - Stop command → immediate halt

4. **Integrate with TeamController (parallel path)**
   - Keep existing behavior tree path working
   - Add new path for SkillCommand input
   - Switch based on configuration flag

#### Relevant Files

| Action   | File                                                              |
| -------- | ----------------------------------------------------------------- |
| Create   | `crates/dies-executor/src/control/skill_executor.rs`              |
| Refactor | `crates/dies-executor/src/skills/go_to_pos.rs`                    |
| Create   | `crates/dies-executor/src/skills/dribble.rs`                      |
| Refactor | `crates/dies-executor/src/skills/fetchball.rs` → `pickup_ball.rs` |
| Refactor | `crates/dies-executor/src/skills/shoot.rs` → `reflex_shoot.rs`    |
| Modify   | `crates/dies-executor/src/skills/mod.rs`                          |
| Modify   | `crates/dies-executor/src/control/mod.rs`                         |

#### Dependencies

- None (can run in parallel with Phase 2-3)

#### Verification

```bash
# Build dies-executor
cargo build -p dies-executor

# Run skill tests
cargo test -p dies-executor -- skill

# Test skill update semantics
cargo test -p dies-executor -- skill_executor

# Run simulator with skill commands (manual test)
cargo run -- sim --skill-test-mode
```

**Success Criteria:**

- [ ] All streamlined skills implemented
- [ ] Parameter updates work smoothly
- [ ] Skill lifecycle behaves correctly
- [ ] Existing behavior trees still work

---

### Phase 5: Strategy Host

**Goal:** Implement the strategy host that manages strategy processes.

**Duration Estimate:** 3-4 days

#### Tasks

1. **Create StrategyHost module** (`dies-executor/src/strategy_host/`)

   - `StrategyHost` struct: process management, socket management
   - `StrategyConnection`: per-strategy IPC connection
   - Strategy discovery (scan `target/strategies/`)

2. **Implement process management**

   - Spawn strategy processes
   - Monitor process health
   - Handle crashes → respawn
   - Graceful shutdown

3. **Implement Unix socket server**

   - Create socket per connection
   - Async message handling
   - Frame-based protocol (length-prefix)

4. **Implement coordinate transformation**

   - Transform world data to team-relative frame before sending
   - Transform skill commands back to world frame after receiving
   - Handle side switching at halftime

5. **Implement world snapshot generation**

   - Convert `TeamData` → `WorldSnapshot`
   - Categorize players as own/opponent
   - Include skill statuses

6. **Implement skill command translation**

   - Convert `SkillCommand` → input for `SkillExecutor`
   - Handle "no command" → continue previous skill

7. **Implement debug data forwarding**

   - Receive debug entries from strategy
   - Transform coordinates
   - Forward to existing debug system

8. **Integrate with Executor**
   - Add StrategyHost to Executor
   - Route world updates to StrategyHost
   - Receive skill commands from StrategyHost
   - Route to TeamController

#### Relevant Files

| Action | File                                                           |
| ------ | -------------------------------------------------------------- |
| Create | `crates/dies-executor/src/strategy_host/mod.rs`                |
| Create | `crates/dies-executor/src/strategy_host/host.rs`               |
| Create | `crates/dies-executor/src/strategy_host/connection.rs`         |
| Create | `crates/dies-executor/src/strategy_host/protocol.rs`           |
| Create | `crates/dies-executor/src/strategy_host/transform.rs`          |
| Modify | `crates/dies-executor/src/lib.rs`                              |
| Modify | `crates/dies-executor/Cargo.toml` (add dies-strategy-protocol) |

#### Dependencies

- Phase 1 (dies-strategy-protocol)
- Phase 4 (SkillExecutor)

#### Verification

```bash
# Build dies-executor
cargo build -p dies-executor

# Run integration tests with mock strategy
cargo test -p dies-executor --test strategy_host_integration

# Test coordinate transformation
cargo test -p dies-executor -- transform

# Manual test: run executor and manually connect a test strategy
cargo run -- sim
# In another terminal: run test strategy binary
./target/strategies/test-strategy
```

**Success Criteria:**

- [ ] Strategy processes spawn correctly
- [ ] IPC connection works
- [ ] Coordinate transformation is correct
- [ ] Skill commands are received and executed
- [ ] Debug data appears in UI

---

### Phase 6: Simplify TeamController

**Goal:** Remove strategy logic from TeamController, keeping only skill execution and compliance.

**Duration Estimate:** 2-3 days

#### Tasks

1. **Extract skill execution path**

   - Move from behavior tree tick to SkillExecutor
   - Receive skill commands from StrategyHost
   - Execute via SkillExecutor

2. **Remove behavior tree integration**

   - Remove `player_behavior_trees` field
   - Remove `bt_context` field
   - Remove `strategy` field
   - Remove `role_solver` field

3. **Remove role assignment**

   - Roles now come from strategy (via player metadata)
   - TeamController just applies role_type for compliance

4. **Simplify update flow**

   - Receive `HashMap<PlayerId, Option<SkillCommand>>` from StrategyHost
   - Pass to SkillExecutor
   - Apply compliance
   - Send to PlayerController

5. **Keep existing functionality**

   - Player removal (yellow cards)
   - Compliance (`comply()` function)
   - Manual override handling
   - MPC integration

6. **Add backward compatibility flag**
   - Allow running with old behavior tree path for testing
   - Feature flag: `legacy-strategy`

#### Relevant Files

| Action | File                                                  |
| ------ | ----------------------------------------------------- |
| Modify | `crates/dies-executor/src/control/team_controller.rs` |
| Modify | `crates/dies-executor/src/control/mod.rs`             |
| Modify | `crates/dies-executor/src/lib.rs`                     |

#### Dependencies

- Phase 4 (SkillExecutor)
- Phase 5 (StrategyHost)

#### Verification

```bash
# Build with new path
cargo build -p dies-executor

# Build with legacy path
cargo build -p dies-executor --features legacy-strategy

# Run tests
cargo test -p dies-executor

# Run simulator and verify robots respond to skill commands
cargo run -- sim
```

**Success Criteria:**

- [ ] TeamController compiles without behavior tree code
- [ ] Skill commands execute correctly
- [ ] Compliance still works
- [ ] Legacy mode still works

---

### Phase 7: Test Strategy

**Goal:** Create a simple test strategy to validate the full pipeline.

**Duration Estimate:** 1-2 days

#### Tasks

1. **Create strategies directory structure**

   ```
   strategies/
   └── test-strategy/
       ├── Cargo.toml
       └── src/
           └── lib.rs
   ```

2. **Implement simple test strategy**

   - All robots go to fixed positions
   - Validates: connection, world data, skill commands, debug visualization

3. **Create build system**

   - Workspace configuration for strategies
   - Build script to compile strategies to `target/strategies/`

4. **Create strategy runner binary**

   - Link test-strategy with dies-strategy-runner
   - Output to `target/strategies/test-strategy`

5. **Write integration test**
   - Spawn executor
   - Spawn test strategy
   - Verify robots move to expected positions

#### Relevant Files

| Action | File                                       |
| ------ | ------------------------------------------ |
| Create | `strategies/test-strategy/Cargo.toml`      |
| Create | `strategies/test-strategy/src/lib.rs`      |
| Create | `strategies/build.rs` (build script)       |
| Modify | `Cargo.toml` (add strategies to workspace) |

#### Dependencies

- Phase 2 (dies-strategy-api)
- Phase 3 (dies-strategy-runner)
- Phase 5 (StrategyHost)

#### Verification

```bash
# Build test strategy
cargo build -p test-strategy

# Verify binary exists
ls target/strategies/test-strategy

# Run full pipeline
cargo run -- sim --strategy test-strategy

# Observe robots moving to expected positions in UI
```

**Success Criteria:**

- [ ] Test strategy compiles
- [ ] Strategy connects to executor
- [ ] Robots respond to skill commands
- [ ] Debug visualization appears
- [ ] Hot reload works (modify code, see changes)

---

### Phase 8: Port v0 Strategy

**Goal:** Port the existing v0 behavior tree strategy to the new architecture.

**Duration Estimate:** 4-5 days

#### Tasks

1. **Create v0-strategy crate**

   ```
   strategies/v0-strategy/
   ├── Cargo.toml
   └── src/
       ├── lib.rs
       ├── behavior_tree/       # Copied from dies-executor
       ├── role_assignment.rs   # Copied from dies-executor
       ├── roles/               # Role implementations
       │   ├── goalkeeper.rs
       │   ├── striker.rs
       │   ├── harasser.rs
       │   ├── waller.rs
       │   └── ...
       └── utils/               # Copied from dies-strategy/v0/utils
   ```

2. **Copy behavior tree core**

   - Copy `dies-executor/src/behavior_tree/` to strategy
   - Adapt to use new API types
   - Remove Rhai-specific code

3. **Copy role assignment**

   - Copy `RoleAssignmentSolver`, `Role`, `RoleBuilder`
   - Adapt to use new API types

4. **Adapt RobotSituation → World + PlayerHandle**

   - Create adapter layer or rewrite to use new API
   - Split read queries (World) from control (PlayerHandle)
   - Remove `team_color` access

5. **Port each role**

   - Goalkeeper: `go_to()` + `facing()` for positioning
   - Striker: `pickup_ball()` + `reflex_shoot()` state machine
   - Harasser: Position to intercept
   - Waller: Defensive positioning
   - Kickoff/FreeKick kickers: Special play handling

6. **Adapt skill mappings**

   - `GoToPosition` → `player.go_to(pos).with_heading(angle)`
   - `FetchBallWithPreshoot` → `pickup_ball(heading)` + state machine
   - `Shoot` → `reflex_shoot(target)`
   - `Face` → `go_to(current_pos).with_heading(angle)`

7. **Port coordination logic**

   - Move semaphores from BtContext to strategy-owned state
   - Implement passing coordination within strategy

8. **Port debug calls**

   - Replace `dies_core::debug_*` with `dies_strategy_api::debug::*`

9. **Test each role individually**
   - Unit tests for role logic
   - Integration tests in simulator

#### Relevant Files

| Action      | File                                             |
| ----------- | ------------------------------------------------ |
| Create      | `strategies/v0-strategy/Cargo.toml`              |
| Create      | `strategies/v0-strategy/src/lib.rs`              |
| Copy+Modify | `strategies/v0-strategy/src/behavior_tree/`      |
| Copy+Modify | `strategies/v0-strategy/src/role_assignment.rs`  |
| Create      | `strategies/v0-strategy/src/roles/mod.rs`        |
| Copy+Modify | `strategies/v0-strategy/src/roles/goalkeeper.rs` |
| Copy+Modify | `strategies/v0-strategy/src/roles/striker.rs`    |
| Copy+Modify | `strategies/v0-strategy/src/roles/harasser.rs`   |
| Copy+Modify | `strategies/v0-strategy/src/roles/waller.rs`     |
| Copy+Modify | `strategies/v0-strategy/src/utils/`              |
| Modify      | `Cargo.toml` (add v0-strategy to workspace)      |

#### Dependencies

- Phase 2 (dies-strategy-api)
- Phase 3 (dies-strategy-runner)
- Phase 7 (test-strategy - for validation patterns)

#### Verification

```bash
# Build v0 strategy
cargo build -p v0-strategy

# Run with v0 strategy
cargo run -- sim --strategy v0-strategy

# Run role-specific tests
cargo test -p v0-strategy

# Compare behavior with legacy mode (if available)
# Visual inspection of robot behavior
```

**Success Criteria:**

- [ ] v0 strategy compiles
- [ ] All roles work correctly
- [ ] Behavior matches original (visual comparison)
- [ ] Role switching works
- [ ] Coordination (semaphores, passing) works

---

### Phase 9: Cleanup & Remove Legacy Code

**Goal:** Remove all deprecated code and the Rhai integration.

**Duration Estimate:** 2-3 days

#### Tasks

1. **Remove behavior tree from dies-executor**

   - Delete `dies-executor/src/behavior_tree/` directory
   - Remove `behavior_tree_api` exports
   - Update `lib.rs`

2. **Remove Rhai dependency**

   - Remove from `Cargo.toml`
   - Remove any Rhai-specific code paths

3. **Remove old skills**

   - Delete deprecated skill implementations
   - Keep only streamlined set: GoToPos, Dribble, PickupBall, ReflexShoot, Stop

4. **Remove dies-strategy crate**

   - Delete `crates/dies-strategy/` entirely
   - Remove from workspace

5. **Update ExecutorSettings**

   - Remove `blue_script_path`, `yellow_script_path`
   - Add `blue_strategy`, `yellow_strategy` (binary names)

6. **Clean up TeamController**

   - Remove legacy-strategy feature flag
   - Remove backward compatibility code

7. **Update documentation**

   - Update architecture docs
   - Update README
   - Update getting started guide

8. **Update Web UI**
   - Add strategy selection dropdown
   - Remove script path configuration
   - Add hot reload button

#### Relevant Files

| Action | File                                                          |
| ------ | ------------------------------------------------------------- |
| Delete | `crates/dies-executor/src/behavior_tree/` (entire directory)  |
| Delete | `crates/dies-strategy/` (entire crate)                        |
| Delete | `crates/dies-executor/src/skills/fetch_ball_with_preshoot.rs` |
| Delete | `crates/dies-executor/src/skills/receive.rs`                  |
| Delete | `crates/dies-executor/src/skills/wait.rs`                     |
| Delete | `crates/dies-executor/src/skills/face.rs`                     |
| Delete | `crates/dies-executor/src/skills/kick.rs`                     |
| Modify | `crates/dies-executor/Cargo.toml` (remove rhai)               |
| Modify | `crates/dies-executor/src/lib.rs`                             |
| Modify | `crates/dies-executor/src/skills/mod.rs`                      |
| Modify | `crates/dies-core/src/executor_settings.rs`                   |
| Modify | `Cargo.toml` (remove dies-strategy from workspace)            |
| Modify | `webui/src/views/Settings.tsx`                                |

#### Dependencies

- Phase 8 (v0 strategy ported and working)

#### Verification

```bash
# Verify no references to removed code
grep -r "behavior_tree" crates/dies-executor/src/
grep -r "rhai" crates/dies-executor/

# Build clean
cargo clean
cargo build

# Run all tests
cargo test

# Run full system
cargo run -- sim --strategy v0-strategy

# Verify hot reload works
# 1. Start system
# 2. Modify strategy code
# 3. Observe automatic reload
```

**Success Criteria:**

- [ ] No compilation errors
- [ ] All tests pass
- [ ] No references to removed code
- [ ] System runs with v0 strategy
- [ ] Hot reload works

---

### Phase 10: Final Testing & Polish

**Goal:** Comprehensive testing and final polish.

**Duration Estimate:** 2-3 days

#### Tasks

1. **Integration testing**

   - Full match simulation with v0 strategy
   - Test all game states (kickoff, free kick, penalty, etc.)
   - Test role switching
   - Test crash recovery

2. **Performance testing**

   - Measure IPC latency
   - Profile memory usage
   - Verify 60Hz update rate is achievable

3. **Hot reload testing**

   - Modify strategy code during run
   - Verify clean restart
   - Verify state is preserved (as designed)

4. **Multi-strategy testing**

   - Run different strategies for each team
   - Test A/B scenarios

5. **Documentation finalization**

   - Strategy development guide
   - API reference
   - Migration guide for existing strategies

6. **CI/CD updates**
   - Add strategy builds to CI
   - Add integration tests

#### Dependencies

- All previous phases

#### Verification

```bash
# Full test suite
cargo test --all

# Long-running stability test
cargo run -- sim --strategy v0-strategy --duration 600

# Performance metrics
cargo run -- sim --strategy v0-strategy --profile

# Multi-strategy test
cargo run -- sim --blue-strategy v0-strategy --yellow-strategy test-strategy
```

**Success Criteria:**

- [ ] All integration tests pass
- [ ] IPC latency < 1ms
- [ ] Memory usage stable over time
- [ ] Hot reload works reliably
- [ ] Documentation complete

---

### Implementation Timeline

| Phase                      | Est. Duration | Dependencies  | Can Parallelize With |
| -------------------------- | ------------- | ------------- | -------------------- |
| 1. Protocol & Shared Types | 1-2 days      | None          | -                    |
| 2. Strategy API            | 2-3 days      | Phase 1       | Phase 4              |
| 3. Strategy Runner         | 2-3 days      | Phase 1, 2    | Phase 4, 5           |
| 4. Skill Executor          | 2-3 days      | None          | Phase 2, 3           |
| 5. Strategy Host           | 3-4 days      | Phase 1, 4    | -                    |
| 6. Simplify TeamController | 2-3 days      | Phase 4, 5    | -                    |
| 7. Test Strategy           | 1-2 days      | Phase 2, 3, 5 | Phase 6              |
| 8. Port v0 Strategy        | 4-5 days      | Phase 2, 3, 7 | -                    |
| 9. Cleanup                 | 2-3 days      | Phase 8       | -                    |
| 10. Final Testing          | 2-3 days      | Phase 9       | -                    |

**Estimated Total:** 20-30 days

**Critical Path:** 1 → 2 → 3 → 5 → 6 → 7 → 8 → 9 → 10

**Parallel Track:** Phase 4 (Skill Executor) can run in parallel with Phases 2-3

---

### Risk Mitigation

| Risk                           | Mitigation                                                 |
| ------------------------------ | ---------------------------------------------------------- |
| IPC performance issues         | Profile early in Phase 5; consider shared memory if needed |
| v0 strategy port takes longer  | Keep legacy-strategy feature until Phase 8 complete        |
| Coordinate transformation bugs | Extensive unit tests in Phase 5; visual debugging          |
| Hot reload instability         | Implement robust crash recovery early                      |
| Breaking existing workflows    | Maintain backward compatibility until Phase 9              |

---

### Rollback Strategy

Each phase is designed to be independently reversible:

1. **Phase 1-4:** New crates, no changes to existing code → just remove new crates
2. **Phase 5:** StrategyHost added, no removal of old code → disable StrategyHost
3. **Phase 6:** TeamController simplified → revert to backup, use legacy-strategy flag
4. **Phase 7-8:** New strategies → just don't use them
5. **Phase 9:** Only performed after Phase 8 verified → git revert if issues
