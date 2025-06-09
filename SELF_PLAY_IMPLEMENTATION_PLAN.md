# Self-Play Implementation Plan

## Overview

This document outlines the implementation plan for full self-play support in the DIES framework. The goal is to enable controlling both teams simultaneously using **team IDs** (distinct from colors), allowing for self-play scenarios and comprehensive testing.

## Key Architectural Changes

### Team IDs vs Colors

- **Team IDs**: Permanent identifiers that persist across matches and halftime
- **Team Colors**: Can change at halftime, but team IDs remain constant
- **Team Names**: Optional display names associated with team IDs

### Context Objects

- Introduce context objects that bundle team information for propagation
- Flow: TeamController → BehaviorTree → Nodes → Skills
- Also: TeamController → PlayerController
- Can be extended for engine references and settings propagation

## Current State

✅ **Completed:**

- Team-agnostic WorldTracker that outputs WorldData
- TeamMap structure for multiple team controllers
- Basic team activation/deactivation via ControlMsg

❌ **Remaining Challenges:**

1. **Team ID infrastructure and context objects**
2. **Web UI team ID handling and selection**
3. **Debug info team separation using context**
4. **Player override and commands for dual-team operation**
5. **Simulator team ID support**

---

## Phase 1: Team ID Infrastructure and Context Objects

### Task 1.1: Define Team ID Types

**Files:** `crates/dies-core/src/lib.rs`, `crates/dies-core/src/teams.rs`

**Objective:** Create robust team identification system

**Sub-tasks:**

- [ ] 1.1.1 Create TeamId type as a wrapper around u32 or String
- [ ] 1.1.2 Create TeamInfo struct with id, name, and current color
- [ ] 1.1.3 Update ExecutorSettings to include team configuration
- [ ] 1.1.4 Add team management to ExecutorInfo

**Code Changes:**

```rust
// crates/dies-core/src/teams.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[typeshare]
pub struct TeamId(pub u32);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct TeamInfo {
    pub id: TeamId,
    pub name: Option<String>,
    pub current_color: TeamColor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct TeamConfiguration {
    pub team_a: TeamInfo,
    pub team_b: TeamInfo,
}
```

### Task 1.2: Create Context Objects

**Files:** `crates/dies-executor/src/context.rs`, `crates/dies-core/src/context.rs`

**Objective:** Bundle team context for propagation through the execution chain

**Sub-tasks:**

- [x] 1.2.1 Create TeamContext with team ID and debug prefix
- [x] 1.2.2 Create ExecutionContext that includes TeamContext
- [x] 1.2.3 Create BehaviorTreeContext that extends ExecutionContext with engine
- [x] 1.2.4 Update all propagation chains to use context objects

**Code Changes:**

```rust
// crates/dies-core/src/context.rs
#[derive(Debug, Clone)]
pub struct TeamContext {
    pub team_id: TeamId,
    pub team_info: TeamInfo,
    pub debug_prefix: String,
}

impl TeamContext {
    pub fn new(team_info: TeamInfo) -> Self {
        let debug_prefix = format!("team_{}", team_info.id.0);
        Self {
            team_id: team_info.id,
            team_info,
            debug_prefix,
        }
    }
}

// crates/dies-executor/src/context.rs
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub team: TeamContext,
    // Future: settings, other execution-wide context
}

#[derive(Clone)]
pub struct BehaviorTreeContext {
    pub execution: ExecutionContext,
    pub engine: Arc<RwLock<Engine>>,
}
```

### Task 1.3: Update Core Types

**Files:** `crates/dies-core/src/executor_info.rs`, `crates/dies-core/src/world.rs`

**Objective:** Update core data structures to use team IDs

**Sub-tasks:**

- [x] 1.3.1 Update ExecutorInfo to track active team IDs instead of colors
- [x] 1.3.2 Update WorldData to include team configuration
- [x] 1.3.3 Add methods to get team info by ID and color

**Code Changes:**

```rust
// crates/dies-core/src/executor_info.rs
#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub struct ExecutorInfo {
    pub paused: bool,
    pub manual_controlled_players: Vec<PlayerId>,
    pub active_team_ids: Vec<TeamId>,
    pub team_configuration: TeamConfiguration,
}

// crates/dies-core/src/world.rs
#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub struct WorldData {
    // ... existing fields
    pub team_configuration: TeamConfiguration,
}

impl WorldData {
    pub fn get_team_data_by_id(&self, team_id: TeamId) -> TeamData {
        let team_info = self.get_team_info_by_id(team_id);
        self.get_team_data(team_info.current_color)
    }

    pub fn get_team_info_by_id(&self, team_id: TeamId) -> &TeamInfo {
        if self.team_configuration.team_a.id == team_id {
            &self.team_configuration.team_a
        } else {
            &self.team_configuration.team_b
        }
    }
}
```

---

## Phase 2: World Tracker Team ID Support

### Task 2.1: Update WorldTracker for Team IDs

**Files:** `crates/dies-world/src/lib.rs`

**Objective:** Track teams by ID instead of just color

**Sub-tasks:**

- [ ] 2.1.1 Add team configuration to WorldTracker
- [ ] 2.1.2 Update player tracking to associate with team IDs
- [ ] 2.1.3 Add methods to get controlled team IDs
- [ ] 2.1.4 Handle team color changes (halftime switches)

**Code Changes:**

```rust
// crates/dies-world/src/lib.rs
pub struct WorldTracker {
    // Replace color-based tracking with ID-based
    team_a_players_tracker: HashMap<PlayerId, PlayerTracker>,
    team_b_players_tracker: HashMap<PlayerId, PlayerTracker>,

    ball_tracker: BallTracker,
    game_state_tracker: GameStateTracker,
    field_geometry: Option<FieldGeometry>,
    side_assignment: Option<SideAssignment>,

    // Team configuration
    team_configuration: TeamConfiguration,
    team_a_controlled: bool,
    team_b_controlled: bool,

    // ... existing timestamp fields
}

impl WorldTracker {
    pub fn get_controlled_team_ids(&self) -> Vec<TeamId> {
        let mut controlled = Vec::new();
        if self.team_a_controlled {
            controlled.push(self.team_configuration.team_a.id);
        }
        if self.team_b_controlled {
            controlled.push(self.team_configuration.team_b.id);
        }
        controlled
    }

    pub fn update_team_configuration(&mut self, config: TeamConfiguration) {
        // Handle color changes - migrate player trackers if needed
        self.team_configuration = config;
    }
}
```

### Task 2.2: Feedback Handling with Team IDs

**Files:** `crates/dies-world/src/lib.rs`

**Objective:** Associate feedback with correct team ID

**Sub-tasks:**

- [ ] 2.2.1 Update feedback processing to determine team ID
- [ ] 2.2.2 Mark teams as controlled based on team ID
- [ ] 2.2.3 Handle cases where color-to-team mapping changes

**Code Changes:**

```rust
// crates/dies-world/src/lib.rs
impl WorldTracker {
    pub fn update_from_feedback(&mut self, feedback: &PlayerFeedbackMsg, time: WorldInstant) {
        // Determine which team this player belongs to by vision data
        let team_id = self.determine_player_team_id(feedback.id);

        match team_id {
            Some(id) if id == self.team_configuration.team_a.id => {
                if let Some(tracker) = self.team_a_players_tracker.get_mut(&feedback.id) {
                    tracker.update_from_feedback(feedback, time);
                    self.team_a_controlled = true;
                }
            }
            Some(id) if id == self.team_configuration.team_b.id => {
                if let Some(tracker) = self.team_b_players_tracker.get_mut(&feedback.id) {
                    tracker.update_from_feedback(feedback, time);
                    self.team_b_controlled = true;
                }
            }
            _ => {
                log::warn!("Received feedback for player {} but could not determine team ID", feedback.id);
            }
        }
    }

    fn determine_player_team_id(&self, player_id: PlayerId) -> Option<TeamId> {
        // Check current vision data to see which team this player belongs to
        // This handles color changes correctly
        // ... implementation
    }
}
```

---

## Phase 3: Executor Team ID Support

### Task 3.1: Update TeamMap for Team IDs

**Files:** `crates/dies-executor/src/lib.rs`

**Objective:** Use team IDs instead of colors for team management

**Sub-tasks:**

- [ ] 3.1.1 Replace blue/yellow controllers with team ID-based map
- [ ] 3.1.2 Update team activation/deactivation to use team IDs
- [ ] 3.1.3 Add methods to get active team IDs
- [ ] 3.1.4 Handle team configuration changes

**Code Changes:**

```rust
// crates/dies-executor/src/lib.rs
#[derive(Debug)]
pub struct TeamMap {
    controllers: HashMap<TeamId, TeamController>,
    team_configuration: TeamConfiguration,
}

impl TeamMap {
    pub fn new(team_configuration: TeamConfiguration) -> Self {
        Self {
            controllers: HashMap::new(),
            team_configuration,
        }
    }

    pub fn activate_team(&mut self, team_id: TeamId, settings: &ExecutorSettings) {
        let team_info = self.get_team_info(team_id);
        let context = TeamContext::new(team_info.clone());
        let controller = TeamController::new(settings, context);
        self.controllers.insert(team_id, controller);
    }

    pub fn get_active_team_ids(&self) -> Vec<TeamId> {
        self.controllers.keys().cloned().collect()
    }

    pub fn update(&mut self, world_data: &WorldData, manual_override: HashMap<PlayerId, PlayerControlInput>) {
        for (team_id, controller) in self.controllers.iter_mut() {
            let team_data = world_data.get_team_data_by_id(*team_id);

            // Filter manual overrides for this team
            let team_overrides = self.filter_overrides_for_team(*team_id, &manual_override, world_data);

            controller.update(team_data, team_overrides);
        }
    }
}
```

### Task 3.2: Update TeamController with Context

**Files:** `crates/dies-executor/src/control/team_controller.rs`

**Objective:** Propagate team context through behavior trees

**Sub-tasks:**

- [ ] 3.2.1 Add TeamContext to TeamController
- [ ] 3.2.2 Create BehaviorTreeContext for behavior tree execution
- [ ] 3.2.3 Update RobotSituation to include context
- [ ] 3.2.4 Propagate context to PlayerController

**Code Changes:**

```rust
// crates/dies-executor/src/control/team_controller.rs
pub struct TeamController {
    player_controllers: HashMap<PlayerId, PlayerController>,
    settings: ExecutorSettings,
    start_time: std::time::Instant,
    player_behavior_trees: HashMap<PlayerId, BehaviorTree>,
    script_host: RhaiHost,

    // Context instead of just team_context
    execution_context: ExecutionContext,
}

impl TeamController {
    pub fn new(settings: &ExecutorSettings, team_context: TeamContext) -> Self {
        let execution_context = ExecutionContext {
            team: team_context,
        };

        Self {
            // ... existing fields
            execution_context,
        }
    }

    pub fn update(&mut self, world_data: TeamData, manual_override: HashMap<PlayerId, PlayerControlInput>) {
        // ... existing code until behavior tree execution

        let bt_context = BehaviorTreeContext {
            execution: self.execution_context.clone(),
            engine: self.script_host.engine(),
        };

        for player_data in &world_data.own_players {
            let player_id = player_data.id;

            let viz_path_prefix = format!("{}.p{}",
                self.execution_context.team.debug_prefix,
                player_id
            );

            let mut robot_situation = RobotSituation::new(
                player_id,
                world_data.clone(),
                bt_context.clone(),
                viz_path_prefix,
            );

            // ... rest of behavior tree execution
        }

        // Update player controllers with context
        for controller in self.player_controllers.values_mut() {
            controller.update_with_context(
                // ... existing parameters
                &self.execution_context,
            );
        }
    }
}
```

---

## Phase 4: Debug Context Propagation

### Task 4.1: Update Debug Functions for Team Context

**Files:** `crates/dies-core/src/debug_info.rs`

**Objective:** Use team context for debug key prefixing

**Sub-tasks:**

- [ ] 4.1.1 Add context-aware debug functions
- [ ] 4.1.2 Update debug macros to accept context
- [ ] 4.1.3 Maintain backward compatibility for non-team debug calls

**Code Changes:**

```rust
// crates/dies-core/src/debug_info.rs
pub fn debug_record_with_context(context: &TeamContext, key: impl Into<String>, value: DebugValue) {
    let prefixed_key = format!("{}.{}", context.debug_prefix, key.into());
    debug_record(prefixed_key, value);
}

// Enhanced macros that accept context
#[macro_export]
macro_rules! dbg_draw_ctx {
    ($context:expr, $key:tt, $shape:ident, $($args:expr),+) => {
        dies_core::debug_record_with_context($context, $key, /* shape construction */);
    };
}

// Convenience functions for common debug operations
pub fn debug_value_ctx(context: &TeamContext, key: impl Into<String>, value: f64) {
    debug_record_with_context(context, key, DebugValue::Number(value));
}

pub fn debug_line_ctx(context: &TeamContext, key: impl Into<String>, start: Vector2, end: Vector2, color: DebugColor) {
    debug_record_with_context(context, key, DebugValue::Shape(DebugShape::Line { start, end, color }));
}
```

### Task 4.2: Update Behavior Tree Context

**Files:** `crates/dies-executor/src/behavior_tree/bt_core.rs`

**Objective:** Include context in behavior tree execution

**Sub-tasks:**

- [ ] 4.2.1 Update RobotSituation to include BehaviorTreeContext
- [ ] 4.2.2 Update all skill implementations to use context for debug
- [ ] 4.2.3 Update behavior tree nodes to propagate context

**Code Changes:**

```rust
// crates/dies-executor/src/behavior_tree/bt_core.rs
#[derive(Clone)]
pub struct RobotSituation {
    pub player_id: PlayerId,
    pub world: Arc<TeamData>,
    pub context: BehaviorTreeContext,
    pub viz_path_prefix: String,
}

impl RobotSituation {
    pub fn debug_value(&self, key: impl Into<String>, value: f64) {
        debug_value_ctx(&self.context.execution.team, key, value);
    }

    pub fn debug_line(&self, key: impl Into<String>, start: Vector2, end: Vector2, color: DebugColor) {
        debug_line_ctx(&self.context.execution.team, key, start, end, color);
    }
}
```

### Task 4.3: Update Player Controller Context

**Files:** `crates/dies-executor/src/control/player_controller.rs`

**Objective:** Use team context in player controller debug output

**Sub-tasks:**

- [ ] 4.3.1 Add context parameter to player controller update
- [ ] 4.3.2 Update all debug calls to use team-prefixed keys
- [ ] 4.3.3 Ensure debug info doesn't conflict between teams

**Code Changes:**

```rust
// crates/dies-executor/src/control/player_controller.rs
impl PlayerController {
    pub fn update_with_context(
        &mut self,
        state: &PlayerData,
        world: &TeamData,
        input: &PlayerControlInput,
        dt: f64,
        is_manual_override: bool,
        obstacles: Vec<Obstacle>,
        all_players: &[&PlayerData],
        context: &ExecutionContext,
    ) {
        // ... existing logic

        // Use context for debug output
        debug_value_ctx(&context.team, format!("p{}.sx", self.id), cmd.sx);
        debug_value_ctx(&context.team, format!("p{}.sy", self.id), cmd.sy);
        debug_value_ctx(&context.team, format!("p{}.w", self.id), cmd.w);

        // ... rest of implementation
    }
}
```

---

## Phase 5: Web UI Team ID Support

### Task 5.1: Update UI Data Structures

**Files:** `crates/dies-webui/src/lib.rs`, `crates/dies-core/src/world.rs`

**Objective:** Handle team IDs in UI communication

**Sub-tasks:**

- [ ] 5.1.1 Update UiWorldState to include team configuration
- [ ] 5.1.2 Add team selection commands
- [ ] 5.1.3 Update executor info to include team IDs

**Code Changes:**

```rust
// crates/dies-webui/src/lib.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub enum UiCommand {
    SetManualOverride {
        player_id: PlayerId,
        manual_override: bool,
    },
    SetActiveTeams {
        team_ids: Vec<TeamId>,
    },
    UpdateTeamConfiguration {
        config: TeamConfiguration,
    },
    SetPrimaryTeam {
        team_id: TeamId,
    },
}
```

### Task 5.2: Update UI Team Selection

**Files:** `webui/src/App.tsx`, `webui/src/api.tsx`

**Objective:** Allow UI to select and control teams by ID

**Sub-tasks:**

- [ ] 5.2.1 Add team ID selector component
- [ ] 5.2.2 Display team names when available
- [ ] 5.2.3 Update local state to track selected team ID
- [ ] 5.2.4 Convert WorldData to TeamData using team ID

**Code Changes:**

```tsx
// webui/src/App.tsx
const [selectedTeamId, setSelectedTeamId] = useState<TeamId | null>(null);
const [teamConfiguration, setTeamConfiguration] =
  useState<TeamConfiguration | null>(null);

// Team selector component
<ToggleGroup
  type="single"
  value={selectedTeamId?.toString()}
  onValueChange={(value) =>
    setSelectedTeamId(value ? new TeamId(parseInt(value)) : null)
  }
  className="border border-gray-500 rounded-lg"
>
  {teamConfiguration && (
    <>
      <ToggleGroupItem value={teamConfiguration.team_a.id.toString()}>
        {teamConfiguration.team_a.name || `Team ${teamConfiguration.team_a.id}`}
        {` (${teamConfiguration.team_a.current_color})`}
      </ToggleGroupItem>
      <ToggleGroupItem value={teamConfiguration.team_b.id.toString()}>
        {teamConfiguration.team_b.name || `Team ${teamConfiguration.team_b.id}`}
        {` (${teamConfiguration.team_b.current_color})`}
      </ToggleGroupItem>
    </>
  )}
</ToggleGroup>;
```

```typescript
// webui/src/api.tsx
const convertWorldDataToTeamData = (
  worldData: WorldData,
  teamId: TeamId
): TeamData => {
  return worldData.get_team_data_by_id(teamId);
};
```

### Task 5.3: Update Debug View for Team Context

**Files:** `webui/src/views/DebugView.tsx`

**Objective:** Filter and display debug info by team ID

**Sub-tasks:**

- [ ] 5.3.1 Update debug filtering to use team ID prefixes
- [ ] 5.3.2 Add team context to debug display
- [ ] 5.3.3 Handle switching between team debug views

**Code Changes:**

```typescript
// webui/src/views/DebugView.tsx
const filterDebugByTeam = (debugMap: DebugMap, teamId: TeamId): DebugMap => {
  const prefix = `team_${teamId}.`;
  return Object.fromEntries(
    Object.entries(debugMap).filter(([key]) => key.startsWith(prefix))
  );
};

const TeamDebugSelector = ({
  teamConfiguration,
  selectedTeamId,
  onTeamSelect,
}: {
  teamConfiguration: TeamConfiguration;
  selectedTeamId: TeamId | null;
  onTeamSelect: (teamId: TeamId) => void;
}) => (
  <div className="mb-4">
    <label className="block text-sm font-medium mb-2">Debug Team:</label>
    <select
      value={selectedTeamId?.toString() || ""}
      onChange={(e) => onTeamSelect(new TeamId(parseInt(e.target.value)))}
      className="border rounded px-2 py-1"
    >
      <option value="">All Teams</option>
      <option value={teamConfiguration.team_a.id.toString()}>
        {teamConfiguration.team_a.name || `Team ${teamConfiguration.team_a.id}`}
      </option>
      <option value={teamConfiguration.team_b.id.toString()}>
        {teamConfiguration.team_b.name || `Team ${teamConfiguration.team_b.id}`}
      </option>
    </select>
  </div>
);
```

---

## Phase 6: Simulator Team ID Support

### Task 6.1: Update Simulator for Team IDs

**Files:** `crates/dies-simulator/src/lib.rs`

**Objective:** Support team IDs in simulation

**Sub-tasks:**

- [ ] 6.1.1 Add team configuration to simulation
- [ ] 6.1.2 Update player creation to specify team ID
- [ ] 6.1.3 Update feedback generation to include team context
- [ ] 6.1.4 Handle team-neutral game state management

**Code Changes:**

```rust
// crates/dies-simulator/src/lib.rs
#[derive(Debug)]
struct Player {
    id: PlayerId,
    team_id: TeamId,
    rigid_body_handle: RigidBodyHandle,
    // ... existing fields
}

pub struct Simulation {
    // ... existing fields
    team_configuration: TeamConfiguration,
}

impl SimulationBuilder {
    pub fn add_player_for_team(mut self, team_id: TeamId, id: u32, position: Vector2, yaw: Angle) -> Self {
        // ... create rigid body as before

        self.sim.players.push(Player {
            id: PlayerId::new(id),
            team_id,
            rigid_body_handle,
            // ... other fields
        });
        self
    }

    pub fn with_team_configuration(mut self, config: TeamConfiguration) -> Self {
        self.sim.team_configuration = config;
        self
    }
}
```

---

## Phase 7: Control Message Updates

### Task 7.1: Update Control Messages for Team IDs

**Files:** `crates/dies-executor/src/handle.rs`

**Objective:** Use team IDs in control messages

**Sub-tasks:**

- [ ] 7.1.1 Update ControlMsg enum to use team IDs
- [ ] 7.1.2 Add team configuration management commands
- [ ] 7.1.3 Update team activation commands

**Code Changes:**

```rust
// crates/dies-executor/src/handle.rs
#[derive(Debug)]
pub enum ControlMsg {
    SetPlayerOverride {
        player_id: PlayerId,
        override_active: bool,
    },
    PlayerOverrideCommand(PlayerId, PlayerOverrideCommand),
    SetPause(bool),
    UpdateSettings(ExecutorSettings),
    GcCommand {
        command: Command,
    },
    SimulatorCmd(SimulatorCmd),

    // Team ID-based commands
    SetActiveTeams {
        team_ids: Vec<TeamId>,
    },
    UpdateTeamConfiguration {
        config: TeamConfiguration,
    },
    SetPrimaryTeam {
        team_id: TeamId,
    },
}
```

---

## Phase 8: Integration and Testing

### Task 8.1: End-to-End Integration

**Files:** All modified files

**Objective:** Ensure all components work together with team IDs

**Sub-tasks:**

- [ ] 8.1.1 Test team ID assignment and tracking
- [ ] 8.1.2 Test debug context propagation
- [ ] 8.1.3 Test team color changes (halftime simulation)
- [ ] 8.1.4 Test UI team selection and control
- [ ] 8.1.5 Validate simulator team ID support

### Task 8.2: Documentation Updates

**Files:** Documentation, README files

**Objective:** Document team ID system and context usage

**Sub-tasks:**

- [ ] 8.2.1 Update README with team ID concepts
- [ ] 8.2.2 Document context object usage patterns
- [ ] 8.2.3 Create examples for team configuration
- [ ] 8.2.4 Document debug context best practices

---

## Implementation Priority

1. **Phase 1** (Critical): Team ID infrastructure and context objects
2. **Phase 2** (High): WorldTracker team ID support
3. **Phase 3** (High): Executor team ID support
4. **Phase 4** (Medium): Debug context propagation
5. **Phase 5** (Medium): Web UI team ID support
6. **Phase 6** (Low): Simulator team ID support
7. **Phase 7** (Medium): Control message updates
8. **Phase 8** (Low): Integration and testing

## Risk Assessment

- **High Risk**: Team ID migration - requires careful data migration and testing
- **Medium Risk**: Context object propagation - touches many files, potential for missed updates
- **Low Risk**: UI team selection - well-contained changes

## Success Criteria

- [ ] Teams can be identified by persistent IDs independent of colors
- [ ] UI can display and control teams by ID with optional names
- [ ] Debug information is properly separated by team context
- [ ] Player overrides work correctly for both teams using team IDs
- [ ] Simulator supports team ID-based operation
- [ ] Team colors can change without breaking team identification
- [ ] Context objects properly propagate team information through execution chain
- [ ] No conflicts or interference between team operations
- [ ] Performance remains acceptable with dual-team operation

## Key Benefits

1. **Robust Team Identity**: Team IDs persist across color changes and matches
2. **Clean Debug Separation**: Context objects ensure proper debug key prefixing
3. **Extensible Architecture**: Context objects can carry additional information (settings, etc.)
4. **Better UI Experience**: Team names provide better user identification
5. **Future-Proof**: Architecture supports tournament scenarios with multiple matches
