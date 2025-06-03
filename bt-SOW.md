---
# Statement of Work: Behavior Tree System Enhancement

## Functionality Requirements
- **Team Level Coordination:** Implement a mechanism for team-wide communication and coordination, including a "Semaphore Node" to limit concurrent execution of specific behaviors by multiple players.
- **Scoring-Based Select Nodes:** Introduce a select node that chooses child branches based on dynamically calculated scores, incorporating hysteresis to prevent rapid oscillations.
- **Visualization System:** Integrate behavior tree state visualization in a new panel in the existing web UI (see for an example `FieldRenderer.tsx`) using `dies_core::debug_info`.
- **Action (Leaf) Nodes:** Develop action nodes that interface with the existing skill system (`crates/dies-executor/src/roles/skills/mod.rs`), translating skill progress into behavior tree statuses and generating `PlayerControlInput`.
- **Integration with TeamController:** Replace the current role and strategy-based behavior control in `crates/dies-executor/src/control/team_controller.rs` with the new behavior tree system.

## Implementation Plan

### Phase 1: Core Behavior Tree Modifications & Action Nodes

**Objective:** Update the core `BehaviorNode` trait to support `PlayerControlInput` generation and implement basic Action Nodes. This is foundational for other features.

- [x] Task 1.1: Modify `BehaviorNode` Trait
    - **Description:** Change the `tick` method signature in `crates/dies-executor/src/behavior_tree/mod.rs` for the `BehaviorNode` trait.
    - **Details:**
        - Current: `fn tick(&mut self, situation: &RobotSituation) -> BehaviorStatus;`
        - New: `fn tick(&mut self, situation: &mut RobotSituation) -> (BehaviorStatus, Option<PlayerControlInput>);`
        - The `RobotSituation` needs to be mutable if we decide to store the `PlayerControlInput` there instead of returning it. For now, let's plan to return it. If `PlayerControlInput` is returned, `RobotSituation` might not need to be mutable for this specific change but could be for `TeamContext` later. We'll stick to returning it.
- [x] Task 1.2: Update Existing Composite Nodes for New `tick` Signature
    - **Description:** Modify `SelectNode`, `SequenceNode`, and `GuardNode` in `crates/dies-executor/src/behavior_tree/mod.rs` to conform to the new `tick` signature.
    - **Details:**
        - These nodes will need to propagate the `Option<PlayerControlInput>` from their children.
        - `SelectNode`: If a child returns `Running` or `Success` with a `Some(input)`, the `SelectNode` should propagate that input.
        - `SequenceNode`:
            - If a child returns `Running` with `Some(input)`, the `SequenceNode` propagates it and remains `Running`.
            - If a child returns `Success` with `Some(input)`, this input might be relevant if the sequence itself is considered to produce a "final" command upon successful completion of that step. However, typically, only the input from a `Running` action is used. For now, propagate input from `Running` or the *last* `Success` in the sequence if the whole sequence succeeds.
        - `GuardNode`: If the condition is met and the child is ticked, propagate the child's `PlayerControlInput`.
- [x] Task 1.3: Define `RobotSituation` Updates (if necessary for input handling)
    - **Description:** Determine if `RobotSituation` needs changes to facilitate `PlayerControlInput` handling (e.g., if we decide against returning it from `tick`).
    - **Details:** For now, we assume `PlayerControlInput` is returned via `tick`. If this proves problematic, `RobotSituation` could gain an `Option<PlayerControlInput>` field that action nodes can set.
- [x] Task 1.4: Implement `ActionNode`
    - **Description:** Create a new `ActionNode` struct in `crates/dies-executor/src/behavior_tree/mod.rs`.
    - **Details:**
        - It will take a `Box<dyn Skill>` as a parameter (e.g., `skills::GoToPosition`).
        - The `tick` method will:
            - Create a `SkillCtx` (from `crates/dies-executor/src/roles/mod.rs`, though its definition/usage might need slight adaptation if roles are entirely replaced).
            - Call the skill's `update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress` method.
            - Translate `SkillProgress` to `(BehaviorStatus, Option<PlayerControlInput>)`:
                - `SkillProgress::Continue(input)` -> `(BehaviorStatus::Running, Some(input))`
                - `SkillProgress::Done(SkillResult::Success)` -> `(BehaviorStatus::Success, None)` (or potentially `Some(PlayerControlInput::default())` if a "stop" command is desired)
                - `SkillProgress::Done(SkillResult::Failure)` -> `(BehaviorStatus::Failure, None)`
        - Implement `description()` for debug purposes, e.g., "Action: GoToPosition {...}".
        - Example Skill usage:
          ```rust
          // In ActionNode::new or similar
          // let skill = GoToPosition::new(target_pos);

          // In ActionNode::tick
          // let player_control_input = match skill.update(skill_ctx) {
          //     SkillProgress::Continue(input) => (BehaviorStatus::Running, Some(input)),
          //     SkillProgress::Done(SkillResult::Success) => (BehaviorStatus::Success, None),
          //     SkillProgress::Done(SkillResult::Failure) => (BehaviorStatus::Failure, None),
          // };
          ```

### Phase 2: Team Level Coordination

**Objective:** Enable team-wide state sharing and implement the Semaphore Node.

- [x] Task 2.1: Define `TeamContext` Struct
    - **Description:** Create a `TeamContext` struct, likely in `crates/dies-executor/src/behavior_tree/mod.rs` or a new `team_context.rs` file.
    - **Details:**
        - This struct will be shared among all players' `RobotSituation` instances for a single team tick.
        - It should contain a mechanism for semaphores, e.g., `semaphores: Arc<RwLock<HashMap<String, (usize, HashSet<PlayerId>)>>>` to store (current_count, set_of_players_holding_semaphore). The `Arc<RwLock<...>>` allows safe shared mutability.
        - Consider methods on `TeamContext` like `try_acquire_semaphore(&self, id: &str, max_count: usize, player_id: PlayerId) -> bool` and `release_semaphore(&self, id: &str, player_id: PlayerId)`.
- [x] Task 2.2: Integrate `TeamContext` into `RobotSituation`
    - **Description:** Add a reference to `TeamContext` in the `RobotSituation` struct.
    - **Details:** `pub team_context: &'a TeamContext,` (lifetime 'a matches other refs in `RobotSituation`).
- [x] Task 2.3: Implement `SemaphoreNode`
    - **Description:** Create the `SemaphoreNode` struct in `crates/dies-executor/src/behavior_tree/mod.rs`.
    - **Details:**
        - Constructor: `SemaphoreNode::new(child: Box<dyn BehaviorNode>, semaphore_id: String, max_count: usize, description: Option<String>)`.
        - `tick` method:
            - Attempts to acquire the semaphore using `TeamContext`.
            - If successful:
                - Tick the child node.
                - If child status is `Success` or `Failure`, release the semaphore. The semaphore should be held if child is `Running`.
                - Return child's status and `PlayerControlInput`.
            - If not successful (semaphore unavailable):
                - Return `(BehaviorStatus::Failure, None)`.
        - `drop` implementation for `SemaphoreNode` might be needed to ensure semaphore release if the node itself is dropped while holding the semaphore (e.g. tree structure changes). A safer pattern is to ensure release happens based on the child's status within the same tick or subsequent ticks.
        - Store `player_id_holding_via_this_node: Option<PlayerId>` in the `SemaphoreNode` to manage release specific to that node instance.
- [ ] Task 2.4: `TeamController` Manages `TeamContext`
    - **Description:** The `TeamController` will be responsible for creating, updating (if necessary, e.g. clearing semaphores each tick or managing their lifecycle), and passing `TeamContext` to `RobotSituation` instances.
    - **Details:** Initialize `TeamContext` once in `TeamController::new` or per-tick in `TeamController::update`. Per-tick re-creation or a clear operation is likely needed for semaphores to reset. `RobotSituation::new` now requires a `TeamContext` reference, laying groundwork for this task.

### Phase 3: Scoring-Based Select Node

**Objective:** Implement a selector that chooses children based on scores with hysteresis.

- [x] Task 3.1: Define Scoring Mechanism
    - **Description:** Define how scores are calculated for children of the `ScoringSelectNode`.
    - **Details:** Each child will be associated with a scoring function: `Box<dyn Fn(&RobotSituation) -> f64>`.
    - The `ScoringSelectNode` will store pairs of `(Box<dyn BehaviorNode>, Box<dyn Fn(&RobotSituation) -> f64>)`.
- [x] Task 3.2: Implement `ScoringSelectNode`
    - **Description:** Create the `ScoringSelectNode` struct in `crates/dies-executor/src/behavior_tree/mod.rs`.
    - **Details:**
        - Constructor: `ScoringSelectNode::new(children_with_scorers: Vec<(Box<dyn BehaviorNode>, Box<dyn Fn(&RobotSituation) -> f64>)>, hysteresis_margin: f64, description: Option<String>)`.
        - Internal state:
            - `current_best_child_index: Option<usize>`
            - `current_best_child_score: f64` (initialized to `f64::NEG_INFINITY`)
        - `tick` method:
            - If `current_best_child_index` is `Some(idx)`:
                - Calculate score of `children_with_scorers[idx]`.
                - If this child is still running from a previous tick and its score is `current_best_child_score` or within a hysteresis margin of the *new* highest score, continue ticking this child.
            - Otherwise, or if the current child failed/succeeded or its score dropped too much:
                - Iterate through all children, calculate their scores.
                - Find the child with the new highest score (`new_max_score`).
                - If `current_best_child_index` was `Some(idx)` and `children_with_scorers[idx]` was running:
                    - If `score_of_current_child >= new_max_score - hysteresis_margin`, stick with `current_best_child_index`.
                    - Else, switch to the child with `new_max_score`. Update `current_best_child_index` and `current_best_child_score`.
                - Else (no previously running child or it stopped):
                    - Switch to the child with `new_max_score`. Update `current_best_child_index` and `current_best_child_score`.
            - Tick the selected child.
            - If the selected child returns `Failure`, reset `current_best_child_index` for the next tick to force re-evaluation without hysteresis for that failed child.
            - Propagate status and `PlayerControlInput`.
- [x] Task 3.3: Reset/Initialization of Scoring Node State
    - **Description:** Ensure the state (`current_best_child_index`) is correctly managed, especially when the tree is first ticked or reset.

### Phase 4: Visualization System

**Objective:** Integrate BT visualization with the debug UI.

- [x] Task 4.0: Add tree node types to `DebugShape`
    - **Description:** Add `TreeNode` to `DebugShape` in `crates/dies-core/src/debug_info.rs`.
    - **Details:**
        - `TreeNode` will have a `name`, `id`, `children_ids`, and `is_active` field.
        - `name`: The node's description.
        - `id`: The node's unique ID.
        - `children_ids`: The unique IDs of the node's children.
        - `is_active`: Set to `true` if the node is Running, or just successfully executed in this tick. For composite nodes, this might mean it's processing, and for leaf nodes, it means it's the one currently chosen.
      - Also add `debug_tree_node` function for convenience.
- [x] Task 4.1: Define Unique Node IDs for Visualization
    - **Description:** Establish a strategy for generating unique IDs for each behavior tree node instance.
    - **Details:**
        - This ID will be used as the `id` field in `DebugShape::TreeNode`.
        - A path-based ID strategy has been implemented. The ID is constructed by concatenating a `viz_path_prefix` (passed via `RobotSituation`) with a node-specific `node_id_fragment`.
        - Each behavior node struct now stores/generates its `node_id_fragment` and uses helper methods on the `BehaviorNode` trait to construct the full ID.
- [x] Task 4.2: Add `get_node_id_fragment()`, `get_full_node_id()`, and `get_child_node_ids()` to `BehaviorNode`
    - **Description:** Add helper methods to the `BehaviorNode` trait to facilitate fetching unique IDs for visualization.
    - **Details:** Implemented `fn get_node_id_fragment(&self) -> String;`, `fn get_full_node_id(&self, current_path_prefix: &str) -> String;`, and `fn get_child_node_ids(&self, current_path_prefix: &str) -> Vec<String>;`. Nodes implement these to provide their part of the ID and their children's IDs.
- [x] Task 4.3: Integrate `debug_tree_node` Calls
    - **Description:** Add calls to `dies_core::debug_info::debug_tree_node` within the `tick` method of each behavior node type (`SelectNode`, `SequenceNode`, `GuardNode`, `ActionNode`, `SemaphoreNode`, `ScoringSelectNode`).
    - **Details:**
        - The `key` for `debug_tree_node` is `format!("bt.p{}.{}", player_id, node_full_id)`.
        - `name`: Uses `self.description()`.
        - `id`: The unique node ID generated by `get_full_node_id()`.
        - `children_ids`: List of unique IDs of direct children, generated by `get_child_node_ids()`.
        - `is_active`: Logic implemented per node to reflect its current execution state.
- [ ] Task 4.4: Pass Necessary Info via `RobotSituation` & `TeamController`
    - **Description:** `RobotSituation` needs to carry `player_id` and the current path prefix. `TeamController` needs to initialize this.
    - **Details:**
        - `RobotSituation` updated to include `pub viz_path_prefix: String`.
        - The part of this task involving `TeamController` creating and passing the initial `viz_path_prefix` to `RobotSituation` instances is deferred to Phase 5. For now, behavior tree construction logic will need to provide a root prefix if testing visualization before Phase 5.

### Phase 5: Integration with `TeamController`

**Objective:** Replace existing strategy/role logic with the Behavior Tree system.

- [x] Task 5.1: Modify `TeamController` Structure
    - **Description:** Update `TeamController` in `crates/dies-executor/src/control/team_controller.rs` to use behavior trees.
    - **Details:**
        - Removed `strategy: StrategyMap`, `active_strat: Option<String>`, `role_states: HashMap<PlayerId, RoleState>`.
        - Added `player_behavior_trees: HashMap<PlayerId, Box<dyn BehaviorNode>>`.
        - Added `team_context: TeamContext`.
        - The `TeamController` now initializes these fields.
- [x] Task 5.2: Update `TeamController::new()`
    - **Description:** Modify the constructor to initialize BTs and `TeamContext`.
    - **Details:** `TeamController::new` now takes only `&ExecutorSettings`, initializes `player_behavior_trees` as an empty `HashMap` and `team_context` using `TeamContext::new()`.
- [x] Task 5.3: Update `TeamController::update()` Method
    - **Description:** Rewrite the core logic in `update()` to use behavior trees.
    - **Details:**
        - At the start of the method, `team_context.clear_semaphores()` is called.
        - For each `player_data` in `world_data.own_players`:
            - Gets or creates (using a placeholder `build_player_bt`) the `player_bt` from `player_behavior_trees`.
            - Creates `RobotSituation` including `player_data`, `world_data`, a reference to `team_context`, and `viz_path_prefix` (e.g., `"p{player_id}"`).
            - `let (status, player_input_opt) = player_bt.tick(&mut robot_situation);`
            - Uses `player_input_opt.unwrap_or_else(PlayerControlInput::default)` as the input. `RoleType` is defaulted if not set by BT (e.g. Goalkeeper for ID 0, Player otherwise) for compatibility with `comply` and `PlayerController`.
            - The existing `manual_override` and `comply` logic are applied to this `PlayerControlInput` (via `PlayerInputs` intermediary for `comply`).
- [x] Task 5.4: Define Behavior Tree Construction Logic
    - **Description:** Design how behavior trees are defined and assigned to players. This might involve creating a new trait or struct similar to `Strategy` but for BTs.
    - **Details:**
        - A placeholder function `fn build_player_bt(player_id: PlayerId, world: &WorldData, settings: &ExecutorSettings) -> Box<dyn BehaviorNode>;` has been implemented in `team_controller.rs`. It currently returns a simple default BT (e.g., go to origin) for any player.
        - Further development is needed to create sophisticated BTs and assignment logic (e.g. based on game states or player roles if roles become a high-level concept for BT selection).
- [x] Task 5.5: Remove Old Strategy/Role Logic
    - **Description:** Phase out the `Strategy::get_role`, `Role::update`, etc., calls.
    - **Details:** All player decisions are now driven by their assigned behavior tree. Old fields (`strategy`, `active_strat`, `role_states`, `halt`) and related logic (strategy selection, role instantiation and updates) have been removed from `TeamController`. `StrategyMap` has been removed from `lib.rs` and call sites updated.

## Progress Tracking
- **Phase 1: Core Behavior Tree Modifications & Action Nodes (Completed)**
    - Modified the `BehaviorNode` trait: Changed the `tick` method signature to `fn tick(&mut self, situation: &mut RobotSituation) -> (BehaviorStatus, Option<PlayerControlInput>)`.
    - Updated `SelectNode`, `SequenceNode`, and `GuardNode`: Adapted their `tick` methods to the new signature and to correctly propagate `PlayerControlInput`.
    - Implemented `ActionNode`: Created a new node that executes a `Skill`, translating its `SkillProgress` into `BehaviorStatus` and `Option<PlayerControlInput>`. `RobotSituation` did not require structural changes for this phase, as `PlayerControlInput` is returned directly from `tick`.
- **Phase 2: Team Level Coordination (Completed)**
    - Defined `TeamContext` struct in `behavior_tree/mod.rs` with `Arc<RwLock<HashMap<String, (usize, HashSet<PlayerId>)>>>` for semaphores and methods `try_acquire_semaphore`, `release_semaphore`, and `clear_semaphores`.
    - Integrated `TeamContext` into `RobotSituation` by adding a `team_context: &'a TeamContext` field and updating `RobotSituation::new`.
    - Implemented `SemaphoreNode` in `behavior_tree/mod.rs`. It uses `TeamContext` to acquire/release semaphores and manages its state (`player_id_holding_via_this_node`) to correctly release the semaphore when its child completes. It does not use `Drop` for semaphore release, relying on tick logic and eventual `TeamContext` clearing by `TeamController`.
    - Task 2.4 (TeamController Manages TeamContext) has foundational changes in place (`RobotSituation` now requires `TeamContext`), with full implementation deferred to Phase 5.
- **Phase 3: Scoring-Based Select Node (Completed)**
    - Defined the scoring mechanism conceptually: children of `ScoringSelectNode` are paired with scoring functions `Box<dyn Fn(&RobotSituation) -> f64>`.
    - Implemented `ScoringSelectNode` in `crates/dies-executor/src/behavior_tree/mod.rs`. This node selects children based on their scores, incorporating a hysteresis margin to prevent rapid switching. It stores `(Box<dyn BehaviorNode>, Box<dyn Fn(&RobotSituation) -> f64>)` pairs.
    - The `ScoringSelectNode`'s internal state (`current_best_child_index`, `current_best_child_score`) is managed to handle initial selection, switching, and reset conditions (e.g., on child failure or success) to ensure correct behavior according to the specified logic.
- **Phase 4: Visualization System (Partially Completed)**
    - Task 4.0: `TreeNode` in `DebugShape` and the `debug_tree_node` function in `dies-core` were confirmed to be already implemented.
    - Task 4.1: Implemented a path-based unique ID generation strategy for behavior tree nodes. Each node contributes a fragment, and the full path is built recursively.
    - Task 4.2: Added `get_node_id_fragment`, `get_full_node_id`, and `get_child_node_ids` methods to the `BehaviorNode` trait and implemented them in all node types.
    - Task 4.3: Integrated `debug_tree_node` calls into the `tick` method of all behavior node types, providing name, ID, children IDs, and active status for visualization.
    - Task 4.4: `RobotSituation` has been updated to include `viz_path_prefix`. The responsibility of `TeamController` to initialize and pass this prefix is now implemented as part of Phase 5.
- **Phase 5: Integration with `TeamController` (Completed)**
    - Modified `TeamController` structure: Removed old strategy/role fields and added `player_behavior_trees` and `team_context`.
    - Updated `TeamController::new()`: Constructor now initializes the new BT-related fields and no longer takes `StrategyMap`.
    - Updated `TeamController::update()`: Core logic rewritten to iterate through players, build/retrieve their BTs, create `RobotSituation` (including `viz_path_prefix` and `team_context`), tick the BTs, and use the resulting `PlayerControlInput`. The `comply` and `manual_override` logic is maintained. `team_context.clear_semaphores()` is called each update.
    - Defined Behavior Tree Construction Logic (Placeholder): A basic `build_player_bt` function is in place. Full BT design and assignment is a subsequent step.
    - Removed Old Strategy/Role Logic: `StrategyMap`, strategy selection, and role management code have been removed from `TeamController` and `lib.rs`.
[This section will be updated as implementation progresses]
---
