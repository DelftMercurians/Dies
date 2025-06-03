---
# Statement of Work: Rhai-Powered Behavior Trees for Dies

This document outlines the Statement of Work (SOW) for integrating the Rhai scripting language into the Dies framework. The primary goal is to enable dynamic construction, composition, and execution of behavior trees (BTs), providing a flexible and potentially hot-reloadable way to define robot behaviors. This SOW is based on the detailed specification found in `scripting.md`.

## Functionality Requirements
- **Full Node Type Support**: All existing Rust behavior tree node types (`SelectNode`, `SequenceNode`, `GuardNode`, `ActionNode`, `SemaphoreNode`, `ScoringSelectNode`) must be constructible and usable from Rhai scripts.
- **Declarative Functional Syntax**: Behavior trees should be constructible in Rhai using a clear, declarative, and functional syntax (e.g., `Select([Guard(HasBall(), AttackGoal()), DefendGoal()])`).
- **Scriptable Logic**:
    - Rhai functions (lambdas or named) must be usable as conditions within `GuardNode`s, receiving `RobotSituation` context.
    - Rhai functions (lambdas or named) must be usable as dynamic scorers within `ScoringSelectNode`s, receiving `RobotSituation` context.
- **MVP Entry Point**: Each Rhai script will expose a main entry-point function (e.g., `build_tree(player_id)`) called by the Rust host for each robot, returning a fully constructed BT instance.
- **Logging Integration**: Output from Rhai's `print()` and `debug()` statements should be redirected to the Rust `log` crate.
- **Hot Reloading (Lower Priority)**: The system should be designed for future hot-reloading of behavior tree scripts.

## Implementation Plan

### Phase 1: Core Rhai Integration & Basic Node Support

This phase focuses on setting up the Rhai engine and enabling the construction of basic behavior tree nodes from Rhai scripts.

- [x] **Task 1.1: Integrate Rhai Engine**
    - [x] Subtask 1.1.1: Add `rhai` crate as a dependency to `dies-executor`. (Already completed prior to this SOW update)
    - [x] Subtask 1.1.2: Create a Rhai `Engine` instance within the `TeamController`.
        ```rust
        // In dies-executor/src/team_controller.rs
        // use rhai::Engine;
        // let mut engine = Engine::new();
        ```
    - [x] Subtask 1.1.3: Implement basic script loading (e.g., `engine.compile_file("path/to/script.rhai")`) for testing. Create a test script in `dies-executor/src/test_scripts/test_script.rhai` and load it in the `TeamController`.
    - [x] Subtask 1.1.4: Implement error handling for Rhai script compilation and execution, logging errors using the `log` crate.
- [x] **Task 1.2: Implement Logging Integration**
    - [x] Subtask 1.2.1: Redirect Rhai's `print()` output to `log::info!`.
        ```rust
        // engine.on_print(|text| log::info!("[RHAI SCRIPT] {}", text));
        ```
    - [x] Subtask 1.2.2: Redirect Rhai's `debug()` output to `log::debug!`, including source and position if available.
        ```rust
        /*
        engine.on_debug(|text, source, pos| {
            let src_info = source.map_or_else(String::new, |s| format!(" in '{}'", s));
            let pos_info = if pos.is_none() { String::new() } else { format!(" @ {}", pos) };
            log::debug!("[RHAI SCRIPT DEBUG]{}{}: {}", src_info, pos_info, text);
        });
        */
        ```
- [x] **Task 1.3: Define `RhaiBehaviorNode` Wrapper**
    - [x] Subtask 1.3.1: Create the `RhaiBehaviorNode` struct in Rust to wrap `BehaviorNode`.
        ```rust
        // In dies-executor/src/behavior_tree/rhai_integration.rs (new file perhaps)
        // #[derive(Clone)] // May not be needed if Dynamic is used carefully
        // pub struct RhaiBehaviorNode(pub BehaviorNode);
        ```
    - [x] Subtask 1.3.2: Implement `CustomType` for `RhaiBehaviorNode`.
        ```rust
        /*
        impl CustomType for RhaiBehaviorNode {
            fn build(mut builder: TypeBuilder<Self>) {
                builder.with_name("BehaviorNode");
            }
        }
        */
        ```
    - [x] Subtask 1.3.3: Register `RhaiBehaviorNode` with the Rhai `Engine` using the name `"BehaviorNode"`.
        ```rust
        // engine.register_type::<RhaiBehaviorNode>(); // or engine.register_type_with_name::<RhaiBehaviorNode>("BehaviorNode");
        ```
- [x] **Task 1.4: Expose `SelectNode` to Rhai**
    - [x] Subtask 1.4.1: Implement `rhai_select_node(children: rhai::Array, description: Option<String>) -> Result<RhaiBehaviorNode, Box<EvalAltResult>>`.
        - This function will take a `rhai::Array` of `RhaiBehaviorNode` and an optional description `String`.
        - It will convert `rhai::Array` elements to `Vec<BehaviorNode>`.
        - It will construct a `dies_executor::behavior_tree::SelectNode` and wrap it in `RhaiBehaviorNode`.
    - [x] Subtask 1.4.2: Register `rhai_select_node` with the Rhai `Engine` as `"Select"`.
        ```rust
        // engine.register_fn("Select", rhai_select_node);
        ```
- [x] **Task 1.5: Expose `SequenceNode` to Rhai**
    - [x] Subtask 1.5.1: Implement `rhai_sequence_node(children: rhai::Array, description: Option<String>) -> Result<RhaiBehaviorNode, Box<EvalAltResult>>`.
        - Similar to `rhai_select_node`, but for `dies_executor::behavior_tree::SequenceNode`.
    - [x] Subtask 1.5.2: Register `rhai_sequence_node` with the Rhai `Engine` as `"Sequence"`.
        ```rust
        // engine.register_fn("Sequence", rhai_sequence_node);
        ```
- [x] **Task 1.6: Expose `ActionNode` and Skills to Rhai**
    - [x] Subtask 1.6.1: Define `RhaiSkill(pub Skill)` struct and implement `CustomType` for it (named `"Skill"` or `"RhaiSkill"`).
        ```rust
        // #[derive(Clone)]
        // pub struct RhaiSkill(pub Skill);
        // impl CustomType for RhaiSkill { /* ... */ }
        // engine.register_type_with_name::<RhaiSkill>("RhaiSkill");
        ```
    - [x] Subtask 1.6.2: Implement Rhai constructor functions for a few representative skills (e.g., `GoToPosition`). These functions will return `RhaiSkill`. **Update: All skills from `skills.rs` are now exposed.**
        Example: `fn rhai_goto_skill(x: f64, y: f64, options: Option<Map>) -> RhaiSkill;`
        Covers: `GoToPosition`, `Face` (angle, position, player), `Kick`, `Wait`, `FetchBall`, `InterceptBall`, `ApproachBall`, `FetchBallWithHeading` (angle, position, player).
    - [x] Subtask 1.6.3: Register these skill constructor functions with the Rhai `Engine` (e.g., `"GoToPositionSkill"`).
    - [x] Subtask 1.6.4: Implement `rhai_action_node(skill_wrapper: RhaiSkill, description: Option<String>) -> Result<RhaiBehaviorNode, Box<EvalAltResult>>`.
        - This function will take a `RhaiSkill` and an optional description.
        - It will construct a `dies_executor::behavior_tree::ActionNode` using `skill_wrapper.0` and wrap it in `RhaiBehaviorNode`.
    - [x] Subtask 1.6.5: Register `rhai_action_node` with the Rhai `Engine` as `"Action"`.
        ```rust
        // engine.register_fn("Action", rhai_action_node);
        ```

### Phase 2: Scriptable Logic - Conditions and Scorers

This phase enables defining conditions for `GuardNode` and scoring functions for `ScoringSelectNode` directly in Rhai.

- [x] **Task 2.1: Expose `RobotSituation` to Rhai (as `RhaiRobotSituationView`)**
    - [x] Subtask 2.1.1: Design the `RhaiRobotSituationView`. This could be a Rust struct registered as a custom type, or a `rhai::Map`. It should provide read-only access to necessary fields from `RobotSituation` (e.g., `player_id`, ball position, own position, `has_ball()`, team context info). (Implemented using `rhai::Map`)
    - [x] Subtask 2.1.2: Implement the `create_rhai_situation_view(rs: &RobotSituation, engine: &Engine) -> Dynamic` function in Rust to populate the `RhaiRobotSituationView` from a `RobotSituation`.
        ```rust
        /*
        fn create_rhai_situation_view(rs: &RobotSituation, engine: &Engine) -> Dynamic {
            let mut map = rhai::Map::new();
            map.insert("player_id".into(), rhai::Dynamic::from(rs.player_id.to_string()));
            map.insert("has_ball".into(), rhai::Dynamic::from(rs.has_ball()));
            // ... add other fields ...
            Dynamic::from(map)
        }
        */
        ```
- [x] **Task 2.2: Implement `GuardNode` with Rhai Conditions**
    - [x] Subtask 2.2.1: Implement `rhai_guard_constructor(context: NativeCallContext, condition_fn_ptr: FnPtr, child_dyn: Dynamic, cond_description: Option<String>) -> Result<RhaiBehaviorNode, Box<EvalAltResult>>`.
        - It will take `FnPtr` for the Rhai condition function.
        - The condition function in Rhai will have a signature like `fn my_condition(situation_view) -> bool`.
        - Use `NativeCallContext` to get access to the `Engine` and `AST` (these might need to be `Arc`-wrapped and cloneable if not already).
        - Create a Rust `dies_executor::behavior_tree::Situation` whose closure:
            - Calls `create_rhai_situation_view`.
            - Calls the Rhai function pointed to by `condition_fn_ptr` using `engine.call_fn()` with the `RhaiRobotSituationView`.
            - Handles potential errors from the Rhai call, defaulting to `false`.
        - Construct a `dies_executor::behavior_tree::GuardNode` with this `Situation` and the child node.
    - [x] Subtask 2.2.2: Register `rhai_guard_constructor` with the Rhai `Engine` as `"Guard"`.
- [x] **Task 2.3: Implement `ScoringSelectNode` with Rhai Scorers**
    - [x] Subtask 2.3.1: Implement `rhai_scoring_select_node(context: NativeCallContext, children_scorers_dyn: rhai::Array, hysteresis_margin: f64, description: Option<String>) -> Result<RhaiBehaviorNode, Box<EvalAltResult>>`.
        - `children_scorers_dyn` will be a `rhai::Array` of `rhai::Map`s, each map like `#{ node: RhaiBehaviorNode, scorer: FnPtr }`.
        - The scorer function in Rhai will have a signature like `fn my_scorer(situation_view) -> float`.
        - For each item in `children_scorers_dyn`:
            - Extract the `RhaiBehaviorNode` and the `FnPtr` for the scorer.
            - Create a Rust closure (similar to `GuardNode`'s condition) that:
                - Calls `create_rhai_situation_view`.
                - Calls the Rhai scorer function using `engine.call_fn()`.
                - Returns `f64`, defaulting to `f64::NEG_INFINITY` on error.
            - Collect these into `Vec<(BehaviorNode, Arc<dyn Fn(&RobotSituation) -> f64>)>`.
        - Construct a `dies_executor::behavior_tree::ScoringSelectNode`.
    - [x] Subtask 2.3.2: Register `rhai_scoring_select_node` with the Rhai `Engine` as `"ScoringSelect"`.
- [x] **Task 2.4: Expose `SemaphoreNode` to Rhai**
    - [x] Subtask 2.4.1: Implement `rhai_semaphore_node(child_dyn: Dynamic, id: String, max_count: i64, description: Option<String>) -> Result<RhaiBehaviorNode, Box<EvalAltResult>>`.
        - Ensure `child_dyn` is cast to `RhaiBehaviorNode`.
        - Construct `dies_executor::behavior_tree::SemaphoreNode`.
    - [x] Subtask 2.4.2: Register `rhai_semaphore_node` with the Rhai `Engine` as `"Semaphore"`.

### Phase 3: Script Entry Point and Host Invocation

This phase defines how Rhai scripts provide behavior trees to the Rust application.

- [x] **Task 3.1: Design and Implement Rhai Script Structure**
    - [x] Subtask 3.1.1: Create an example Rhai script (e.g., `standard_player_tree.rhai`).
    - [x] Subtask 3.1.2: This script will define a main function, e.g., `fn build_player_bt(player_id_str) -> BehaviorNode`.
    - [x] Subtask 3.1.3: Inside this Rhai function, construct a behavior tree using the registered node constructors (`Select`, `Sequence`, `Action`, `Guard`, `ScoringSelect`, `Semaphore`).
    - [x] Subtask 3.1.4: Include example Rhai condition functions (e.g., `fn IsBallClose(s) { s.has_ball }`) and scorer functions.
- [x] **Task 3.2: Implement Rust Host Invocation Logic**
    - [x] Subtask 3.2.1: In `TeamController`, fields `rhai_engine` and `rhai_ast` store `Arc<Engine>` and `Arc<AST>` for the main behavior tree script.
    - [x] Subtask 3.2.2: Load and compile the main BT script (e.g., `standard_player_tree.rhai`) once during initialization. Store the `AST`.
    - [x] Subtask 3.2.3: For each robot:
        - Get its `player_id`.
        - Check if a `BehaviorNode` already exists for this `player_id` in a structure like `HashMap<PlayerId, BehaviorNode>` within `TeamController`.
        - If not, or if a tree rebuild is explicitly triggered (e.g., after hot reload):
            - Call the Rhai entry point function (e.g., `"build_player_bt"`) using `engine.call_fn::<RhaiBehaviorNode>(&mut scope, &self.main_bt_ast, "build_player_bt", (player_id_str,))`.
            - Store the retrieved `RhaiBehaviorNode.0` (which is `BehaviorNode`) in the `HashMap`.
        - Handle errors from `call_fn`, potentially falling back to a default Rust-defined BT and storing that.
        - The existing `BehaviorNode` for the player is then used for the current update cycle.

### Phase 4: Hot Reloading (Lower Priority)

This phase adds the capability to reload behavior tree scripts at runtime.

- [ ] **Task 4.1: Implement Script File Monitoring**
    - [ ] Subtask 4.1.1: Choose and integrate a file watching library (e.g., `notify`) or implement a simple polling mechanism to detect changes in Rhai script files.
- [ ] **Task 4.2: Implement AST Recompilation and Swapping**
    - [ ] Subtask 4.2.1: When a script change is detected, recompile the script file into a new `AST` using `engine.compile_file()`.
    - [ ] Subtask 4.2.2: Atomically swap the old `Arc<AST>` in `TeamController` with the new one (`self.main_bt_ast = Arc::new(new_ast);`).
- [ ] **Task 4.3: Trigger Tree Rebuild**
    - [ ] Subtask 4.3.1: After an AST swap, the `TeamController` should invalidate or clear existing player BTs. On the next update cycle for each player (or immediately), if a BT is requested and not present (or marked as invalid), it will be rebuilt by calling the entry point function from the *new* `AST` as per Task 3.2.3.
- [ ] **Task 4.4: State Management for Hot Reload**
    - [ ] Subtask 4.4.1: Investigate and implement strategies for handling state in nodes like `SemaphoreNode` (`player_id_holding_via_this_node`) and `ScoringSelectNode` (`current_best_child_index`, `current_best_child_score`) when a tree is rebuilt. This might involve resetting their state or attempting to transfer it if possible and meaningful. For MVP, resetting might be sufficient.

## Progress Tracking
**Phase 1: Core Rhai Integration & Basic Node Support - Completed**
- Rhai engine integrated into `TeamController`.
- Logging for `print()` and `debug()` from Rhai scripts redirected to Rust's `log` crate.
- Test script loading and error handling implemented.
- `RhaiBehaviorNode` wrapper created and registered.
- `SelectNode` and `SequenceNode` constructors exposed to Rhai as `Select()` and `Sequence()`.
- `RhaiSkill` wrapper created and registered.
- `ActionNode` constructor exposed to Rhai as `Action()`, along with constructors for all skills defined in `skills.rs` (e.g., `GoToPositionSkill`, `FaceAngleSkill`, `KickSkill`, etc.).

**Phase 2: Scriptable Logic - Conditions and Scorers - Completed**
- `RobotSituation` exposed to Rhai scripts via `RhaiRobotSituationView` (implemented as `rhai::Map`).
- `create_rhai_situation_view` function implemented to populate the view.
- `GuardNode` constructor (`Guard`) exposed to Rhai, allowing conditions as Rhai functions.
- `ScoringSelectNode` constructor (`ScoringSelect`) exposed to Rhai, allowing scorers as Rhai functions.
- `SemaphoreNode` constructor (`Semaphore`) exposed to Rhai.

**Phase 3: Script Entry Point and Host Invocation - Completed**
- Rhai script structure designed and implemented.
- Rhai entry point function defined and implemented.
- Behavior tree construction within Rhai script.
- Example condition and scorer functions included.
- Rust host invocation logic implemented.

[This section will be updated as implementation progresses]
---
