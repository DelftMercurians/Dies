# Rhai-Powered Behavior Trees for Dies: Specification & Design

## 1. Introduction

This document outlines the design and implementation strategy for integrating the Rhai scripting language into the Dies framework to enable dynamic construction and composition of behavior trees (BTs). The primary goal is to provide a flexible, declarative, and potentially hot-reloadable way to define robot behaviors.

## 2. Core Requirements

The integration must satisfy the following core requirements:

1.  **Full Node Type Support**: All existing Rust behavior tree node types (`SelectNode`, `SequenceNode`, `GuardNode`, `ActionNode`, `SemaphoreNode`, `ScoringSelectNode`) must be constructible and usable from Rhai scripts.
2.  **Declarative Functional Syntax**: Behavior trees should be constructible in Rhai using a clear, declarative, and functional syntax (e.g., `Select([Guard(HasBall(), AttackGoal()), DefendGoal()])`).
3.  **Scriptable Logic**:
    - **Conditions (`Situation`)**: Rhai functions (lambdas or regular named functions) must be usable as conditions within `GuardNode`s. These functions will receive `RobotSituation` context.
    - **Scoring Functions**: Rhai functions (lambdas or regular named functions) must be usable as dynamic scorers within `ScoringSelectNode`s. These functions will also receive `RobotSituation` context.
4.  **MVP Entry Point**: For the Minimum Viable Product (MVP), each Rhai script will expose a main entry-point function (e.g., `build_tree(player_id)`). This function will be called by the Rust host for each robot, returning a fully constructed BT instance tailored for that robot.
5.  **Logging Integration**: Output from Rhai's `print()` and `debug()` statements should be redirected to the Rust `log` crate.
6.  **Hot Reloading (Lower Priority)**: The system should be designed with future hot-reloading of behavior tree scripts in mind.

## 3. Design and Implementation Details

### 3.1. Rhai Engine Integration

- **Engine Setup**: A Rhai `Engine` instance will be created and configured within the Dies `TeamController` or a dedicated `BehaviorTreeManager`.
  ```rust
  use rhai::Engine;
  let mut engine = Engine::new();
  // Further configuration (e.g., packages, custom types, functions) will follow.
  ```
- **Script Loading**: Initially, scripts might be loaded from files. The `Engine::compile_file()` method can be used.
  ```rust
  // let ast = engine.compile_file("path/to/robot_bt_script.rhai")?;
  ```
- **Error Handling**: Rhai's `EvalAltResult` will be used to handle script compilation and execution errors, logging them appropriately.

### 3.2. Exposing Behavior Tree Nodes to Rhai

Each Rust BT node will have a corresponding Rhai constructor function. These functions will be registered with the Rhai `Engine`.

- **Node Representation in Rhai**:

  - A Rust wrapper struct, `RhaiBehaviorNode(Box<dyn BehaviorNode>)`, will be created.
  - This struct will be registered as a custom type in Rhai (e.g., named `"BehaviorNode"`).
  - All Rhai node constructor functions will return an instance of this `RhaiBehaviorNode` custom type.

  ```rust
  #[derive(Clone)] // May not be strictly necessary if used carefully with Dynamic
  pub struct RhaiBehaviorNode(pub Box<dyn BehaviorNode>);

  impl CustomType for RhaiBehaviorNode {
      fn build(mut builder: TypeBuilder<Self>) {
          builder.with_name("BehaviorNode");
          // Potentially add methods later if needed, e.g., for debugging from Rhai
      }
  }
  // engine.register_type::<RhaiBehaviorNode>(); // In engine setup
  ```

- **Constructor Functions (Rust side, registered to Rhai)**:

  - **`SelectNode(children: Array, description: String) -> BehaviorNode`**:

    ```rust
    // Rust function registered as "Select" in Rhai
    fn rhai_select_node(children_dyn: rhai::Array, description: Option<String>) -> Result<RhaiBehaviorNode, Box<EvalAltResult>> {
        let mut rust_children: Vec<Box<dyn BehaviorNode>> = Vec::new();
        for child_dyn in children_dyn {
            let node_wrapper = child_dyn.try_cast::<RhaiBehaviorNode>()
                .ok_or_else(|| Box::new(EvalAltResult::ErrorMismatchDataType("Expected BehaviorNode".into(), child_dyn.type_name().into(), Position::NONE)))?;
            rust_children.push(node_wrapper.0);
        }
        Ok(RhaiBehaviorNode(Box::new(SelectNode::new(rust_children, description))))
    }
    // engine.register_fn("Select", rhai_select_node);
    ```

  - **`SequenceNode(children: Array, description: String) -> BehaviorNode`**: Similar to `SelectNode`.

  - **`ActionNode(skill: Dynamic, description: String) -> BehaviorNode`**:

    - Skills (implementing `dies_executor::roles::Skill`) will be wrapped in a custom Rhai type, e.g., `RhaiSkill(Box<dyn Skill>)`.
    - Rust functions will be registered to construct specific skills (e.g., `GoToPositionSkill(x: float, y: float) -> RhaiSkill`).

    ```rust
    // Example: Skill wrapper and constructor
    #[derive(Clone)]
    pub struct RhaiSkill(pub Box<dyn Skill>);
    impl CustomType for RhaiSkill { /* ... */ }

    fn rhai_goto_skill(x: f64, y: f64) -> RhaiSkill {
        RhaiSkill(Box::new(GoToPosition::new(Vector2::new(x, y))))
    }
    // engine.register_fn("GoToPositionSkill", rhai_goto_skill);

    // ActionNode constructor in Rhai
    fn rhai_action_node(skill_wrapper_dyn: Dynamic, description: Option<String>) -> Result<RhaiBehaviorNode, Box<EvalAltResult>> {
        let skill_wrapper = skill_wrapper_dyn.try_cast::<RhaiSkill>()
            .ok_or_else(|| /* error */ )?;
        Ok(RhaiBehaviorNode(Box::new(ActionNode::new(skill_wrapper.0, description))))
    }
    // engine.register_fn("Action", rhai_action_node);
    ```

  - **`GuardNode(condition: FnPtr, child: BehaviorNode, description: String) -> BehaviorNode`**: See section 3.3.

  - **`SemaphoreNode(child: BehaviorNode, id: String, max_count: int, description: String) -> BehaviorNode`**:

    ```rust
    fn rhai_semaphore_node(child_dyn: Dynamic, id: String, max_count: i64, description: Option<String>) -> Result<RhaiBehaviorNode, Box<EvalAltResult>> {
        let child_wrapper = child_dyn.try_cast::<RhaiBehaviorNode>()
            .ok_or_else(|| /* error */ )?;
        Ok(RhaiBehaviorNode(Box::new(SemaphoreNode::new(child_wrapper.0, id, max_count as usize, description))))
    }
    // engine.register_fn("Semaphore", rhai_semaphore_node);
    ```

  - **`ScoringSelectNode(children_scorers: Array, hysteresis: float, description: String) -> BehaviorNode`**: See section 3.4.

### 3.3. Handling `Situation` (Conditions) in Rhai

`GuardNode` requires a `Situation` which wraps a `Box<dyn Fn(&RobotSituation) -> bool>`.

- **Exposing `RobotSituation` to Rhai**:

  - A simplified, read-only proxy for `RobotSituation` will be created, say `RhaiRobotSituationView`. This can be an object map (`rhai::Map`) or a registered custom type.
  - This proxy will be populated from the Rust `RobotSituation` on each tick before calling the Rhai condition function.
  - Fields to expose: `player_id` (as string/int), relevant `world` data (ball position, opponent positions), `player_data` (own position, velocity, `has_ball`), relevant `team_context`.

  ```rust
  // Example: Creating RhaiRobotSituationView as a Map
  fn create_rhai_situation_view(rs: &RobotSituation, engine: &Engine) -> Dynamic {
      let mut map = rhai::Map::new();
      map.insert("player_id".into(), 현실세계::from(rs.player_id.to_string())); // Example
      map.insert("has_ball".into(), 현실세계::from(rs.has_ball()));
      // ... add other necessary fields from rs.world, rs.player_data
      Dynamic::from(map)
  }
  ```

- **Rhai Condition Function Signature**: Rhai functions used as conditions will have the signature: `fn my_condition_func(situation_view) -> bool`.

- **`GuardNode` Constructor in Rhai**:

  - The Rhai `Guard` constructor will take an `FnPtr` (function pointer) to the Rhai condition function.
  - The Rust implementation of this constructor (`rhai_guard_constructor`) will capture this `FnPtr`, along with necessary context like `Arc<Engine>` and `Arc<AST>`.
  - It will then create a Rust `Situation` whose closure calls the captured Rhai function.

  ```rust
  // In TeamController or a shared BT context
  // thread_local! {
  // static RHAI_CONTEXT: RefCell<Option<(Arc<Engine>, Arc<AST>)>> = RefCell::new(None);
  // }
  // This context would be set before calling build_tree and cleared after.

  // Rust function registered as "Guard" for Rhai
  fn rhai_guard_constructor(
      context: NativeCallContext, // To get current Engine/AST
      condition_fn_ptr: FnPtr,
      child_dyn: Dynamic,
      cond_description: Option<String> // Optional description for the condition itself
  ) -> Result<RhaiBehaviorNode, Box<EvalAltResult>> {
      let child_wrapper = child_dyn.try_cast::<RhaiBehaviorNode>().ok_or_else(/* error */)?;

      let engine = context.engine().clone(); // Arc<Engine> if Engine is Arc-wrapped or cloneable
      let ast = context.ast().clone(); // Arc<AST> if AST is Arc-wrapped or cloneable
                                       // Note: NativeCallContext::engine() returns &Engine,
                                       // ast() returns &AST. We need a way to share them or
                                       // have the RhaiSituationCallback hold them if they are Arc-wrapped.
                                       // For simplicity, assume engine and ast are Arc-wrapped and cloneable.

      let callback_fn_name = condition_fn_ptr.fn_name().to_string(); // Store name
      let description = cond_description.unwrap_or_else(|| format!("RhaiCond:{}", callback_fn_name));

      let actual_situation = Situation::new(
          move |rs: &RobotSituation| -> bool {
              let situation_view_dyn = create_rhai_situation_view(rs, &engine); // Pass engine for type registration if needed
              match engine.call_fn(&mut Scope::new(), &ast, &callback_fn_name, (situation_view_dyn,)) {
                  Ok(r) => r, // Expecting Rhai function to return bool
                  Err(e) => {
                      log::error!("Error executing Rhai condition '{}': {:?}", callback_fn_name, e);
                      false // Default to false on error
                  }
              }
          },
          &description,
      );
      let guard_desc_override = Some(format!("Guard_If_{}_Then_{}", description, child_wrapper.0.description()));


      Ok(RhaiBehaviorNode(Box::new(GuardNode::new(actual_situation, child_wrapper.0, guard_desc_override))))
  }
  // engine.register_fn("Guard", rhai_guard_constructor);
  ```

  - **Obtaining Engine/AST**: The `NativeCallContext` available to registered Rust functions can provide access to the current `Engine` and `AST`. These will need to be cloneable (e.g., wrapped in `Arc`) to be captured by the closure. Dies' `TeamController` might need to hold `Arc<Engine>` and `Arc<AST>` for the current script.

### 3.4. Handling Scoring Functions in Rhai

This follows a similar pattern to `Situation` for `ScoringSelectNode`.

- **Rhai Scorer Function Signature**: `fn my_scorer_func(situation_view) -> float`.
- **`ScoringSelectNode` Constructor in Rhai**:

  - Input: `children_scorers: Array` where each element is a map like `#{ node: BehaviorNode, scorer: FnPtr }`.
  - The Rust implementation (`rhai_scoring_select_constructor`) will iterate this array, create `RhaiScorerCallback` instances (similar to `RhaiSituationCallback`) for each scorer `FnPtr`, and build the `Vec<(Box<dyn BehaviorNode>, Box<dyn Fn(&RobotSituation) -> f64>)>` required by the Rust `ScoringSelectNode`.

  ```rust
  // Rust function registered as "ScoringSelect" in Rhai
  fn rhai_scoring_select_node(
      context: NativeCallContext,
      children_scorers_dyn: rhai::Array,
      hysteresis_margin: f64,
      description: Option<String>
  ) -> Result<RhaiBehaviorNode, Box<EvalAltResult>> {
      let engine = context.engine().clone();
      let ast = context.ast().clone();
      let mut rust_children_scorers = Vec::new();

      for item_dyn in children_scorers_dyn {
          let map = item_dyn.try_cast::<rhai::Map>().ok_or_else(/* error: not a map */)?;
          let node_dyn = map.get("node").ok_or_else(/* error: missing node */)?.clone();
          let scorer_fn_ptr = map.get("scorer").ok_or_else(/* error: missing scorer */)?.clone().cast::<FnPtr>();

          let node_wrapper = node_dyn.try_cast::<RhaiBehaviorNode>().ok_or_else(/* error */)?;

          let callback_fn_name = scorer_fn_ptr.fn_name().to_string();
          // Similar closure as in GuardNode, but returns f64
          let scorer_closure = move |rs: &RobotSituation| -> f64 {
              let situation_view_dyn = create_rhai_situation_view(rs, &engine);
              match engine.call_fn(&mut Scope::new(), &ast, &callback_fn_name, (situation_view_dyn,)) {
                  Ok(r) => r, // Expecting Rhai function to return float
                  Err(e) => {
                      log::error!("Error executing Rhai scorer '{}': {:?}", callback_fn_name, e);
                      f64::NEG_INFINITY // Default to very low score on error
                  }
              }
          };
          rust_children_scorers.push((node_wrapper.0, Box::new(scorer_closure) as Box<dyn Fn(&RobotSituation) -> f64>));
      }
      Ok(RhaiBehaviorNode(Box::new(ScoringSelectNode::new(rust_children_scorers, hysteresis_margin, description))))
  }
  // engine.register_fn("ScoringSelect", rhai_scoring_select_node);
  ```

### 3.5. Rhai Entry Point

- A Rhai script (e.g., `standard_player_tree.rhai`) will define a function:

  ```rhai
  // standard_player_tree.rhai
  fn build_player_bt(player_id_str) {
      // Use player_id_str if different roles need structurally different trees.
      // Or, more commonly, the role/ID influences conditions/scorers.
      let tree_description = "Standard Player Tree for " + player_id_str;

      return Select([ // Root node
          // Example sequence: If ball is close, try to get it and shoot
          Sequence([
              Guard(IsBallCloseCondition, Action(GoToBallSkill())), // IsBallCloseCondition and GoToBallSkill defined in Rhai/Rust
              Action(ShootSkill())
          ]),
          // Example: Scoring select for positioning
          ScoringSelect([
              #{ node: Action(PositionStrategicallyA()), scorer: ScorerForA },
              #{ node: Action(PositionStrategicallyB()), scorer: ScorerForB }
          ], 0.1),
          // Fallback action
          Action(StayIdleSkill())
      ], tree_description);
  }

  // Example condition (Rhai lambda or named function)
  fn IsBallCloseCondition(s) {
      // Assuming s.ball_distance is available in the situation_view passed to Rhai
      return s.ball_distance < 0.5;
  }
  // Example scorer
  fn ScorerForA(s) {
      return s.strategic_value_A;
  }
  // ... and so on for other skills/conditions/scorers
  ```

- **Rust Host Call**:

  - The `TeamController` will load this script once.
  - For each robot, it will call `build_player_bt` using `engine.call_fn()`.

  ```rust
  // In TeamController::update or similar
  // Assume 'self.rhai_engine' and 'self.main_bt_ast' are available and Arc-wrapped
  // Assume `self.player_behavior_trees` is a `HashMap<PlayerId, Box<dyn BehaviorNode>>`

  let player_id = player_data.id;
  let player_id_str = player_id.to_string(); // Or appropriate representation

  if !self.player_behavior_trees.contains_key(&player_id) {
      // Create a new scope for each call to avoid interference, though for tree building it might be fine.
      let mut scope = Scope::new();

      match self.rhai_engine.call_fn::<RhaiBehaviorNode>(&mut scope, &self.main_bt_ast, "build_player_bt", (player_id_str.clone(),)) {
          Ok(node_wrapper) => {
              let tree: Box<dyn BehaviorNode> = node_wrapper.0;
              self.player_behavior_trees.insert(player_id, tree);
          }
          Err(e) => {
              log::error!("Failed to build BT for player {}: {:?}. Falling back to default.", player_id_str, e);
              // Fallback to a default safe tree or handle error, then insert it.
              // let default_tree = build_default_fallback_tree();
              // self.player_behavior_trees.insert(player_id, default_tree);
          }
      }
  }

  // Retrieve and use the tree for the current player for this tick
  // let current_player_tree = self.player_behavior_trees.get(&player_id).unwrap(); // Assuming it's always there after the above logic
  // current_player_tree.tick(world, player_data, team_context);
  ```

### 3.6. Logging Integration

Redirect Rhai's `print` and `debug` to Rust's `log` crate using `Engine::on_print` and `Engine::on_debug`.

```rust
// During Engine setup
engine.on_print(|text| log::info!("[RHAI SCRIPT] {}", text));
engine.on_debug(|text, source, pos| {
    let src_info = source.map_or_else(String::new, |s| format!(" in '{}'", s));
    let pos_info = if pos.is_none() { String::new() } else { format!(" @ {}", pos) };
    log::debug!("[RHAI SCRIPT DEBUG]{}{}: {}", src_info, pos_info, text);
});
```

Reference from Rhai book:

> Override `print` and `debug` with Callback Functions
>
> When embedding Rhai into an application, it is usually necessary to trap `print` and `debug` output (for logging into a tracking log, for example) with the `Engine::on_print` and `Engine::on_debug` methods.

### 3.7. Hot Reloading (Lower Priority)

- **Mechanism**:
  1.  The Rust host will monitor the Rhai script file(s) for changes.
  2.  Upon detecting a change, the script will be recompiled into a new `AST`.
  3.  The `TeamController` will then replace its stored `AST` (e.g., `self.main_bt_ast = Arc::new(new_ast);`).
  4.  After the `AST` is swapped, the `TeamController` should invalidate existing compiled trees for all players (e.g., by clearing the `player_behavior_trees` map or marking entries as stale). The next time a tree is needed for a player (typically on their next update cycle), it will be rebuilt using the new `AST` as per the logic in section 3.5. Existing `player_id_holding_via_this_node` in `SemaphoreNode` and `current_best_child_index` in `ScoringSelectNode` might need careful handling or resetting when a tree is rebuilt.
- **Rhai Support**: Rhai's `Engine::compile_file` can be used for recompilation. ASTs are cloneable and can be replaced.
  Reference from Rhai book on Hot Reloading:
  > Hot reload entire script upon change
  >
  > If the control scripts are small enough and changes are infrequent, it is much simpler just to recompile the whole set of script and replace the original `AST` with the new one.
  >
  > ```rust
  > // Watch for script file change
  > system.watch(|sys: &System, file: &str| {
  >     // Compile the new script
  >     let ast = sys.engine.compile_file(file.into())?;
  >
  >     // Hot reload - just replace the old script!
  >     *sys.script.borrow_mut() = ast;
  >
  >     Ok(())
  > });
  > ```

## 4. Rhai Functionality Usage Summary

- **`Engine`**: Core of the scripting environment. Methods like `new()`, `register_fn()`, `register_type_with_name()`, `compile_file()`, `call_fn()`, `on_print()`, `on_debug()`.
- **Custom Types**: `RhaiBehaviorNode` and `RhaiSkill` to bridge Rust types with Rhai. `RhaiRobotSituationView` for exposing context.
- **`FnPtr`**: To pass Rhai functions (lambdas/named) as arguments to node constructors (e.g., for conditions and scorers).
- **`NativeCallContext`**: Used within Rust functions registered to Rhai (like node constructors) to get access to the current `Engine` and `AST`, enabling the creation of callbacks that can execute Rhai code.
- **`rhai::Array`**: For passing lists of children nodes or scorer entries.
- **`rhai::Map`**: For `RhaiRobotSituationView` if not a custom type, or for structured arguments like scorer entries (`#{ node: ..., scorer: ... }`).
- **Error Handling**: `EvalAltResult` and `Result<T, Box<EvalAltResult>>` for fallible operations.
- **Lambdas/Closures**: Rhai's anonymous functions `|args| { body }` for concise conditions/scorers.

## 5. Example Rhai Script Snippet (Conceptual)

```rhai
// --- registered_types.rhai (conceptual, shows type names Rhai would see) ---
// type BehaviorNode;
// type RhaiSkill;
// type PlayerSituationView; // Proxy for RobotSituation

// --- skills_constructors.rhai (conceptual, shows how Rust skills are exposed) ---
// fn GoToSkill(x, y) -> RhaiSkill;
// fn ShootSkill(target_x, target_y) -> RhaiSkill;
// fn DefendGoalSkill() -> RhaiSkill;

// --- bt_script.rhai ---
fn IsBallOurs(ctx) {
    return ctx.world.ball_owner == OUR_TEAM_ID; // Example, actual fields depend on RhaiRobotSituationView
}

fn IsNearOurGoal(ctx) {
    let goal_pos = OUR_GOAL_CENTER; // Assume this is a global constant or accessible via ctx
    return distance(ctx.player.pos, goal_pos) < 2.0;
}

fn ScoreDefensePosition(ctx) {
    // Complex scoring logic based on ball, opponents, player position etc.
    let score = 100.0;
    score -= distance(ctx.player.pos, OUR_GOAL_CENTER) * 10.0;
    if ctx.world.ball_is_threatening {
        score += 50.0;
    }
    return score;
}

fn build_tree(player_id_str) {
    let default_seek_ball = Action(GoToSkill(0.0, 0.0)); // Go to ball default (if skill allows dynamic target)

    let defend_sequence = Sequence([
        Guard(IsNearOurGoal, Action(DefendGoalSkill())),
        Action(ClearBallSkill()) // Another example skill
    ]);

    let attack_sequence = Sequence([
        Guard(IsBallOurs, Action(GoToOpponentGoalSkill())),
        Action(ShootSkill(OPP_GOAL_X, OPP_GOAL_Y))
    ]);

    let position_scorer = ScoringSelect([
        #{ node: Action(DefensiveStanceA()), scorer: ScoreDefensePosition },
        #{ node: Action(MidfieldStance()), scorer: |ctx| { return 50.0 - ctx.player.dist_to_center; } } // Lambda scorer
    ], 0.2, "StrategicPositioning");

    return Select([
        defend_sequence,
        attack_sequence,
        position_scorer,
        default_seek_ball
    ], "PlayerTree_" + player_id_str);
}
```
