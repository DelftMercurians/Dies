# Project Dies: Strategy System Evolution Plan

This document outlines a comprehensive plan for enhancing the strategic capabilities of the Dies robotic soccer team. It details proposed implementations for advanced defensive and offensive strategies, a new 'Plays' system for coordination, and seamless free-kick handling, all while adhering to the core design principle of simplicity and dynamic adaptation.

## 1. Advanced Defensive Strategy

To create a more robust and adaptive defense, we will introduce specialized defensive roles and enhance our system's ability to coordinate targeting.

### 1.1. Dynamic 'Waller' and 'Harasser' Roles via Role Assignment

Instead of a single "defender" behavior tree that internally switches between walling and harassing, we will define these as two distinct roles. This approach keeps the individual behavior trees simple and delegates the selection logic to the strengths of the role assignment engine.

**Implementation via `main.rhai`:**

We will add `waller` and `harasser` as potential roles in the main strategy file. The role assignment solver will then dynamically assign defenders to the most appropriate role based on the game situation, evaluated through scorer functions.

```rhai
// strategies/main.rhai

fn main(game) {
    // ... existing goalkeeper and other roles ...

    // Add dynamic defender roles. A single robot can be either a waller or harasser.
    game.add_role("waller")
        .min(0).max(2) // Allow 0-2 wallers
        .score(|s| util::score_as_waller(s))
        .exclude(|s| s.player_id == 0) // Goalkeeper cannot be a waller
        .behavior(|| trees::build_waller_tree())
        .build();

    game.add_role("harasser")
        .min(0).max(2) // Allow 0-2 harassers
        .score(|s| util::score_as_harasser(s))
        .exclude(|s| s.player_id == 0) // Goalkeeper cannot be a harasser
        .behavior(|| trees::build_harasser_tree())
        .build();
}
```

**New Scorer Functions (in `strategies/shared/utilities.rhai`):**

- `score_as_waller(s)`: This function will return a high score if the ball is in a central, threatening position. The score is higher if the robot is already positioned between the ball and the goal.
- `score_as_harasser(s)`: This function will return a high score if there are un-marked opponent attackers in our half. The score will be highest for the defender closest to the highest-threat, un-marked opponent.

### 1.2. Dynamic Target Tagging via Enhanced Composite Nodes

The core challenge with dynamically tagging opponents is that the set of targets changes on every tick, while a behavior tree's structure is typically static.

The ideal solution is to make our composite nodes more powerful, allowing them to dynamically generate their children on each tick.

#### The Approach: Dynamic Child Generation

We will enhance our `Select` and `ScoringSelect` nodes so they can accept a Rhai callback function instead of a static list of children. This callback will be executed on every tick, returning a fresh list of child nodes to evaluate.

For `ScoringSelect`, which uses hysteresis to prevent oscillation, we will require the dynamic callback to provide a stable `key` for each child. This allows the node to track the previously selected child even if its position in the list changes.

#### Implementation Outline (Rust Core)

1.  **Modify `select_node_impl` and `scoring_select_node_impl` in `rhai_plugin.rs`**: These functions will be overloaded to accept a `FnPtr` (function pointer) as an alternative to an `Array`.
2.  **Update `SelectNode`**: It will hold an `enum ChildrenSource { Static(Vec<...>), Dynamic(BtCallback<...>) }`. The `tick` method will resolve this source to get the list of children before executing its logic.
3.  **Update `ScoringSelectNode`**:
    - Its `ChildrenSource` enum will expect the dynamic callback to return an array of maps, where each map must now contain `node`, `scorer`, and a new `key: String`.
    - The node's internal state will change from `current_best_child_index: Option<usize>` to `current_best_child_key: Option<String>`.
    - The `tick` method will:
      a. Call the callback to get the fresh list of `{node, scorer, key}` maps.
      b. Find the candidate with the highest score.
      c. If a `current_best_child_key` exists, find the corresponding child in the _new_ list by its key to get its current score.
      d. Apply hysteresis logic using the key-based lookup.
      e. If the best child changes, store its `key` in `current_best_child_key`.

This approach keeps our core nodes generic and powerful, moving all the strategic list-generation logic into the Rhai scripts where it belongs.

#### Example: Dynamic Harasser Tree

With these changes, the `build_harasser_tree` becomes incredibly expressive and fully dynamic, gracefully handling any number of opponents.

```rhai
// In strategies/shared/subtrees.rhai

fn build_harasser_tree() {
    // Use ScoringSelect with a dynamic callback.
    return ScoringSelect(
        // 1. Find all opponents who are threats (e.g., in our half).
        // 2. Map each threat to a potential behavior branch.
        |s|
            s.filter_opp_players_by(|p| p.position.x < 0).map(|opponent| #{
                // The behavior is to try and claim this opponent via a semaphore.
                node: Semaphore(
                    GoToPosition(opponent.position, #{}, "Harass Opponent"),
                    "tag_opponent_" + opponent.id.to_string(), // Dynamic semaphore ID
                    1 // Only one player can claim this tag.
                ),

                // The score for this branch is how threatening this opponent is.
                scorer: |s| util::score_opponent_threat(s, opponent),

                // The stable key for hysteresis tracking.
                key: "harass_" + opponent.id.to_string()
            }),
        0.1, // Hysteresis margin
        "Dynamic Harasser"
    );
}
```

**How it works:**

1.  On every tick, the `ScoringSelect` executes the callback, getting a list of harassment options, one for each current opponent on our side of the field.
2.  It scores all options. `score_opponent_threat` would rank opponents based on their proximity to our goal and if they have the ball.
3.  The node applies its hysteresis logic using the stable `key` (e.g., `"harass_3"`) to see if it should switch from its currently marked opponent.
4.  It then ticks the highest-scoring, valid child. The `Semaphore` inside that child attempts to acquire the lock. If another defender has already tagged `opponent_3` on this tick, the semaphore will fail, and `ScoringSelect` would move to the next-best option automatically.

This elegant solution provides fully dynamic, multi-robot target selection with minimal changes to the core engine, keeping the power in the hands of the strategy scripter.

## 2. Advanced Attacker Strategy

To improve offensive capabilities, strikers need smarter positioning logic and a better understanding of their options.

### 2.1. Dynamic Field Section Allocation for Strikers

To ensure strikers spread out and cover the field effectively, we can use the same semaphore pattern to dynamically allocate sections of the opponent's half to different strikers.

**Implementation:**

The attacker's behavior tree will use a `ScoringSelect` to choose a field section (`top`, `middle`, `bottom`). The chosen behavior will be guarded by a `Semaphore` for that section, ensuring only one striker operates there.

```rhai
// In build_striker_tree() in strategies/shared/subtrees.rhai

ScoringSelect([
    #{
        node: Semaphore(OperateInZone("top"), "striker_zone_top", 1),
        scorer: |s| util::score_for_zone(s, "top")
    },
    #{
        node: Semaphore(OperateInZone("middle"), "striker_zone_middle", 1),
        scorer: |s| util::score_for_zone(s, "middle")
    },
    #{
        node: Semaphore(OperateInZone("bottom"), "striker_zone_bottom", 1),
        scorer: |s| util::score_for_zone(s, "bottom")
    }
], 0.1, "Choose Attacker Zone")
```

### 2.2. New Helper Functions for Attacker AI

We will implement a suite of new helper functions in `strategies/shared/utilities.rhai` to empower the attacker AI.

- `find_optimal_shot_target(s)`

  - **Goal**: Find the best point on the opponent's goal line to aim at.
  - **Logic**:
    1.  Get the opponent goalkeeper's position.
    2.  Get the positions of any opponent defenders near the goal.
    3.  Project the "shadow" of the goalie and defenders onto the goal line.
    4.  Identify the largest clear segment on the goal line.
    5.  Return the center of this largest segment as the target `Vec2`.

- `evaluate_shot_quality(s, from_pos, to_pos)`

  - **Goal**: Return a score (0.0 to 1.0) indicating how clear a shot path is.
  - **Logic**:
    1.  Use `s.cast_ray(from_pos, to_pos)` to check for direct obstructions.
    2.  To make it more robust, cast multiple rays from `from_pos` to a small area around `to_pos` (e.g., a 10cm radius).
    3.  The quality score is the percentage of rays that are not blocked by opponent players.

- `find_optimal_striker_position(s, zone)`

  - **Goal**: Find the best position for a striker within their allocated `zone` to receive a pass and prepare for a shot.
  - **Logic**:
    1.  Sample a grid of points within the specified `zone`.
    2.  For each point, calculate a score:
        - `pass_reception_score`: Quality of the line-of-sight to the ball (or primary teammate with the ball). Use `s.cast_ray`.
        - `shot_potential_score`: The `evaluate_shot_quality` from this point to the `find_optimal_shot_target`.
    3.  Return the point with the highest combined score.

- `score_for_shooting(s)` and `score_for_passing(s)`
  - **Goal**: Used in a `ScoringSelect` to help a striker decide whether to shoot or pass.
  - **`score_for_shooting` Logic**:
    - Returns `evaluate_shot_quality` from the robot's current position.
    - Score is higher if closer to the goal.
    - Score is zero if the robot does not have the ball.
  - **`score_for_passing` Logic**:
    - Finds the best potential pass recipient (a teammate in a good offensive position).
    - Uses `evaluate_shot_quality` to check if the pass path is clear.
    - The score is based on the quality of the pass path and the quality of the recipient's potential shot after receiving the pass.
    - Includes a check for a "safe" backward pass to a defender if under heavy pressure.

## 3. Seamless Free Kick Handling

We will integrate free kick logic directly into the existing role-based system.

- **Offensive Free Kick**: When `game.game_state == "FreeKick"` and `game.us_operating == true`, `main.rhai` will dynamically add a `free_kicker` role. The `score_free_kicker(s)` function will heavily favor non-defensive players who are close to the ball.
- **Defensive Free Kick**: When `game.game_state == "FreeKick"` and `game.us_operating == false`, the defender behavior trees (`build_waller_tree` and `build_harasser_tree`) will contain a high-priority `Guard` that checks for this state. If true, they will execute a `GoToPosition` skill to move to a rule-compliant interference position (e.g., 500mm away from the ball).

## 4. "Plays" Feature: Detailed Requirements & Design

The "Plays" system provides a structured mechanism for short-term, multi-robot coordinated actions like passing.

### 4.1. Core Concepts & Goals

- **Goal**: To simplify the creation of complex, coordinated maneuvers without cluttering the main role-based behavior trees.
- **Initiation**: A "leader" robot initiates a Play.
- **Recruitment**: The Play recruits "follower" robots based on specified criteria.
- **Temporary Roles**: Recruited followers temporarily adopt a simple, specific behavior defined by the Play.
- **Lifecycle**: Plays are short-lived. Once the action is complete (or fails), all involved robots seamlessly return to their standard role-assigned behaviors.

### 4.2. Proposed Node API (for Rhai scripts)

- `StartPlay(play_name, recruitment_config, leader_subtree)`:

  - `play_name`: A string to identify the play, e.g., `"backward_pass"`.
  - `recruitment_config`: A map defining the sub-roles to recruit.
    - Example: `#{ "receiver": #{ count: 1, scorer: |s| score_as_receiver(s) } }`
  - `leader_subtree`: The behavior tree the initiating robot will execute.

- `JoinPlay(accept_condition_fn)`:
  - This node is placed at the root of a robot's main behavior tree.
  - It constantly scans for open recruitment slots in active plays.
  - `accept_condition_fn`: A function that returns `true` if the robot is currently able to join a play (e.g., the goalkeeper might reject plays if the ball is in its own penalty area). If the condition is met and the robot is selected for a play, it executes the play-specific subtree. Otherwise, this node fails instantly, and the robot executes its normal tree.

### 4.3. Core Engine Requirements (Rust)

- **`BtContext` Extension**: The `BtContext` struct will be extended with a `active_plays: Arc<RwLock<HashMap<String, Play>>>`.
- **`Play` Struct**: A new `Play` struct in Rust will contain:
  - The play's unique ID.
  - The leader's `PlayerId`.
  - A map of recruitment slots (sub-role name to `RecruitmentSlot` struct).
  - A map of sub-role names to their temporary `BehaviorNode` (the subtree they run).
- **`RecruitmentSlot` Struct**: Will contain the number of players needed, the list of assigned players, and the Rhai scorer function for this slot.
- **New Rust Nodes**: The `StartPlayNode` and `JoinPlayNode` will need to be implemented in Rust and exposed to Rhai. They will interact with the new `active_plays` map in the `BtContext`.

### 4.4. Example Usage: Backward Pass Play

**1. Attacker (Leader) under pressure initiates the play:**

```rhai
// In the attacker's BT

// This sequence is triggered when sit::under_pressure(s) is true
Sequence([
    // Start a "backward_pass" play
    StartPlay(
        "backward_pass",
        // We need to recruit one "receiver"
        #{
            "receiver": #{
                count: 1,
                // The best receiver is a defender behind me with a clear pass lane
                scorer: |s| util::score_defender_for_pass(s)
            }
        },
        // The leader's (my) behavior is to face the receiver and kick
        Sequence([
            FaceTowardsPosition(|s| s.get_play_teammate("receiver").position),
            Kick("Pass backward!")
        ])
    ),
    // After kicking, wait briefly for the play to resolve
    Wait(0.5)
])
```

**2. Defender (Follower) joins the play:**

```rhai
// At the root of the defender's main BT
Select([
    // Highest priority: check if I should join a play
    JoinPlay(|s| {
        // I will only join a play if the ball is not in our penalty area
        return !sit::ball_in_own_penalty_area(s);
    }),

    // If not joining a play, proceed with normal defender logic...
    build_defender_tree()
])
```

**How it works:**

1.  The attacker is pressured and executes `StartPlay`. A new "backward_pass" play is created in `BtContext` and begins recruiting for a "receiver".
2.  The `RoleAssignmentSolver` finds the best defender based on the `score_defender_for_pass` scorer.
3.  On the defender's next tick, its `JoinPlay` node sees it has been recruited. The `accept_condition` passes.
4.  The defender **does not** run its normal `build_defender_tree`. Instead, it runs the "receiver" subtree that was defined by the attacker (this subtree would be part of the `Play` object). This might be a simple `GoToPosition` to an optimal reception spot.
5.  The attacker, executing its `leader_subtree`, sees the defender has been assigned, faces them, and kicks.
6.  The play completes. The `Play` object is removed from `BtContext`.
7.  On the next tick, both robots' `JoinPlay` nodes fail, and they seamlessly resume their normal `attacker` and `defender` roles.

This design provides a powerful, clean, and scalable way to script complex team interactions.
