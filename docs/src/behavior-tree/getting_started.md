# Getting Started: Your First Behavior Tree

This guide will walk you through creating a simple behavior tree script.

## The Entry Point

All behavior tree scripts must define entry point functions for different game states. The main entry points are `build_play_bt`, `build_kickoff_bt`, and `build_penalty_bt`. These functions take a `player_id` as an argument and must return a `BehaviorNode`.

```rust
// In strategies/main.rhai

fn build_play_bt(player_id) {
    // Return a BehaviorNode for normal gameplay
}

fn build_kickoff_bt(player_id) {
    // Return a BehaviorNode for kickoff situations
}

fn build_penalty_bt(player_id) {
    // Return a BehaviorNode for penalty situations
}

// Legacy entry point for backward compatibility
fn build_player_bt(player_id) {
    return build_play_bt(player_id);
}
```

## Example 1: The Simplest Action

Let's start with the most basic tree: a single action. We'll make the robot fetch the ball.

```rust
// strategies/main.rhai
fn build_play_bt(player_id) {
    // This tree has only one node: an ActionNode that executes the "FetchBall" skill.
    return FetchBall();
}

// Legacy entry point
fn build_player_bt(player_id) {
    return build_play_bt(player_id);
}
```

With this script, every robot will simply try to fetch the ball, no matter the situation.

## Example 2: A Sequence of Actions

A more useful behavior might involve a sequence of actions. For example, fetch the ball, then turn towards the opponent's goal, and then kick. We can achieve this with a `Sequence` node.

```rust
// strategies/main.rhai
fn build_play_bt(player_id) {
    return Sequence([
        FetchBall(),
        FaceTowardsPosition(vec2(6000.0, 0.0)), // Assuming opponent goal is at (6000, 0)
        Kick()
    ]);
}

// Legacy entry point
fn build_player_bt(player_id) {
    return build_play_bt(player_id);
}
```

A `Sequence` node executes its children in order. It will only proceed to the next child if the previous one returns `Success`. If any child `Fail`s, the sequence stops and fails. If a child is `Running`, the sequence is also `Running`.

## Example 3: Conditional Behavior with Guards

Now, let's make the behavior conditional. We only want to kick if we actually have the ball. We can use a `Guard` node for this. A `Guard` takes a condition function and a child node. It only executes the child if the condition is true.

```rust
// strategies/main.rhai

import "shared/situations" as sit;
import "shared/utilities" as util;

fn build_play_bt(player_id) {
    return Select([
        // This branch is for when we have the ball
        Sequence([
            // This Guard ensures the rest of the sequence only runs if we have the ball.
            Guard(sit::i_have_ball,
                Sequence([
                    FaceTowardsPosition(util::get_opponent_goal(), #{}, "Face opponent goal"),
                    Kick("Kick!")
                ], "Ball Actions"),
                "Do I have the ball?"
            ),
        ], "Have Ball Sequence"),

        // This is the fallback branch if the first one fails (i.e., we don't have the ball)
        FetchBall("Get the ball"),
    ]);
}

// Legacy entry point
fn build_player_bt(player_id) {
    return build_play_bt(player_id);
}
```

This example shows how the modular system works:

- Condition functions are imported from `shared/situations`
- Utility functions are imported from `shared/utilities`
- The code is much cleaner and more organized

This tree uses a `Select` node. A `Select` node tries its children in order until one succeeds.

1.  It first tries the `Sequence`.
2.  The `Sequence` starts with a `Guard`. If `i_have_ball` returns `false`, the `Guard` fails, which makes the `Sequence` fail.
3.  The `Select` node then moves to its next child, which is `FetchBall()`.
4.  If `i_have_ball` returns `true`, the `Guard` succeeds, and the `Sequence` continues to face the goal and kick. If the `Sequence` succeeds, the `Select` node also succeeds and stops.

This creates a simple but effective offensive logic. You can now build upon these concepts to create more complex and intelligent behaviors.
