# Getting Started: Your First Behavior Tree

This guide will walk you through creating a simple behavior tree script.

## The Entry Point

All behavior tree scripts must define an entry point function called `build_player_bt`. This function takes a `player_id` as an argument and must return a `BehaviorNode`.

```rust
// In standard_player_tree.rhai

fn build_player_bt(player_id) {
    // Return a BehaviorNode
}
```

## Example 1: The Simplest Action

Let's start with the most basic tree: a single action. We'll make the robot fetch the ball.

```rust
// standard_player_tree.rhai
fn build_player_bt(player_id) {
    // This tree has only one node: an ActionNode that executes the "FetchBall" skill.
    return FetchBall();
}
```

With this script, every robot will simply try to fetch the ball, no matter the situation.

## Example 2: A Sequence of Actions

A more useful behavior might involve a sequence of actions. For example, fetch the ball, then turn towards the opponent's goal, and then kick. We can achieve this with a `Sequence` node.

```rust
// standard_player_tree.rhai
fn build_player_bt(player_id) {
    return Sequence([
        FetchBall(),
        FaceTowardsPosition(6000.0, 0.0), // Assuming opponent goal is at (6000, 0)
        Kick()
    ]);
}
```

A `Sequence` node executes its children in order. It will only proceed to the next child if the previous one returns `Success`. If any child `Fail`s, the sequence stops and fails. If a child is `Running`, the sequence is also `Running`.

## Example 3: Conditional Behavior with Guards

Now, let's make the behavior conditional. We only want to kick if we actually have the ball. We can use a `Guard` node for this. A `Guard` takes a condition function and a child node. It only executes the child if the condition is true.

```rust
// standard_player_tree.rhai

// A condition function. It receives the 'RobotSituation' object.
fn i_have_ball(s) {
    // The 's' object gives access to the robot's state.
    // The 'has_ball()' method checks if the robot's breakbeam sensor detects the ball.
    // NOTE: The exact API of the situation object is subject to change.
    // For now, we assume 'has_ball()' is available.
    return s.has_ball();
}

fn build_player_bt(player_id) {
    return Select([
        // This branch is for when we have the ball
        Sequence([
            // This Guard ensures the rest of the sequence only runs if we have the ball.
            Guard(i_have_ball, "Do I have the ball?"),
            FaceTowardsPosition(6000.0, 0.0, "Face opponent goal"),
            Kick("Kick!"),
        ]),

        // This is the fallback branch if the first one fails (i.e., we don't have the ball)
        FetchBall("Get the ball"),
    ]);
}
```

This tree uses a `Select` node. A `Select` node tries its children in order until one succeeds.

1.  It first tries the `Sequence`.
2.  The `Sequence` starts with a `Guard`. If `i_have_ball` returns `false`, the `Guard` fails, which makes the `Sequence` fail.
3.  The `Select` node then moves to its next child, which is `FetchBall()`.
4.  If `i_have_ball` returns `true`, the `Guard` succeeds, and the `Sequence` continues to face the goal and kick. If the `Sequence` succeeds, the `Select` node also succeeds and stops.

This creates a simple but effective offensive logic. You can now build upon these concepts to create more complex and intelligent behaviors.
