# Scripting with Rhai

To enable dynamic and flexible behavior definition, the Dies project uses the **Rhai** scripting language. This page provides an overview of the most relevant Rhai language features for writing behavior tree scripts.

## What is Rhai?

Rhai is a tiny, fast, and easy-to-use embedded scripting language for Rust. It features a syntax similar to a combination of JavaScript and Rust, and is designed for deep integration with a host Rust application.

While Rhai is a rich language, you only need to know a subset of its features to be productive in the Dies system.

## Core Language Features

Here are the most important language features you will use when writing BT scripts.

### Variables

Variables are declared with the `let` keyword. They are dynamically typed.

```rust
let x = 42;
let name = "droste";
let is_active = true;
```

### Functions

You can define your own functions using the `fn` keyword. These are essential for creating conditions for `Guard` nodes and scorers for `ScoringSelect` nodes.

```rust
fn my_function(a, b) {
    return a + b > 10;
}

// Functions can be called as you'd expect
let result = my_function(5, 6);
```

The last expression in a function is implicitly returned, so you can often omit the `return` keyword:

```rust
fn my_function(a, b) {
    a + b > 10
}
```

### Data Types

You'll primarily work with these data types:

- **Integers** (`-1`, `0`, `42`)
- **Floats** (`3.14`, `-0.5`)
- **Booleans** (`true`, `false`)
- **Strings** (`"hello"`, `"a description"`)
- **Arrays**: A list of items, enclosed in `[]`.
  ```rust
  let my_array = [1, 2, "three", true];
  ```
- **Maps**: A collection of key-value pairs, enclosed in `#{}`. These are used for `options` parameters in many skills and for defining scorers in `ScoringSelect`.
  ```rust
  let my_map = #{
      heading: 3.14,
      with_ball: false
  };
  ```

### Control Flow

Standard `if/else` statements are supported for conditional logic within your functions.

```rust
fn my_scorer(s) {
    let score = 100.0;
    if s.has_ball {
        score += 50.0;
    } else {
        score -= 20.0;
    }
    return score;
}
```

### Comments

Use `//` for single-line comments and `/* ... */` for block comments.

```rust
// This is a single line comment.

/*
  This is a
  multi-line
  comment.
*/
```

## How Rhai is Used in Dies

In Dies, we use Rhai to declaratively construct behavior trees. Instead of building the tree structure in Rust, we write a Rhai script that defines the tree. This script is then loaded and executed by the system to generate the behavior tree for each robot.

The core of this system is a Rhai script, typically located at `crates/dies-executor/src/bt_scripts/standard_player_tree.rhai`. This script must contain an entry-point function:

```rust
fn build_player_bt(player_id) {
    // ... script logic to build and return a BehaviorNode ...
}
```

This function is called for each robot, and it is expected to return a `BehaviorNode` which serves as the root of that robot's behavior tree.

## Rust vs. Rhai: Where to Write Logic?

A key design principle of the Dies behavior system is the separation of concerns between Rust and Rhai. Here's a guideline for what code belongs where:

### What to write in Rust (Skills)

Rust is used to implement the fundamental "building blocks" of robot behavior. These are called **Skills**. A skill should be:

- **Atomic**: It should represent a single, clear action (e.g., `Kick`, `FetchBall`, `GoToPosition`).
- **Reusable**: It should be generic enough to be used in many different contexts within the behavior tree.
- **Self-contained**: It should manage its own state (e.g., tracking whether a `GoToPosition` action is complete).
- **Parameterized**: It should be configurable via arguments (e.g., the target for `GoToPosition`).

Examples of good skills implemented in Rust are `Kick`, `FetchBall`, and `GoToPosition`. They are exposed to Rhai as functions that create Action Nodes.

### What to write in Rhai (Strategy)

Rhai is used to compose these building blocks into complex, high-level strategies. The Rhai script defines the "brain" of the robot by wiring skills together using behavior tree nodes like `Sequence`, `Select`, and `Guard`. Your Rhai script should focus on:

- **Decision-making**: Using `Select` and `Guard` nodes to choose the right action based on the game state (`RobotSituation`).
- **Orchestration**: Using `Sequence` to define a series of actions to achieve a goal.
- **Team Coordination**: Using `Semaphore` to coordinate behavior between multiple robots.
- **Dynamic Behavior**: Using callbacks to dynamically provide arguments to skills based on real-time world data.

The `standard_player_tree.rhai` script is a perfect example of this. It doesn't implement the _how_ of moving or kicking, but it defines the _when_ and _why_: _when_ to be an attacker, _when_ to support, what defines those roles, and how to transition between them.

### The Core Idea

> **Simple, reusable skills go into Rust. Complex, high-level behavior goes into Rhai.**

This separation allows strategists to rapidly iterate on high-level tactics in Rhai without needing to recompile the entire Rust application. Meanwhile, the core skills can be implemented and optimized in performant, reliable Rust code.
