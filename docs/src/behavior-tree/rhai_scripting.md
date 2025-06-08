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
