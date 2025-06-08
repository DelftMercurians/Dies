# Helpers & The Situation Object

This section covers utility functions and the `RobotSituation` object that provides world context to scripts.

## Helper Functions

### `vec2`

Creates a 2D vector. This is primarily for internal use but can be useful for clarity.

**Syntax:**
`vec2(x: Float, y: Float) -> Vec2`

---

### Player ID Helpers

These functions help in working with `PlayerId`'s.

#### `to_string`

Converts a `PlayerId` to its string representation.

**Syntax:**
`to_string(id: PlayerId) -> String`

#### `hash_float`

Returns a float between 0.0 and 1.0 based on the player's ID. This is a deterministic hash, meaning the same ID will always produce the same float. This is useful for inducing different behaviors for different players in a predictable way (e.g., distributing players on the field).

**Syntax:**
`hash_float(id: PlayerId) -> Float`

## The `RobotSituation` Object

When you define a condition function for a `Guard` or a scorer function for a `ScoringSelect`, it receives a single argument. This argument is an object that holds the current state of the world from the robot's perspective. It is a snapshot of the `RobotSituation` struct from Rust.

**Signature of callbacks:**

```rust
fn my_condition(situation) -> bool { ... }
fn my_scorer(situation) -> Float { ... }
```

### Accessing Data

You can access data from the `situation` object to make decisions.

> **Note on Implementation Discrepancy:** The original design for the scripting system specified that a simplified "view" of the `RobotSituation` would be created as a Rhai `Map` for scripts to use. However, the current implementation passes the Rust `RobotSituation` struct directly. For fields and methods of this struct to be accessible from Rhai, they need to be registered with the Rhai engine. This registration does not appear to be implemented yet.
>
> **Therefore, the API described below is based on the intended design and may not be fully functional until the necessary getters are registered in `rhai_plugin.rs`.**

Here are the key properties you can expect to access from the `situation` object:

- `situation.player_id`: The `PlayerId` of the current robot.
- `situation.has_ball`: A boolean (`true` or `false`) indicating if the robot's breakbeam sensor detects the ball.
- `situation.world`: An object containing the world data.
  - `situation.world.ball`: Information about the ball (e.g., `position`, `velocity`).
  - `situation.world.own_players`: An array of data for your teammates.
  - `situation.world.opp_players`: An array of data for opponents.
- `situation.player`: Data for the current robot (e.g., `position`, `velocity`).

**Example Usage in a Condition:**

```rust
fn is_ball_in_our_half(s) {
    // Access ball position from the world state
    if s.world.ball.position.x < 0.0 {
        return true;
    } else {
        return false;
    }
}

fn is_near_ball(s) {
    // Calculate distance between player and ball
    let dx = s.player.position.x - s.world.ball.position.x;
    let dy = s.player.position.y - s.world.ball.position.y;
    let dist_sq = dx*dx + dy*dy;

    return dist_sq < 500.0 * 500.0; // Is the robot within 500mm of the ball?
}
```

This powerful object allows you to create highly context-aware and dynamic behaviors.
