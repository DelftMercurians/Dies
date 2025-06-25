# Skills (Action Nodes)

Skills are the leaf nodes of a behavior tree that perform concrete actions. In our system, calling a skill function in Rhai creates an `ActionNode`, which is a type of `BehaviorNode`.

## Dynamic Arguments

Many skills accept **dynamic arguments**. This means that instead of providing a fixed, static value, you can provide a callback function. This function will be executed by the behavior tree on each tick to determine the value for that argument dynamically.

A callback function always receives the `RobotSituation` object (usually named `s`) as its only parameter, giving you access to the full world state to make decisions.

**Example:**

```rust
// Static argument
GoToPosition(vec2(100.0, 200.0))

// Dynamic argument using a callback
fn get_ball_pos(s) {
    return s.world.ball.position;
}
GoToPosition(get_ball_pos)
```

This allows for creating highly reactive and flexible behaviors.

---

### `GoToPosition`

Moves the robot to a target position.

**Syntax:**
`GoToPosition(target: Vec2 | FnPtr, [options: Map], [description: String]) -> BehaviorNode`

**Parameters:**

- `target`: The target position. Can be a static `Vec2` (created with `vec2(x, y)`) or a callback function that returns a `Vec2`.
- `options` (optional): A map with optional parameters. Any of these can also be dynamic callbacks.
  - `heading` (Float | FnPtr): The final heading of the robot in radians.
  - `with_ball` (Bool | FnPtr): If `true`, the robot will try to keep the ball while moving.
  - `avoid_ball` (Bool | FnPtr): If `true`, the robot will actively try to avoid the ball while moving.
- `description` (optional): A string description for debugging.

**Example (Static):**

```rust
// Go to a fixed position with a fixed heading.
GoToPosition(
    vec2(0.0, 0.0),
    #{ heading: 1.57, with_ball: true },
    "Go to center with ball"
)
```

**Example (Dynamic):**

```rust
fn get_ball_pos(s) {
    return s.world.ball.position;
}

fn get_heading_towards_goal(s) {
    let goal_pos = vec2(6000.0, 0.0);
    // Assumes a 'angle_to' helper exists
    return s.player.position.angle_to(goal_pos);
}

// Go towards the ball, while always facing the opponent's goal.
GoToPosition(
    get_ball_pos,
    #{ heading: get_heading_towards_goal, avoid_ball: false },
    "Follow ball while facing goal"
)
```

---

### `FaceAngle`

Rotates the robot to face a specific angle.

**Syntax:**
`FaceAngle(angle: Float | FnPtr, [options: Map], [description: String]) -> BehaviorNode`

**Parameters:**

- `angle`: The target angle in radians. Can be a static `Float` or a callback function that returns a `Float`.
- `options` (optional): A map with the following keys:
  - `with_ball` (Bool | FnPtr): If `true`, the robot will try to keep the ball while turning.
- `description` (optional): A string description for debugging.

---

### `FaceTowardsPosition`

Rotates the robot to face a specific world position.

**Syntax:**
`FaceTowardsPosition(target: Vec2 | FnPtr, [options: Map], [description: String]) -> BehaviorNode`

**Parameters:**

- `target`: The target position to face. Can be a static `Vec2` or a callback function that returns a `Vec2`.
- `options` (optional): A map with `with_ball` (Bool | FnPtr).
- `description` (optional): A string description for debugging.

---

### `FaceTowardsOwnPlayer`

Rotates the robot to face a teammate.

**Syntax:**
`FaceTowardsOwnPlayer(player_id: Int, [options: Map], [description: String]) -> BehaviorNode`

**Parameters:**

- `player_id`: The ID of the teammate to face.
- `options` (optional): A map with `with_ball` (Bool).
- `description` (optional): A string description for debugging.

---

### `Kick`

Kicks the ball. This skill assumes the robot has the ball.

**Syntax:**
`Kick([description: String]) -> BehaviorNode`

**Parameters:**

- `description` (optional): A string description for debugging.

---

### `Wait`

Makes the robot wait for a specified duration.

**Syntax:**
`Wait(duration_secs: Float, [description: String]) -> BehaviorNode`

**Parameters:**

- `duration_secs`: The duration to wait in seconds.
- `description` (optional): A string description for debugging.

---

### `FetchBall`

Moves the robot to the ball to capture it.

**Syntax:**
`FetchBall([description: String]) -> BehaviorNode`

**Parameters:**

- `description` (optional): A string description for debugging.

---

### `InterceptBall`

Moves the robot to intercept a moving ball.

**Syntax:**
`InterceptBall([description: String]) -> BehaviorNode`

**Parameters:**

- `description` (optional): A string description for debugging.

---

### `ApproachBall`

Moves the robot close to the ball without necessarily capturing it.

**Syntax:**
`ApproachBall([description: String]) -> BehaviorNode`

**Parameters:**

- `description` (optional): A string description for debugging.

---

### `FetchBallWithHeadingAngle`

Moves to the ball and aligns the robot to a specific angle after capturing it.

**Syntax:**
`FetchBallWithHeadingAngle(angle_rad: Float, [description: String]) -> BehaviorNode`

**Parameters:**

- `angle_rad`: The target angle in radians after fetching the ball.
- `description` (optional): A string description for debugging.

---

### `FetchBallWithHeadingPosition`

Moves to the ball and aligns the robot to face a specific position after capturing it.

**Syntax:**
`FetchBallWithHeadingPosition(x: Float, y: Float, [description: String]) -> BehaviorNode`

**Parameters:**

- `x`, `y`: The coordinates of the target position to face.
- `description` (optional): A string description for debugging.

---

### `FetchBallWithHeadingPlayer`

Moves to the ball and aligns the robot to face a teammate after capturing it.

**Syntax:**
`FetchBallWithHeadingPlayer(player_id: Int, [description: String]) -> BehaviorNode`

**Parameters:**

- `player_id`: The ID of the teammate to face.
- `description` (optional): A string description for debugging.
