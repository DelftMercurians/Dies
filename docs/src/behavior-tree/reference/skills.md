# Skills (Action Nodes)

Skills are the leaf nodes of a behavior tree that perform concrete actions. In our system, calling a skill function in Rhai creates an `ActionNode`.

---

### `GoToPosition`

Moves the robot to a target position.

**Syntax:**
`GoToPosition(x: Float, y: Float, [options: Map], [description: String]) -> BehaviorNode`

**Parameters:**

- `x`, `y`: The target coordinates.
- `options` (optional): A map with the following keys:
  - `heading` (Float): The final heading of the robot in radians.
  - `with_ball` (Bool): If `true`, the robot will try to keep the ball while moving.
  - `avoid_ball` (Bool): If `true`, the robot will actively try to avoid the ball while moving.
- `description` (optional): A string description for debugging.

**Example:**

```rust
GoToPosition(0.0, 0.0, #{ heading: 1.57, with_ball: true }, "Go to center with ball")
```

---

### `FaceAngle`

Rotates the robot to face a specific angle.

**Syntax:**
`FaceAngle(angle_rad: Float, [options: Map], [description: String]) -> BehaviorNode`

**Parameters:**

- `angle_rad`: The target angle in radians.
- `options` (optional): A map with the following keys:
  - `with_ball` (Bool): If `true`, the robot will try to keep the ball while turning.
- `description` (optional): A string description for debugging.

---

### `FaceTowardsPosition`

Rotates the robot to face a specific world position.

**Syntax:**
`FaceTowardsPosition(x: Float, y: Float, [options: Map], [description: String]) -> BehaviorNode`

**Parameters:**

- `x`, `y`: The coordinates of the target position to face.
- `options` (optional): A map with `with_ball` (Bool).
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
