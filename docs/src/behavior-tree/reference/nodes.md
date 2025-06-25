# Behavior Nodes

Behavior nodes are the fundamental building blocks for constructing the logic of a behavior tree. They control the flow of execution.

## Composite Nodes

Composite nodes have multiple children and execute them in a specific order.

### `Select`

A `Select` node, also known as a Fallback or Priority node, executes its children sequentially until one of them returns `Success` or `Running`.

- **Returns `Success`** if any child returns `Success`.
- **Returns `Running`** if any child returns `Running`.
- **Returns `Failure`** only if all children return `Failure`.

This is equivalent to a logical **OR**.

**Syntax:**

```rust
Select(children: Array, [description: String]) -> BehaviorNode
```

**Parameters:**

- `children`: An array of `BehaviorNode`s.
- `description` (optional): A string description for debugging.

**Example:**

```rust
Select([
    // Try to score if possible
    TryToScoreAGoal(),
    // Otherwise, pass to a teammate
    PassToTeammate(),
    // If all else fails, just hold the ball
    HoldBall(),
])
```

### `Sequence`

A `Sequence` node executes its children sequentially.

- **Returns `Failure`** if any child returns `Failure`.
- **Returns `Running`** if any child returns `Running`.
- **Returns `Success`** only if all children return `Success`.

This is equivalent to a logical **AND**.

**Syntax:**

```rust
Sequence(children: Array, [description: String]) -> BehaviorNode
```

**Parameters:**

- `children`: An array of `BehaviorNode`s.
- `description` (optional): A string description for debugging.

**Example:**

```rust
Sequence([
    FetchBall(),
    FaceTowardsPosition(6000.0, 0.0),
    Kick()
])
```

### `ScoringSelect`

A `ScoringSelect` node evaluates a score for each of its children on every tick and executes the child with the highest score. This is useful for dynamic decision-making where multiple options are viable and need to be weighed against each other. It includes a hysteresis margin to prevent rapid, oscillating switching between behaviors.

**Syntax:**

```rust
ScoringSelect(children_scorers: Array, hysteresis_margin: Float, [description: String]) -> BehaviorNode
```

**Parameters:**

- `children_scorers`: An array of maps, where each map has two keys:
  - `node`: The `BehaviorNode` child.
  - `scorer`: A function pointer to a scorer function. The scorer function receives the `RobotSituation` object and must return a floating-point score.
- `hysteresis_margin`: A float value. A new child will only be chosen if its score exceeds the current best score by this margin. This prevents flip-flopping.
- `description` (optional): A string description for debugging.

**Example:**

```rust
fn score_attack(s) { /* ... returns a score ... */ }
fn score_defend(s) { /* ... returns a score ... */ }

ScoringSelect(
    [
        #{ node: AttackBehavior(), scorer: score_attack },
        #{ node: DefendBehavior(), scorer: score_defend }
    ],
    0.1, // Hysteresis margin of 0.1
    "Choose between attacking and defending"
)
```

## Decorator Nodes

Decorator nodes have a single child and modify its behavior.

### `Guard`

A `Guard` node, or condition node, checks a condition before executing its child. If the condition is true, it ticks the child. If the condition is false, it returns `Failure` immediately without ticking the child.

**Syntax:**

```rust
Guard(condition_fn: FnPtr, child: BehaviorNode, cond_description: String) -> BehaviorNode
```

**Parameters:**

- `condition_fn`: A function pointer to a condition function. The condition function receives the `RobotSituation` object and must return `true` or `false`.
- `child`: The `BehaviorNode` to execute if the condition is true.
- `cond_description`: A string description of the condition for debugging.

**Example:**

```rust
fn we_have_the_ball(s) {
    return s.has_ball();
}

// Only execute the 'ShootGoal' action if we have the ball.
Guard(
    we_have_the_ball,
    ShootGoal(),
    "Check if we have the ball"
)
```

### `Semaphore`

A `Semaphore` node is used for team-level coordination. It limits the number of robots that can execute its child node at the same time. Each semaphore is identified by a unique string ID.

For example, you can use a semaphore to ensure only one robot tries to be the primary attacker.

**Syntax:**

```rust
Semaphore(child: BehaviorNode, id: String, max_count: Int, [description: String]) -> BehaviorNode
```

**Parameters:**

- `child`: The `BehaviorNode` to execute.
- `id`: A unique string identifier for the semaphore.
- `max_count`: The maximum number of robots that can acquire this semaphore.
- `description` (optional): A string description for debugging.

**Example:**

```rust
// Only one player can be the attacker at a time.
Semaphore(
    AttackerBehavior(),
    "attacker_semaphore",
    1
)
```
