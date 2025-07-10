# Role Assignment API

The role assignment system allows you to define different roles for robots and automatically assign them based on scoring functions and filters.

## Basic Structure

```rhai
fn main() {
    AssignRoles([
        Role("role_name")
            .score(|situation| /* scoring function */)
            .require(|situation| /* filter that must return true */)
            .exclude(|situation| /* filter that must return false */)
            .min(1)  // minimum number of robots for this role
            .max(3)  // maximum number of robots for this role
            .count(2)  // exact count (sets min = max = 2)
            .behavior(|situation| /* behavior tree builder */)
            .build(),
        // ... more roles
    ])
}
```

## Role Methods

### `.score(|s| -> f64)`

**Required** - Function that returns a score for how well a robot fits this role. Higher scores are better.

```rhai
.score(|s| {
    if let Some(ball) = s.world.ball {
        let distance = (s.player().position - ball.position.xy()).magnitude();
        100.0 - distance  // Closer to ball = higher score
    } else {
        50.0  // Default score
    }
})
```

### `.require(|s| -> bool)`

_Optional_ - Filter function that must return `true` for a robot to be eligible for this role.

```rhai
.require(|s| s.player_id != 0)  // Only non-goalkeeper robots
.require(|s| {
    // Only robots with working kicker
    if let Some(status) = s.player().kicker_status {
        status.to_string() == "Ready"
    } else {
        false
    }
})
```

### `.exclude(|s| -> bool)`

_Optional_ - Filter function that must return `false` for a robot to be eligible for this role.

```rhai
.exclude(|s| s.player_id == 0)  // Exclude goalkeeper
.exclude(|s| {
    // Exclude robots that are too far from ball
    if let Some(ball) = s.world.ball {
        let distance = (s.player().position - ball.position.xy()).magnitude();
        distance > 2000.0
    } else {
        false
    }
})
```

### `.behavior(|| -> BehaviorNode)`

**Required** - Function that builds the behavior tree for robots assigned to this role.

```rhai
.behavior(|s| {
    Select([
        Guard(|s| s.has_ball(), Kick("Shoot"), "has_ball"),
        FetchBall("Get ball")
    ], "Attacker behavior")
})
```

### Count Constraints

- `.min(count)` - Minimum number of robots for this role
- `.max(count)` - Maximum number of robots for this role
- `.count(count)` - Exact count (equivalent to `.min(count).max(count)`)

## Complete Example

```rhai
fn main() {
    AssignRoles([
        // Goalkeeper - exactly one robot, must be robot 0
        Role("goalkeeper")
            .count(1)
            .score(|s| 100.0)  // Fixed high score
            .require(|s| s.player_id == 0)
            .behavior(|s| build_goalkeeper_tree(s))
            .build(),

        // Attacker - 1-2 robots, prefers closer to ball, excludes goalkeeper
        Role("attacker")
            .min(1)
            .max(2)
            .score(|s| {
                if let Some(ball) = s.world.ball {
                    let distance = (s.player().position - ball.position.xy()).magnitude();
                    200.0 - distance
                } else {
                    50.0
                }
            })
            .exclude(|s| s.player_id == 0)
            .behavior(|s| build_attacker_tree(s))
            .build(),

        // Defender - remaining robots
        Role("defender")
            .min(1)
            .max(3)
            .score(|s| {
                let own_goal = vec2(-FIELD_HALF_LENGTH, 0.0);
                let distance = (s.player().position - own_goal).magnitude();
                150.0 - distance
            })
            .exclude(|s| s.player_id == 0)
            .behavior(|s| build_defender_tree(s))
            .build()
    ])
}
```

## Situation Object

The situation object `s` passed to scoring and filter functions contains:

- `s.player_id` - ID of the current robot
- `s.player()` - Player data (position, velocity, status, etc.)
- `s.world` - World data (ball, other players, game state)
- `s.has_ball()` - Whether this robot has the ball
- `s.hash_float()` - Deterministic random value for this robot (0.0-1.0)

## Migration from Old API

The new API removes:

- Capability system (`require_capability`, `add_robot_capability`)
- Robot ID constraints (`exclude_robots`, `only_robots`)
- `RoleAssignmentBuilder` - now use `AssignRoles()` directly
- Separate scorer function - each role has its own scorer

Instead, use filter functions (`.require()` and `.exclude()`) for all constraints.
