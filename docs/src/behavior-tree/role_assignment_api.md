# Role Assignment API

The role assignment system allows you to define different roles for robots and automatically assign them based on scoring functions and filters. The system is **dynamic** - the main function is called whenever the number of players changes or the game state changes, allowing strategies to adapt in real-time.

## Entry Point

Your strategy script must provide a `main(game)` function that adds roles using the game context:

```rhai
fn main(game) {
    // Access game information
    let current_state = game.game_state;
    let player_count = game.num_own_players;

    // Add roles using the fluent API
    game.add_role("goalkeeper")
        .count(1)
        .score(|s| 100.0)
        .require(|s| s.player_id == 0)
        .behavior(|| build_goalkeeper_tree())
        .build();

    // Add special roles based on game state
    if current_state == "FreeKick" {
        game.add_role("free_kicker")
            .max(1)
            .score(|s| score_free_kicker(s))
            .behavior(|| build_free_kicker_tree())
            .build();
    }
}
```

## When main() is Called

The `main(game)` function is automatically called by the system in the following situations:

- **Player count changes**: When robots join or leave the field
- **Game state changes**: When transitioning between game states (Normal, FreeKick, PenaltyKick, etc.)
- **Initial startup**: When the system first starts

This allows your strategy to dynamically adapt roles based on the current situation.

## Game Context Object

The `game` parameter provides access to current game information:

- `game.game_state` - Current game state (GameState enum)
- `game.num_own_players` - Number of your team's active players
- `game.num_opp_players` - Number of opponent players
- `game.field_geom` - Field geometry information (may be `()` if not available)
- `game.add_role(name)` - Add a new role and return a builder for configuration

### Game State Values

The `game.game_state` can be one of:

- `"Halt"` - Game is halted
- `"Stop"` - Game is stopped
- `"Play"` - Normal gameplay
- `"FreeKick"` - Free kick situation
- `"PenaltyKick"` - Penalty kick situation
- `"BallReplacement"` - Ball placement
- `"Kickoff"` - Kickoff situation

## Modern API Structure

```rhai
fn main(game) {
    game.add_role("role_name")
        .score(|situation| /* scoring function */)
        .require(|situation| /* filter that must return true */)
        .exclude(|situation| /* filter that must return false */)
        .min(1)  // minimum number of robots for this role
        .max(3)  // maximum number of robots for this role
        .count(2)  // exact count (sets min = max = 2)
        .behavior(|| /* behavior tree builder */)
        .build();  // Build and store the role automatically
}
```

## Role Builder Methods

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
```

### `.exclude(|s| -> bool)`

_Optional_ - Filter function that must return `false` for a robot to be eligible for this role.

```rhai
.exclude(|s| s.player_id == 0)  // Exclude goalkeeper
```

### `.behavior(|| -> BehaviorNode)`

**Required** - Function that builds the behavior tree for robots assigned to this role.

```rhai
.behavior(|| {
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

### `.build()`

**Required** - Builds the role and automatically stores it in the game context. No return value needed.

## Complete Example

```rhai
fn main(game) {
    // Goalkeeper - exactly one robot, must be robot 0
    game.add_role("goalkeeper")
        .count(1)
        .score(|s| 100.0)  // Fixed high score
        .require(|s| s.player_id == 0)
        .behavior(|| build_goalkeeper_tree())
        .build();

    // Attacker - 1-2 robots, prefers closer to ball, excludes goalkeeper
    game.add_role("attacker")
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
        .behavior(|| build_attacker_tree())
        .build();

    // Defender - remaining robots
    game.add_role("defender")
        .min(1)
        .max(3)
        .score(|s| {
            let own_goal = vec2(-FIELD_HALF_LENGTH, 0.0);
            let distance = (s.player().position - own_goal).magnitude();
            150.0 - distance
        })
        .exclude(|s| s.player_id == 0)
        .behavior(|| build_defender_tree())
        .build();
}
```

## Situation Object

The situation object `s` passed to scoring and filter functions contains:

- `s.player_id` - ID of the current robot
- `s.player()` - Player data (position, velocity, status, etc.)
- `s.world` - World data (ball, other players, game state)
- `s.has_ball()` - Whether this robot has the ball
- `s.hash_float()` - Deterministic random value for this robot (0.0-1.0)
