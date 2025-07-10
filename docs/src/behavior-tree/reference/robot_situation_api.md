# RobotSituation API Reference

The `RobotSituation` object is the primary interface for accessing world state information in your behavior tree scripts. It provides comprehensive methods for querying the environment, calculating distances and geometry, and making intelligent decisions.

## Core Properties

### Basic Information

- `s.player_id` - The `PlayerId` of the current robot
- `s.has_ball()` - Returns `true` if the robot's breakbeam sensor detects the ball
- `s.player()` - Returns `PlayerData` object for the current robot
- `s.world` - The world data containing all players, ball, and game state

## World State Queries

These methods help you query the world state relative to the current player.

### Player Proximity Queries

#### `closest_own_player_to_ball()`

Returns the closest teammate to the ball (excluding the current robot).

```rust
fn should_support_attacker(s) {
    let closest = s.closest_own_player_to_ball();
    if closest != () {
        // There's a teammate closer to the ball
        return true;
    }
    return false;
}
```

#### `closest_own_player_to_me()`

Returns the closest teammate to the current robot.

```rust
fn find_pass_target(s) {
    let teammate = s.closest_own_player_to_me();
    if teammate != () {
        return teammate.position;
    }
    return s.get_opp_goal_position(); // Default to goal
}
```

#### `closest_own_player_to_position(pos)`

Returns the closest teammate to a specific position.

```rust
fn who_should_defend_goal(s) {
    let goal_pos = s.get_own_goal_position();
    return s.closest_own_player_to_position(goal_pos);
}
```

#### `closest_opp_player_to_me()`

Returns the closest opponent to the current robot.

```rust
fn am_i_being_pressured(s) {
    let opp = s.closest_opp_player_to_me();
    if opp != () {
        return s.distance_to_position(opp.position) < 800.0;
    }
    return false;
}
```

#### `closest_opp_player_to_position(pos)`

Returns the closest opponent to a specific position.

### Distance Calculations

#### `distance_to_ball()`

Returns the distance from the current robot to the ball.

```rust
fn score_as_attacker(s) {
    let dist_to_ball = s.distance_to_ball();
    return 1000.0 - dist_to_ball; // Closer = higher score
}
```

#### `distance_to_player(player_id)`

Returns the distance to another player (teammate or opponent).

```rust
fn should_pass_to_player(s, target_id) {
    let distance = s.distance_to_player(target_id);
    return distance < 2000.0 && distance > 200.0; // Not too far, not too close
}
```

#### `distance_to_position(pos)`

Returns the distance from the current robot to a position.

```rust
fn am_i_near_goal(s) {
    let goal_pos = s.get_opp_goal_position();
    return s.distance_to_position(goal_pos) < 1000.0;
}
```

## Geometry Calculations

Methods for field-related geometric queries.

### Field Boundaries

#### `distance_to_nearest_wall()`

Returns the distance to the closest field boundary.

```rust
fn am_i_near_wall(s) {
    return s.distance_to_nearest_wall() < 300.0;
}
```

#### `distance_to_wall_in_direction(angle)`

Returns the distance to the field boundary in a specific direction (angle in radians).

```rust
fn can_i_move_forward(s) {
    let my_heading = s.player().heading;
    return s.distance_to_wall_in_direction(my_heading) > 500.0;
}
```

### Goal and Field Positions

#### `get_own_goal_position()`

Returns the position of your team's goal center.

```rust
fn get_defensive_position(s) {
    let goal = s.get_own_goal_position();
    let ball_pos = s.world.ball.position;
    // Position between ball and goal
    return goal + (ball_pos - goal) * 0.3;
}
```

#### `get_opp_goal_position()`

Returns the position of the opponent's goal center.

```rust
fn should_shoot(s) {
    let goal_pos = s.get_opp_goal_position();
    let my_pos = s.player().position;
    return (goal_pos - my_pos).norm() < 3000.0; // Within shooting range
}
```

#### `get_field_center()`

Returns the center position of the field (0, 0).

#### `is_position_in_field(pos)`

Checks if a position is within the field boundaries.

```rust
fn is_target_safe(s, target_pos) {
    return s.is_position_in_field(target_pos);
}
```

#### `get_field_bounds()`

Returns a map with field boundary coordinates.

```rust
fn get_safe_position(s) {
    let bounds = s.get_field_bounds();
    let safe_x = bounds.min_x + 200.0; // 200mm from boundary
    let safe_y = 0.0;
    return vec2(safe_x, safe_y);
}
```

### Field Zone Queries

#### `is_in_penalty_area(pos)`

Checks if a position is in any penalty area (own or opponent).

```rust
fn avoid_penalty_areas(s, target_pos) {
    if s.is_in_penalty_area(target_pos) {
        // Find alternative position
        return s.get_field_center();
    }
    return target_pos;
}
```

#### `is_in_own_penalty_area(pos)`

Checks if a position is in your team's penalty area.

```rust
fn am_i_goalkeeper_now(s) {
    let my_pos = s.player().position;
    return s.is_in_own_penalty_area(my_pos);
}
```

#### `is_in_opp_penalty_area(pos)`

Checks if a position is in the opponent's penalty area.

```rust
fn can_i_score_from_here(s) {
    let my_pos = s.player().position;
    return s.is_in_opp_penalty_area(my_pos) && s.has_ball();
}
```

#### `is_in_center_circle(pos)`

Checks if a position is within the center circle.

```rust
fn kickoff_positioning(s) {
    let my_pos = s.player().position;
    if s.is_in_center_circle(my_pos) {
        // Move out of center circle during kickoff
        return vec2(-1000.0, 0.0);
    }
    return my_pos;
}
```

#### `is_in_attacking_half(pos)`

Checks if a position is in the attacking half (x > 0).

```rust
fn count_attackers(s) {
    return s.count_own_players_where(|player| {
        s.is_in_attacking_half(player.position)
    });
}
```

#### `is_in_defensive_half(pos)`

Checks if a position is in the defensive half (x < 0).

```rust
fn should_defend(s) {
    let my_pos = s.player().position;
    let ball_pos = s.world.ball.position.xy();
    return s.is_in_defensive_half(ball_pos) || s.is_in_defensive_half(my_pos);
}
```

#### `distance_to_own_penalty_area()`

Returns the distance from the current robot to your team's penalty area.

```rust
fn goalkeeper_positioning(s) {
    let dist_to_area = s.distance_to_own_penalty_area();
    if dist_to_area > 100.0 {
        // Too far from penalty area, move closer
        return s.get_own_penalty_mark();
    }
    return s.player().position;
}
```

#### `distance_to_opp_penalty_area()`

Returns the distance from the current robot to the opponent's penalty area.

```rust
fn should_advance_attack(s) {
    let dist_to_area = s.distance_to_opp_penalty_area();
    return dist_to_area < 2000.0 && s.has_ball();
}
```

### Additional Field Positions

#### `get_own_penalty_mark()`

Returns the position of your team's penalty mark.

```rust
fn penalty_shooter_position(s) {
    let penalty_mark = s.get_own_penalty_mark();
    // Position slightly behind penalty mark
    return penalty_mark + vec2(-200.0, 0.0);
}
```

#### `get_opp_penalty_mark()`

Returns the position of the opponent's penalty mark.

```rust
fn penalty_shot_target(s) {
    return s.get_opp_penalty_mark();
}
```

#### `get_own_goal_corners()`

Returns an array with the two corners of your team's goal.

```rust
fn goal_coverage(s) {
    let corners = s.get_own_goal_corners();
    let top_corner = corners[0];
    let bottom_corner = corners[1];
    // Choose corner based on ball position
    let ball_pos = s.world.ball.position.xy();
    if ball_pos.y > 0.0 {
        return top_corner;
    } else {
        return bottom_corner;
    }
}
```

#### `get_opp_goal_corners()`

Returns an array with the two corners of the opponent's goal.

```rust
fn choose_shot_target(s) {
    let corners = s.get_opp_goal_corners();
    let top_corner = corners[0];
    let bottom_corner = corners[1];

    // Aim for the corner farther from goalkeeper
    let goalkeeper = s.find_opp_player_min_by(|player| {
        s.distance_to_position(player.position)
    });

    if goalkeeper != () {
        if goalkeeper.position.y > 0.0 {
            return bottom_corner; // Aim low if keeper is high
        } else {
            return top_corner; // Aim high if keeper is low
        }
    }

    return top_corner; // Default
}
```

#### `get_corner_positions()`

Returns an array with all four field corner positions.

```rust
fn corner_kick_positioning(s) {
    let corners = s.get_corner_positions();
    let ball_pos = s.world.ball.position.xy();

    // Find the closest corner to the ball
    let mut closest_corner = corners[0];
    let mut min_distance = ball_pos.distance_to(closest_corner);

    for corner in corners {
        let distance = ball_pos.distance_to(corner);
        if distance < min_distance {
            min_distance = distance;
            closest_corner = corner;
        }
    }

    return closest_corner;
}
```

## Global World Queries

Advanced queries that search through all players using custom criteria.

### Player Search Functions

#### `find_own_player_min_by(scorer_fn)`

Finds the teammate with the minimum score according to your function.

```rust
// Find teammate closest to ball
fn find_ball_chaser(s) {
    return s.find_own_player_min_by(|player| {
        (player.position - s.world.ball.position.xy()).norm()
    });
}
```

#### `find_own_player_max_by(scorer_fn)`

Finds the teammate with the maximum score according to your function.

```rust
// Find teammate with highest y-coordinate (topmost)
fn find_topmost_teammate(s) {
    return s.find_own_player_max_by(|player| player.position.y);
}
```

#### `find_opp_player_min_by(scorer_fn)` / `find_opp_player_max_by(scorer_fn)`

Same as above but for opponent players.

### Player Filtering

#### `filter_own_players_by(predicate_fn)`

Returns all teammates that match your condition.

```rust
// Get all teammates in attacking half
fn get_attacking_teammates(s) {
    return s.filter_own_players_by(|player| player.position.x > 0.0);
}
```

#### `filter_opp_players_by(predicate_fn)`

Returns all opponents that match your condition.

#### `count_own_players_where(predicate_fn)`

Counts teammates matching your condition.

```rust
fn count_defenders(s) {
    return s.count_own_players_where(|player| player.position.x < -1000.0);
}
```

#### `count_opp_players_where(predicate_fn)`

Counts opponents matching your condition.

### Player Collections by Location

#### `get_players_within_radius(center, radius)`

Returns all players (teammates and opponents) within radius of center.

```rust
fn is_area_crowded(s, pos, radius) {
    let players = s.get_players_within_radius(pos, radius);
    return players.len() > 3;
}
```

#### `get_own_players_within_radius(center, radius)`

Returns only teammates within radius of center.

#### `get_opp_players_within_radius(center, radius)`

Returns only opponents within radius of center.

## Ray Casting and Prediction

Advanced physics-based queries for intelligent positioning and prediction.

### Ray Casting

#### `cast_ray(from, to)`

Casts a ray from one position to another and returns hit information.

```rust
fn is_path_clear(s, from_pos, to_pos) {
    let ray_result = s.cast_ray(from_pos, to_pos);
    return !ray_result.hit; // True if no obstruction
}

fn what_blocks_shot(s) {
    let my_pos = s.player().position;
    let goal_pos = s.get_opp_goal_position();
    let result = s.cast_ray(my_pos, goal_pos);

    if result.hit {
        print(`Shot blocked by: ${result.hit_type} at ${result.hit_position}`);
    }
}
```

### Ball Prediction

#### `predict_ball_position(time_seconds)`

Predicts where the ball will be after a given time using physics simulation.

```rust
fn intercept_ball(s) {
    // Predict where ball will be in 1 second
    let future_ball_pos = s.predict_ball_position(1.0);
    if future_ball_pos != () {
        return future_ball_pos;
    }
    return s.world.ball.position.xy(); // Fallback to current position
}
```

#### `predict_ball_collision_time()`

Predicts when the ball will collide with a field boundary.

```rust
fn will_ball_go_out(s) {
    let collision_time = s.predict_ball_collision_time();
    return collision_time < 3.0; // Ball will hit wall in 3 seconds
}
```

## Vector Utilities

The `Vec2` and `Vec3` types now support extensive mathematical operations.

### Vector Methods

#### Basic Properties

```rust
let vec = vec2(3.0, 4.0);
let length = vec.norm();        // 5.0
let unit_vec = vec.unit();      // (0.6, 0.8)
```

#### Geometric Operations

```rust
let from = vec2(0.0, 0.0);
let to = vec2(3.0, 4.0);

let angle = from.angle_to(to);           // Angle in radians
let distance = from.distance_to(to);     // 5.0
let rotated = to.rotate(1.57);           // Rotate 90 degrees
let halfway = from.halfway_to(to);       // (1.5, 2.0)
let lerped = from.interpolate(to, 0.5);  // (1.5, 2.0)
```

### Vector Arithmetic with Operators

Vectors now support natural mathematical operators:

```rust
let a = vec2(1.0, 2.0);
let b = vec2(3.0, 4.0);

// Vector addition/subtraction
let sum = a + b;        // vec2(4.0, 6.0)
let diff = b - a;       // vec2(2.0, 2.0)

// Scalar multiplication/division
let scaled = a * 2.0;   // vec2(2.0, 4.0)
let halved = a / 2.0;   // vec2(0.5, 1.0)

// Unary negation
let negated = -a;       // vec2(-1.0, -2.0)
```

## Complete Example: Intelligent Attacker

Here's a comprehensive example using multiple API features:

```rust
fn should_attack(s) {
    // Am I the closest to the ball?
    let closest_to_ball = s.closest_own_player_to_ball();
    if closest_to_ball != () {
        return false; // Someone else is closer
    }

    // Is the ball in a good attacking position?
    let ball_pos = s.world.ball.position.xy();
    let goal_pos = s.get_opp_goal_position();
    let distance_to_goal = ball_pos.distance_to(goal_pos);

    if distance_to_goal > 4000.0 {
        return false; // Too far from goal
    }

    // Is there a clear shot to goal?
    let shot_clear = s.cast_ray(ball_pos, goal_pos);
    if shot_clear.hit && shot_clear.hit_type == "player" {
        return false; // Shot blocked
    }

    // Check if I'm being pressured by opponents
    let nearest_opp = s.closest_opp_player_to_me();
    if nearest_opp != () {
        let pressure_distance = s.distance_to_position(nearest_opp.position);
        if pressure_distance < 600.0 {
            return false; // Too much pressure
        }
    }

    return true; // All conditions met, attack!
}

fn get_attack_target(s) {
    let goal_pos = s.get_opp_goal_position();
    let ball_pos = s.world.ball.position.xy();

    // Look for a pass opportunity first
    let passing_teammates = s.filter_own_players_by(|player| {
        let pos = player.position;
        // Must be in attacking half and closer to goal than me
        pos.x > 0.0 && pos.distance_to(goal_pos) < ball_pos.distance_to(goal_pos)
    });

    if passing_teammates.len() > 0 {
        // Find the teammate with the clearest shot
        let best_teammate = s.find_own_player_min_by(|player| {
            let shot_result = s.cast_ray(player.position, goal_pos);
            if shot_result.hit {
                1000.0 // Penalize blocked shots
            } else {
                player.position.distance_to(goal_pos)
            }
        });

        if best_teammate != () {
            return best_teammate.position;
        }
    }

    // No good pass, go for goal
    return goal_pos;
}
```

This comprehensive API gives you powerful tools to create intelligent, context-aware robot behaviors that can adapt to complex game situations.
