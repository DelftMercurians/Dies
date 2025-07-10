use std::sync::Arc;

use dies_core::{
    is_pos_in_field, BallData, BallPrediction, FieldGeometry, GameState, GameStateData, PlayerData,
    PlayerId, TeamData, Vector2, Vector3,
};
use rhai::{Array, Dynamic, Engine, FnPtr, Map, NativeCallContext};

use crate::behavior_tree::{
    role_assignment::{Role, RoleAssignmentProblem, RoleBuilder},
    RobotSituation,
};

/// Register all types and their methods with the Rhai engine
pub fn register_all_types(engine: &mut Engine) {
    register_robot_situation_type(engine);
    register_team_data_type(engine);
    register_player_data_type(engine);
    register_ball_data_type(engine);
    register_vector_types(engine);
    register_game_state_types(engine);
    register_field_geometry_type(engine);
    register_role_assignment_types(engine);
    register_vector_operators(engine);
}

/// Register RobotSituation type with extended world query methods
fn register_robot_situation_type(engine: &mut Engine) {
    engine
        .register_type_with_name::<RobotSituation>("RobotSituation")
        .register_fn("has_ball", |rs: &mut RobotSituation| rs.has_ball())
        .register_get("player", |rs: &mut RobotSituation| rs.player_data().clone())
        .register_get("world", |rs: &mut RobotSituation| rs.world.clone())
        .register_get("player_id", |rs: &mut RobotSituation| rs.player_id)
        // World state queries relative to current player
        .register_fn("closest_own_player_to_ball", closest_own_player_to_ball)
        .register_fn("closest_own_player_to_me", closest_own_player_to_me)
        .register_fn(
            "closest_own_player_to_position",
            closest_own_player_to_position,
        )
        .register_fn("closest_opp_player_to_me", closest_opp_player_to_me)
        .register_fn(
            "closest_opp_player_to_position",
            closest_opp_player_to_position,
        )
        .register_fn("distance_to_ball", distance_to_ball)
        .register_fn("distance_to_player", distance_to_player)
        .register_fn("distance_to_position", distance_to_position)
        // Geometry calculations
        .register_fn("distance_to_nearest_wall", distance_to_nearest_wall)
        .register_fn(
            "distance_to_wall_in_direction",
            distance_to_wall_in_direction,
        )
        .register_fn("get_own_goal_position", get_own_goal_position)
        .register_fn("get_opp_goal_position", get_opp_goal_position)
        .register_fn("get_field_center", get_field_center)
        .register_fn("is_position_in_field", is_position_in_field)
        .register_fn("get_field_bounds", get_field_bounds)
        // Field zone and area queries
        .register_fn("is_in_penalty_area", is_in_penalty_area)
        .register_fn("is_in_own_penalty_area", is_in_own_penalty_area)
        .register_fn("is_in_opp_penalty_area", is_in_opp_penalty_area)
        .register_fn("is_in_center_circle", is_in_center_circle)
        .register_fn("is_in_attacking_half", is_in_attacking_half)
        .register_fn("is_in_defensive_half", is_in_defensive_half)
        .register_fn("distance_to_own_penalty_area", distance_to_own_penalty_area)
        .register_fn("distance_to_opp_penalty_area", distance_to_opp_penalty_area)
        // Additional field positions
        .register_fn("get_own_penalty_mark", get_own_penalty_mark)
        .register_fn("get_opp_penalty_mark", get_opp_penalty_mark)
        .register_fn("get_goal_corners", get_goal_corners)
        .register_fn("get_own_goal_corners", get_own_goal_corners)
        .register_fn("get_opp_goal_corners", get_opp_goal_corners)
        .register_fn("get_corner_positions", get_corner_positions)
        // Global world queries
        .register_fn("find_own_player_min_by", find_own_player_min_by)
        .register_fn("find_own_player_max_by", find_own_player_max_by)
        .register_fn("find_opp_player_min_by", find_opp_player_min_by)
        .register_fn("find_opp_player_max_by", find_opp_player_max_by)
        .register_fn("filter_own_players_by", filter_own_players_by)
        .register_fn("filter_opp_players_by", filter_opp_players_by)
        .register_fn("count_own_players_where", count_own_players_where)
        .register_fn("count_opp_players_where", count_opp_players_where)
        // Player collections
        .register_fn("get_players_within_radius", get_players_within_radius)
        .register_fn(
            "get_own_players_within_radius",
            get_own_players_within_radius,
        )
        .register_fn(
            "get_opp_players_within_radius",
            get_opp_players_within_radius,
        )
        // Ray casting and prediction
        .register_fn("cast_ray", cast_ray)
        .register_fn("predict_ball_position", predict_ball_position)
        .register_fn("predict_ball_collision_time", predict_ball_collision_time);
}

/// Register TeamData (World) type
fn register_team_data_type(engine: &mut Engine) {
    engine
        .register_type_with_name::<Arc<TeamData>>("World")
        .register_get("ball", |wd: &mut Arc<TeamData>| {
            if let Some(ball) = &wd.ball {
                Dynamic::from(ball.clone())
            } else {
                Dynamic::from(())
            }
        })
        .register_get("own_players", |wd: &mut Arc<TeamData>| {
            wd.own_players.clone()
        })
        .register_get("opp_players", |wd: &mut Arc<TeamData>| {
            wd.opp_players.clone()
        })
        .register_get("game_state", |wd: &mut Arc<TeamData>| {
            wd.current_game_state.clone()
        })
        .register_get("field_geom", |wd: &mut Arc<TeamData>| {
            if let Some(field_geom) = &wd.field_geom {
                Dynamic::from(field_geom.clone())
            } else {
                Dynamic::from(())
            }
        });
}

/// Register PlayerData type
fn register_player_data_type(engine: &mut Engine) {
    engine
        .register_type_with_name::<PlayerData>("PlayerData")
        .register_get("id", |pd: &mut PlayerData| pd.id)
        .register_get("position", |pd: &mut PlayerData| pd.position)
        .register_get("velocity", |pd: &mut PlayerData| pd.velocity)
        .register_get("heading", |pd: &mut PlayerData| pd.yaw.radians());
}

/// Register BallData type
fn register_ball_data_type(engine: &mut Engine) {
    engine
        .register_type_with_name::<BallData>("BallData")
        .register_get("position", |bd: &mut BallData| bd.position.xy())
        .register_get("position3", |bd: &mut BallData| bd.position)
        .register_get("velocity", |bd: &mut BallData| bd.velocity);
}

/// Register Vector2 and Vector3 types with extended math operations
fn register_vector_types(engine: &mut Engine) {
    engine
        .register_type_with_name::<Vector2>("Vec2")
        .register_get("x", |v: &mut Vector2| v.x)
        .register_get("y", |v: &mut Vector2| v.y)
        // Vector math methods
        .register_fn("norm", |v: &mut Vector2| v.norm())
        .register_fn("unit", |v: &mut Vector2| v.normalize())
        .register_fn("angle_to", vec2_angle_to)
        .register_fn("distance_to", vec2_distance_to)
        .register_fn("rotate", vec2_rotate)
        .register_fn("interpolate", vec2_interpolate)
        .register_fn("halfway_to", vec2_halfway_to);

    engine
        .register_type_with_name::<Vector3>("Vec3")
        .register_get("x", |v: &mut Vector3| v.x)
        .register_get("y", |v: &mut Vector3| v.y)
        .register_get("z", |v: &mut Vector3| v.z)
        .register_fn("norm", |v: &mut Vector3| v.norm())
        .register_fn("unit", |v: &mut Vector3| v.normalize())
        .register_fn("xy", |v: &mut Vector3| v.xy());
}

/// Register GameState types
fn register_game_state_types(engine: &mut Engine) {
    engine
        .register_type_with_name::<GameStateData>("GameStateData")
        .register_get("game_state", |gsd: &mut GameStateData| gsd.game_state)
        .register_get("us_operating", |gsd: &mut GameStateData| gsd.us_operating);

    engine.register_type_with_name::<GameState>("GameState");
}

/// Register FieldGeometry type
fn register_field_geometry_type(engine: &mut Engine) {
    engine
        .register_type_with_name::<FieldGeometry>("FieldGeometry")
        .register_get("field_length", |fg: &mut FieldGeometry| fg.field_length)
        .register_get("field_width", |fg: &mut FieldGeometry| fg.field_width)
        .register_get("goal_width", |fg: &mut FieldGeometry| fg.goal_width)
        .register_get("goal_depth", |fg: &mut FieldGeometry| fg.goal_depth)
        .register_get("boundary_width", |fg: &mut FieldGeometry| fg.boundary_width)
        .register_get("penalty_area_depth", |fg: &mut FieldGeometry| {
            fg.penalty_area_depth
        })
        .register_get("penalty_area_width", |fg: &mut FieldGeometry| {
            fg.penalty_area_width
        })
        .register_get("center_circle_radius", |fg: &mut FieldGeometry| {
            fg.center_circle_radius
        })
        .register_get("goal_line_to_penalty_mark", |fg: &mut FieldGeometry| {
            fg.goal_line_to_penalty_mark
        })
        .register_get("ball_radius", |fg: &mut FieldGeometry| fg.ball_radius);
}

/// Register role assignment types
fn register_role_assignment_types(engine: &mut Engine) {
    engine
        .register_type_with_name::<RoleBuilder>("RoleBuilder")
        .register_type_with_name::<Role>("Role")
        .register_type_with_name::<RoleAssignmentProblem>("RoleAssignmentProblem");
}

/// Register vector arithmetic operators
fn register_vector_operators(engine: &mut Engine) {
    // Vector2 + Vector2
    engine.register_fn("+", |a: Vector2, b: Vector2| a + b);
    // Vector3 + Vector2
    engine.register_fn("+", |a: Vector3, b: Vector2| a.xy() + b);
    // Vector2 + Vector3
    engine.register_fn("+", |a: Vector2, b: Vector3| a + b.xy());
    // Vector2 - Vector2
    engine.register_fn("-", |a: Vector2, b: Vector2| a - b);
    // Vector3 - Vector2
    engine.register_fn("-", |a: Vector3, b: Vector2| a.xy() - b);
    // Vector2 - Vector3
    engine.register_fn("-", |a: Vector2, b: Vector3| a - b.xy());
    // Vector2 * scalar
    engine.register_fn("*", |a: Vector2, s: f64| a * s);
    engine.register_fn("*", |s: f64, a: Vector2| a * s);
    // Vector2 / scalar
    engine.register_fn("/", |a: Vector2, s: f64| a / s);
    // Unary minus
    engine.register_fn("-", |a: Vector2| -a);

    // Vector3 operations
    engine.register_fn("+", |a: Vector3, b: Vector3| a + b);
    engine.register_fn("-", |a: Vector3, b: Vector3| a - b);
    engine.register_fn("*", |a: Vector3, s: f64| a * s);
    engine.register_fn("*", |s: f64, a: Vector3| a * s);
    engine.register_fn("/", |a: Vector3, s: f64| a / s);
    engine.register_fn("-", |a: Vector3| -a);
}

// ===== RobotSituation World Query Methods =====

fn closest_own_player_to_ball(rs: &mut RobotSituation) -> Dynamic {
    if let Some(ball) = &rs.world.ball {
        let ball_pos = ball.position.xy();
        rs.world
            .own_players
            .iter()
            .filter(|p| p.id != rs.player_id) // Exclude self
            .min_by(|a, b| {
                let dist_a = (a.position - ball_pos).norm();
                let dist_b = (b.position - ball_pos).norm();
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| Dynamic::from(p.clone()))
            .unwrap_or_else(|| Dynamic::from(()))
    } else {
        Dynamic::from(())
    }
}

fn closest_own_player_to_me(rs: &mut RobotSituation) -> Dynamic {
    let my_pos = rs.player_data().position;
    rs.world
        .own_players
        .iter()
        .filter(|p| p.id != rs.player_id) // Exclude self
        .min_by(|a, b| {
            let dist_a = (a.position - my_pos).norm();
            let dist_b = (b.position - my_pos).norm();
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| Dynamic::from(p.clone()))
        .unwrap_or_else(|| Dynamic::from(()))
}

fn closest_own_player_to_position(rs: &mut RobotSituation, pos: Vector2) -> Dynamic {
    rs.world
        .own_players
        .iter()
        .filter(|p| p.id != rs.player_id) // Exclude self
        .min_by(|a, b| {
            let dist_a = (a.position - pos).norm();
            let dist_b = (b.position - pos).norm();
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| Dynamic::from(p.clone()))
        .unwrap_or_else(|| Dynamic::from(()))
}

fn closest_opp_player_to_me(rs: &mut RobotSituation) -> Dynamic {
    let my_pos = rs.player_data().position;
    rs.world
        .opp_players
        .iter()
        .min_by(|a, b| {
            let dist_a = (a.position - my_pos).norm();
            let dist_b = (b.position - my_pos).norm();
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| Dynamic::from(p.clone()))
        .unwrap_or_else(|| Dynamic::from(()))
}

fn closest_opp_player_to_position(rs: &mut RobotSituation, pos: Vector2) -> Dynamic {
    rs.world
        .opp_players
        .iter()
        .min_by(|a, b| {
            let dist_a = (a.position - pos).norm();
            let dist_b = (b.position - pos).norm();
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| Dynamic::from(p.clone()))
        .unwrap_or_else(|| Dynamic::from(()))
}

fn distance_to_ball(rs: &mut RobotSituation) -> f64 {
    if let Some(ball) = &rs.world.ball {
        (rs.player_data().position - ball.position.xy()).norm()
    } else {
        f64::INFINITY
    }
}

fn distance_to_player(rs: &mut RobotSituation, player_id: PlayerId) -> f64 {
    let my_pos = rs.player_data().position;

    // Check own players first
    if let Some(player) = rs.world.own_players.iter().find(|p| p.id == player_id) {
        return (my_pos - player.position).norm();
    }

    // Check opponent players
    if let Some(player) = rs.world.opp_players.iter().find(|p| p.id == player_id) {
        return (my_pos - player.position).norm();
    }

    f64::INFINITY
}

fn distance_to_position(rs: &mut RobotSituation, pos: Vector2) -> f64 {
    (rs.player_data().position - pos).norm()
}

// ===== Geometry Calculations =====

fn distance_to_nearest_wall(rs: &mut RobotSituation) -> f64 {
    if let Some(field) = &rs.world.field_geom {
        let pos = rs.player_data().position;
        let half_length = field.field_length / 2.0;
        let half_width = field.field_width / 2.0;

        let dist_to_left = pos.x + half_length;
        let dist_to_right = half_length - pos.x;
        let dist_to_bottom = pos.y + half_width;
        let dist_to_top = half_width - pos.y;

        dist_to_left
            .min(dist_to_right)
            .min(dist_to_bottom)
            .min(dist_to_top)
    } else {
        f64::INFINITY
    }
}

fn distance_to_wall_in_direction(rs: &mut RobotSituation, angle: f64) -> f64 {
    if let Some(field) = &rs.world.field_geom {
        let pos = rs.player_data().position;
        let direction = Vector2::new(angle.cos(), angle.sin());

        // Cast ray to field boundaries
        if let Some(intersection) = rs.world.cast_ray(pos, direction * 10000.0) {
            (intersection - pos).norm()
        } else {
            f64::INFINITY
        }
    } else {
        f64::INFINITY
    }
}

fn get_own_goal_position(rs: &mut RobotSituation) -> Vector2 {
    if let Some(field) = &rs.world.field_geom {
        Vector2::new(-field.field_length / 2.0, 0.0)
    } else {
        Vector2::new(-4500.0, 0.0) // Default
    }
}

fn get_opp_goal_position(rs: &mut RobotSituation) -> Vector2 {
    if let Some(field) = &rs.world.field_geom {
        Vector2::new(field.field_length / 2.0, 0.0)
    } else {
        Vector2::new(4500.0, 0.0) // Default
    }
}

fn get_field_center(_rs: &mut RobotSituation) -> Vector2 {
    Vector2::zeros()
}

fn is_position_in_field(rs: &mut RobotSituation, pos: Vector2) -> bool {
    if let Some(field) = &rs.world.field_geom {
        is_pos_in_field(pos, field)
    } else {
        true // Default to allowing all positions
    }
}

fn get_field_bounds(rs: &mut RobotSituation) -> Map {
    if let Some(field) = &rs.world.field_geom {
        let half_length = field.field_length / 2.0;
        let half_width = field.field_width / 2.0;

        let mut bounds = Map::new();
        bounds.insert("min_x".into(), Dynamic::from(-half_length));
        bounds.insert("max_x".into(), Dynamic::from(half_length));
        bounds.insert("min_y".into(), Dynamic::from(-half_width));
        bounds.insert("max_y".into(), Dynamic::from(half_width));
        bounds
    } else {
        let mut bounds = Map::new();
        bounds.insert("min_x".into(), Dynamic::from(-4500.0));
        bounds.insert("max_x".into(), Dynamic::from(4500.0));
        bounds.insert("min_y".into(), Dynamic::from(-3000.0));
        bounds.insert("max_y".into(), Dynamic::from(3000.0));
        bounds
    }
}

// ===== Field Zone and Area Queries =====

fn is_in_penalty_area(rs: &mut RobotSituation, pos: Vector2) -> bool {
    is_in_own_penalty_area(rs, pos) || is_in_opp_penalty_area(rs, pos)
}

fn is_in_own_penalty_area(rs: &mut RobotSituation, pos: Vector2) -> bool {
    if let Some(field) = &rs.world.field_geom {
        let half_length = field.field_length / 2.0;
        let half_penalty_width = field.penalty_area_width / 2.0;

        pos.x >= -half_length
            && pos.x <= -half_length + field.penalty_area_depth
            && pos.y >= -half_penalty_width
            && pos.y <= half_penalty_width
    } else {
        false
    }
}

fn is_in_opp_penalty_area(rs: &mut RobotSituation, pos: Vector2) -> bool {
    if let Some(field) = &rs.world.field_geom {
        let half_length = field.field_length / 2.0;
        let half_penalty_width = field.penalty_area_width / 2.0;

        pos.x <= half_length
            && pos.x >= half_length - field.penalty_area_depth
            && pos.y >= -half_penalty_width
            && pos.y <= half_penalty_width
    } else {
        false
    }
}

fn is_in_center_circle(rs: &mut RobotSituation, pos: Vector2) -> bool {
    if let Some(field) = &rs.world.field_geom {
        pos.norm() <= field.center_circle_radius
    } else {
        pos.norm() <= 500.0 // Default radius
    }
}

fn is_in_attacking_half(rs: &mut RobotSituation, pos: Vector2) -> bool {
    pos.x > 0.0
}

fn is_in_defensive_half(rs: &mut RobotSituation, pos: Vector2) -> bool {
    pos.x < 0.0
}

fn distance_to_own_penalty_area(rs: &mut RobotSituation) -> f64 {
    distance_to_penalty_area_impl(rs, true)
}

fn distance_to_opp_penalty_area(rs: &mut RobotSituation) -> f64 {
    distance_to_penalty_area_impl(rs, false)
}

fn distance_to_penalty_area_impl(rs: &mut RobotSituation, own_area: bool) -> f64 {
    if let Some(field) = &rs.world.field_geom {
        let pos = rs.player_data().position;
        let half_length = field.field_length / 2.0;
        let half_penalty_width = field.penalty_area_width / 2.0;

        let (area_left, area_right) = if own_area {
            (-half_length, -half_length + field.penalty_area_depth)
        } else {
            (half_length - field.penalty_area_depth, half_length)
        };

        // Calculate distance to rectangular penalty area
        let dx = if pos.x < area_left {
            area_left - pos.x
        } else if pos.x > area_right {
            pos.x - area_right
        } else {
            0.0
        };

        let dy = if pos.y < -half_penalty_width {
            -half_penalty_width - pos.y
        } else if pos.y > half_penalty_width {
            pos.y - half_penalty_width
        } else {
            0.0
        };

        (dx * dx + dy * dy).sqrt()
    } else {
        f64::INFINITY
    }
}

// ===== Additional Field Positions =====

fn get_own_penalty_mark(rs: &mut RobotSituation) -> Vector2 {
    if let Some(field) = &rs.world.field_geom {
        Vector2::new(
            -field.field_length / 2.0 + field.goal_line_to_penalty_mark,
            0.0,
        )
    } else {
        Vector2::new(-3500.0, 0.0) // Default
    }
}

fn get_opp_penalty_mark(rs: &mut RobotSituation) -> Vector2 {
    if let Some(field) = &rs.world.field_geom {
        Vector2::new(
            field.field_length / 2.0 - field.goal_line_to_penalty_mark,
            0.0,
        )
    } else {
        Vector2::new(3500.0, 0.0) // Default
    }
}

fn get_goal_corners(rs: &mut RobotSituation, own_goal: bool) -> Array {
    if let Some(field) = &rs.world.field_geom {
        let x = if own_goal {
            -field.field_length / 2.0
        } else {
            field.field_length / 2.0
        };
        let half_goal_width = field.goal_width / 2.0;

        vec![
            Dynamic::from(Vector2::new(x, half_goal_width)), // Top corner
            Dynamic::from(Vector2::new(x, -half_goal_width)), // Bottom corner
        ]
    } else {
        let x = if own_goal { -4500.0 } else { 4500.0 };
        vec![
            Dynamic::from(Vector2::new(x, 500.0)),  // Top corner
            Dynamic::from(Vector2::new(x, -500.0)), // Bottom corner
        ]
    }
}

fn get_own_goal_corners(rs: &mut RobotSituation) -> Array {
    get_goal_corners(rs, true)
}

fn get_opp_goal_corners(rs: &mut RobotSituation) -> Array {
    get_goal_corners(rs, false)
}

fn get_corner_positions(rs: &mut RobotSituation) -> Array {
    if let Some(field) = &rs.world.field_geom {
        let half_length = field.field_length / 2.0;
        let half_width = field.field_width / 2.0;

        vec![
            Dynamic::from(Vector2::new(-half_length, half_width)), // Top left
            Dynamic::from(Vector2::new(half_length, half_width)),  // Top right
            Dynamic::from(Vector2::new(half_length, -half_width)), // Bottom right
            Dynamic::from(Vector2::new(-half_length, -half_width)), // Bottom left
        ]
    } else {
        vec![
            Dynamic::from(Vector2::new(-4500.0, 3000.0)), // Top left
            Dynamic::from(Vector2::new(4500.0, 3000.0)),  // Top right
            Dynamic::from(Vector2::new(4500.0, -3000.0)), // Bottom right
            Dynamic::from(Vector2::new(-4500.0, -3000.0)), // Bottom left
        ]
    }
}

// ===== Global World Queries =====

fn find_own_player_min_by(
    context: NativeCallContext,
    rs: &mut RobotSituation,
    scorer_fn: FnPtr,
) -> Dynamic {
    find_player_by_score(&context, &rs.world.own_players, rs, scorer_fn, true)
}

fn find_own_player_max_by(
    context: NativeCallContext,
    rs: &mut RobotSituation,
    scorer_fn: FnPtr,
) -> Dynamic {
    find_player_by_score(&context, &rs.world.own_players, rs, scorer_fn, false)
}

fn find_opp_player_min_by(
    context: NativeCallContext,
    rs: &mut RobotSituation,
    scorer_fn: FnPtr,
) -> Dynamic {
    find_player_by_score(&context, &rs.world.opp_players, rs, scorer_fn, true)
}

fn find_opp_player_max_by(
    context: NativeCallContext,
    rs: &mut RobotSituation,
    scorer_fn: FnPtr,
) -> Dynamic {
    find_player_by_score(&context, &rs.world.opp_players, rs, scorer_fn, false)
}

fn find_player_by_score(
    context: &NativeCallContext,
    players: &[PlayerData],
    rs: &RobotSituation,
    scorer_fn: FnPtr,
    find_min: bool,
) -> Dynamic {
    let mut best_player: Option<&PlayerData> = None;
    let mut best_score = if find_min {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };

    for player in players {
        if let Ok(score) = scorer_fn.call_within_context::<f64>(context, (player.clone(),)) {
            let is_better = if find_min {
                score < best_score
            } else {
                score > best_score
            };
            if is_better {
                best_score = score;
                best_player = Some(player);
            }
        }
    }

    best_player
        .map(|p| Dynamic::from(p.clone()))
        .unwrap_or_else(|| Dynamic::from(()))
}

fn filter_own_players_by(
    context: NativeCallContext,
    rs: &mut RobotSituation,
    predicate_fn: FnPtr,
) -> Vec<PlayerData> {
    filter_players_by(&context, &rs.world.own_players, predicate_fn)
}

fn filter_opp_players_by(
    context: NativeCallContext,
    rs: &mut RobotSituation,
    predicate_fn: FnPtr,
) -> Vec<PlayerData> {
    filter_players_by(&context, &rs.world.opp_players, predicate_fn)
}

fn filter_players_by(
    context: &NativeCallContext,
    players: &[PlayerData],
    predicate_fn: FnPtr,
) -> Vec<PlayerData> {
    players
        .iter()
        .filter(|player| {
            predicate_fn
                .call_within_context::<bool>(context, ((*player).clone(),))
                .unwrap_or(false)
        })
        .cloned()
        .collect()
}

fn count_own_players_where(
    context: NativeCallContext,
    rs: &mut RobotSituation,
    predicate_fn: FnPtr,
) -> i64 {
    count_players_where(&context, &rs.world.own_players, predicate_fn)
}

fn count_opp_players_where(
    context: NativeCallContext,
    rs: &mut RobotSituation,
    predicate_fn: FnPtr,
) -> i64 {
    count_players_where(&context, &rs.world.opp_players, predicate_fn)
}

fn count_players_where(
    context: &NativeCallContext,
    players: &[PlayerData],
    predicate_fn: FnPtr,
) -> i64 {
    players
        .iter()
        .filter(|player| {
            predicate_fn
                .call_within_context::<bool>(context, ((*player).clone(),))
                .unwrap_or(false)
        })
        .count() as i64
}

// ===== Player Collections =====

fn get_players_within_radius(
    rs: &mut RobotSituation,
    center: Vector2,
    radius: f64,
) -> Vec<PlayerData> {
    rs.world
        .players_within_radius(center, radius)
        .into_iter()
        .cloned()
        .collect()
}

fn get_own_players_within_radius(
    rs: &mut RobotSituation,
    center: Vector2,
    radius: f64,
) -> Vec<PlayerData> {
    rs.world
        .own_players
        .iter()
        .filter(|p| (p.position - center).norm() < radius)
        .cloned()
        .collect()
}

fn get_opp_players_within_radius(
    rs: &mut RobotSituation,
    center: Vector2,
    radius: f64,
) -> Vec<PlayerData> {
    rs.world
        .opp_players
        .iter()
        .filter(|p| (p.position - center).norm() < radius)
        .cloned()
        .collect()
}

// ===== Ray Casting and Prediction =====

fn cast_ray(rs: &mut RobotSituation, from: Vector2, to: Vector2) -> Map {
    let direction = to - from;
    let mut result = Map::new();

    if let Some(intersection) = rs.world.cast_ray(from, direction) {
        result.insert("hit".into(), Dynamic::from(true));
        result.insert("hit_position".into(), Dynamic::from(intersection));

        // Determine what was hit
        if let Some(_ball) = &rs.world.ball {
            // TODO: Check if intersection is near ball
            result.insert("hit_type".into(), Dynamic::from("unknown"));
        } else {
            result.insert("hit_type".into(), Dynamic::from("wall"));
        }
    } else {
        result.insert("hit".into(), Dynamic::from(false));
    }

    result
}

fn predict_ball_position(rs: &mut RobotSituation, time_seconds: f64) -> Dynamic {
    if let Some(prediction) = rs.world.predict_ball_position(time_seconds) {
        match prediction {
            BallPrediction::Linear(pos) => Dynamic::from(pos),
            BallPrediction::Collision(pos) => Dynamic::from(pos),
        }
    } else {
        Dynamic::from(())
    }
}

fn predict_ball_collision_time(rs: &mut RobotSituation) -> f64 {
    // Simple linear prediction to nearest collision
    if let Some(ball) = &rs.world.ball {
        let ball_pos = ball.position.xy();
        let ball_vel = ball.velocity.xy();

        if ball_vel.norm() < 1.0 {
            return f64::INFINITY; // Ball is stationary
        }

        // Check collision with field boundaries
        if let Some(field) = &rs.world.field_geom {
            let half_length = field.field_length / 2.0;
            let half_width = field.field_width / 2.0;

            let time_to_x_boundary = if ball_vel.x > 0.0 {
                (half_length - ball_pos.x) / ball_vel.x
            } else if ball_vel.x < 0.0 {
                (-half_length - ball_pos.x) / ball_vel.x
            } else {
                f64::INFINITY
            };

            let time_to_y_boundary = if ball_vel.y > 0.0 {
                (half_width - ball_pos.y) / ball_vel.y
            } else if ball_vel.y < 0.0 {
                (-half_width - ball_pos.y) / ball_vel.y
            } else {
                f64::INFINITY
            };

            return time_to_x_boundary.min(time_to_y_boundary).max(0.0);
        }
    }

    f64::INFINITY
}

// ===== Vector Math Utilities =====

fn vec2_angle_to(from: &mut Vector2, to: Vector2) -> f64 {
    (to - *from).y.atan2((to - *from).x)
}

fn vec2_distance_to(from: &mut Vector2, to: Vector2) -> f64 {
    (*from - to).norm()
}

fn vec2_rotate(v: &mut Vector2, angle: f64) -> Vector2 {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    Vector2::new(v.x * cos_a - v.y * sin_a, v.x * sin_a + v.y * cos_a)
}

fn vec2_interpolate(from: &mut Vector2, to: Vector2, t: f64) -> Vector2 {
    *from + (to - *from) * t.clamp(0.0, 1.0)
}

fn vec2_halfway_to(from: &mut Vector2, to: Vector2) -> Vector2 {
    (*from + to) * 0.5
}
