use dies_core::Vector2;
use dies_executor::behavior_tree_api::*;

use crate::v0::utils::{calculate_intercept_position, evaluate_ball_threat, get_defender_heading};

pub fn build_waller_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            // Block immediate threats
            guard_node()
                .condition(is_ball_threatening)
                .then(
                    go_to_position(Argument::callback(|s| {
                        constrain_to_defensive_area(s, calculate_intercept_position(s))
                    }))
                    .with_heading(get_defender_heading)
                    .description("Block threat")
                    .build(),
                )
                .description("Ball threatening")
                .build(),
        )
        .add(
            // Normal wall positioning
            go_to_position(Argument::callback(|s| calculate_wall_position(s, 1500.0)))
                .with_heading(Argument::callback(get_defender_heading))
                .description("Wall position".to_string())
                .build(),
        )
        .description("Waller")
        .build()
        .into()
}

pub fn score_as_waller(s: &RobotSituation) -> f64 {
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        let goal_pos = s.get_own_goal_position();

        // Base score for defender role
        let mut score = 40.0;

        // Higher score if ball is in central threatening position
        let ball_threat = evaluate_ball_threat(s);
        score += ball_threat * 30.0;

        // Higher score if already positioned between ball and goal
        let positioning_score = evaluate_waller_positioning(s, ball_pos, goal_pos);
        score += positioning_score * 25.0;

        // Lower score if too far from defensive area
        let goal_dist = s.distance_to_position(goal_pos);
        if goal_dist > 3000.0 {
            score -= 20.0;
        }

        score
    } else {
        0.0
    }
}

fn is_ball_threatening(s: &RobotSituation) -> bool {
    if let Some(ball) = &s.world.ball {
        let ball_vel = ball.velocity;
        let goal_pos = s.get_own_goal_position();
        let ball_pos = ball.position.xy();

        // Check if ball is moving toward our goal
        let ball_to_goal = (goal_pos - ball_pos).normalize();
        let vel_direction = ball_vel.normalize();

        let moving_toward_goal =
            ball_to_goal.x * vel_direction.x + ball_to_goal.y * vel_direction.y > 0.5;
        let in_our_half = ball_pos.x < 0.0;
        let moving_fast = ball_vel.norm() > 500.0;

        moving_toward_goal && in_our_half && moving_fast
    } else {
        false
    }
}

fn calculate_wall_position(s: &RobotSituation, wall_distance: f64) -> Vector2 {
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        let goal_pos = s.get_own_goal_position();

        let ball_to_goal = (goal_pos - ball_pos).normalize();
        ball_pos + ball_to_goal * wall_distance
    } else {
        s.player_data().position
    }
}

fn constrain_to_defensive_area(s: &RobotSituation, pos: Vector2) -> Vector2 {
    if let Some(field) = &s.world.field_geom {
        let half_width = field.field_width / 2.0;

        // Keep in own half with margin
        let x = pos.x.min(0.0);
        let y = pos.y.max(-half_width + 200.0).min(half_width - 200.0);

        Vector2::new(x, y)
    } else {
        pos
    }
}

fn evaluate_waller_positioning(s: &RobotSituation, ball_pos: Vector2, goal_pos: Vector2) -> f64 {
    let my_pos = s.player_data().position;

    // Check if we're between ball and goal
    let ball_to_goal = goal_pos - ball_pos;
    let ball_to_me = my_pos - ball_pos;

    // Project position onto ball-goal line
    let projection =
        (ball_to_me.x * ball_to_goal.x + ball_to_me.y * ball_to_goal.y) / ball_to_goal.norm();
    let projection_ratio = projection / ball_to_goal.norm();

    // Best position is about 30-70% between ball and goal
    if projection_ratio > 0.3 && projection_ratio < 0.7 {
        // Check lateral deviation
        let on_line_pos = ball_pos + ball_to_goal.normalize() * projection;
        let deviation = (my_pos - on_line_pos).norm();

        (1.0 - deviation / 1000.0).max(0.0)
    } else {
        0.0
    }
}
