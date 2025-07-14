use dies_executor::behavior_tree_api::*;

use crate::v0::{
    striker::calculate_striker_advance_position,
    utils::{
        can_pass_to_teammate, find_best_pass_target, find_optimal_shot_target, get_heading_to_goal,
        has_clear_shot, under_pressure,
    },
};

pub fn build_free_kick_kicker_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            // Direct shot if close to goal
            guard_node()
                .condition(|s| {
                    s.distance_to_position(s.get_opp_goal_position()) < 3000.0 && has_clear_shot(s)
                })
                .then(
                    sequence_node()
                        .add(
                            fetch_ball()
                                .description("Get ball for free kick".to_string())
                                .build(),
                        )
                        .add(
                            guard_node()
                                .condition(|s| s.has_ball())
                                .then(
                                    sequence_node()
                                        .add(
                                            face_position(find_optimal_shot_target)
                                                .with_ball()
                                                .description("Aim at goal".to_string())
                                                .build(),
                                        )
                                        .add(
                                            wait(Argument::Static(0.5))
                                                .description("Aim".to_string())
                                                .build(),
                                        )
                                        .add(
                                            kick()
                                                .description("Free kick shot!".to_string())
                                                .build(),
                                        )
                                        .description("Execute shot")
                                        .build(),
                                )
                                .description("Have ball?")
                                .build(),
                        )
                        .description("Direct shot")
                        .build(),
                )
                .description("Close shot opportunity")
                .build(),
        )
        .add(
            // Pass to teammate
            guard_node()
                .condition(can_pass_to_teammate)
                .then(
                    sequence_node()
                        .add(
                            fetch_ball()
                                .description("Get ball for free kick".to_string())
                                .build(),
                        )
                        .add(
                            guard_node()
                                .condition(|s| s.has_ball())
                                .then(
                                    sequence_node()
                                        .add(
                                            face_position(find_best_pass_target)
                                                .with_ball()
                                                .description("Aim pass".to_string())
                                                .build(),
                                        )
                                        .add(
                                            wait(Argument::Static(0.3))
                                                .description("Aim pass".to_string())
                                                .build(),
                                        )
                                        .add(
                                            kick()
                                                .description("Free kick pass!".to_string())
                                                .build(),
                                        )
                                        .description("Execute pass")
                                        .build(),
                                )
                                .description("Have ball?")
                                .build(),
                        )
                        .description("Pass option")
                        .build(),
                )
                .description("Can pass to teammate")
                .build(),
        )
        .add(
            // Default: advance with ball
            sequence_node()
                .add(
                    fetch_ball()
                        .description("Get ball for free kick".to_string())
                        .build(),
                )
                .add(
                    guard_node()
                        .condition(|s| s.has_ball())
                        .then(
                            go_to_position(calculate_striker_advance_position)
                                .with_heading(get_heading_to_goal)
                                .with_ball()
                                .description("Advance with ball")
                                .build(),
                        )
                        .description("Have ball?")
                        .build(),
                )
                .description("Advance")
                .build(),
        )
        .description("Free kick actions")
        .build()
        .into()
}

pub fn score_kickoff_kicker(s: &RobotSituation) -> f64 {
    let mut score = 80.0;

    // Prefer robots closer to center
    let center_dist = s.distance_to_position(s.get_field_center());
    score += (1000.0 - center_dist.min(1000.0)) / 20.0;

    // Prefer robots not under pressure
    if !under_pressure(s) {
        score += 15.0;
    }

    score
}

pub fn score_free_kick_kicker(s: &RobotSituation) -> f64 {
    let mut score = 80.0;

    // Prefer robots closest to ball
    let ball_dist = s.distance_to_ball();
    score += (1000.0 - ball_dist.min(1000.0)) / 10.0;

    // Prefer robots with good angle to goal
    if let Some(ball) = &s.world.ball {
        let goal_pos = s.get_opp_goal_position();
        let ball_pos = ball.position.xy();
        let ball_to_goal = (goal_pos - ball_pos).normalize();
        let robot_to_ball = (ball_pos - s.player_data().position).normalize();
        let angle_alignment = ball_to_goal.x * robot_to_ball.x + ball_to_goal.y * robot_to_ball.y;
        score += angle_alignment * 15.0;
    }

    score
}
