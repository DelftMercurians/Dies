use dies_executor::behavior_tree_api::*;

use crate::v0::utils::{
    find_best_pass_target, find_optimal_shot_target, has_clear_shot, under_pressure,
};

pub fn build_free_kick_kicker_tree(_s: &RobotSituation) -> BehaviorNode {
    sequence_node()
        .add(
            fetch_ball()
                .description("Get ball for free kick".to_string())
                .build(),
        )
        .add(
            select_node()
                .description("Free kick actions")
                .add(
                    // Direct shot if close to goal
                    guard_node()
                        .description("Direct shot?")
                        .condition(|s| {
                            s.distance_to_position(s.get_opp_goal_position()) < 3000.0
                                && has_clear_shot(s)
                        })
                        .then(
                            sequence_node()
                                .description("Execute shot")
                                .add(
                                    face_position(find_optimal_shot_target)
                                        .description("Aim at goal".to_string())
                                        .with_ball()
                                        .build(),
                                )
                                .add(
                                    wait(Argument::Static(0.5))
                                        .description("Aim".to_string())
                                        .build(),
                                )
                                .add(kick().description("Free kick shot!".to_string()).build())
                                .build(),
                        )
                        .build(),
                )
                .add(
                    sequence_node()
                        .add(
                            face_position(find_best_pass_target)
                                .with_ball()
                                .description("Aim pass".to_string())
                                .build(),
                        )
                        .add(kick().description("Free kick pass!".to_string()).build())
                        .description("Execute pass")
                        .build(),
                )
                .description("Pass option")
                .build(),
        )
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
