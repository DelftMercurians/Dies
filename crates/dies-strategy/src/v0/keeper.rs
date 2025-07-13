use dies_core::{Angle, Vector2};
use dies_executor::behavior_tree_api::*;

pub fn build_goalkeeper_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            // Penalty mode behavior
            guard_node()
                .condition(|s| s.is_penalty_state())
                .then(
                    go_to_position(|s: &RobotSituation| s.get_own_goal_position())
                        .description("Penalty Defense")
                        .build(),
                )
                .description("Penalty mode")
                .build(),
        )
        .add(
            // Emergency ball clearing if ball is very close
            guard_node()
                .condition(|s| s.ball_in_own_penalty_area() && s.distance_to_ball() < 1000.0)
                .then(
                    sequence_node()
                        .add(fetch_ball().description("Clear Ball".to_string()).build())
                        .add(
                            guard_node()
                                .condition(|s| s.has_ball())
                                .then(
                                    sequence_node()
                                        .add(
                                            face_position(|s: &RobotSituation| {
                                                s.get_field_center()
                                            })
                                            .description("Face Field".to_string())
                                            .build(),
                                        )
                                        .add(kick().description("Clear!".to_string()).build())
                                        .description("Execute Clear")
                                        .build(),
                                )
                                .description("Have ball?")
                                .build(),
                        )
                        .description("Emergency Clear")
                        .build(),
                )
                .description("Ball in penalty area")
                .build(),
        )
        .add(
            // Normal goalkeeper behavior
            go_to_position(calculate_goalkeeper_position)
                .with_heading(get_goalkeeper_heading)
                .description("Guard Goal")
                .build(),
        )
        .description("Goalkeeper")
        .build()
        .into()
}

fn calculate_goalkeeper_position(s: &RobotSituation) -> Vector2 {
    let goal_pos = s.get_own_goal_position();
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        let direction = (ball_pos - goal_pos).normalize();
        goal_pos + direction * 800.0 // 800mm from goal
    } else {
        goal_pos + Vector2::new(500.0, 0.0) // Default position
    }
}

fn get_goalkeeper_heading(s: &RobotSituation) -> Angle {
    if let Some(ball) = &s.world.ball {
        let goal_pos = s.get_own_goal_position();
        let ball_pos = ball.position.xy();
        Angle::between_points(ball_pos, goal_pos)
    } else {
        Angle::from_radians(0.0)
    }
}
