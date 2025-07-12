import "shared/situations" as sit;

fn build_goalkeeper_tree() {
    return Select([
        // Penalty mode behavior
        Guard(|s| sit::is_penalty_state(s),
            GoToPosition(
                |s| s.get_own_goal_position(),
                #{ heading: 0.0 },
                "Penalty Defense"
            ),
            "Penalty mode"
        ),
        
        // Emergency ball clearing if ball is very close
        Guard(|s| sit::ball_in_own_penalty_area(s) && s.distance_to_ball() < 1000.0,
            Sequence([
                FetchBall("Clear Ball"),
                Guard(|s| s.has_ball(),
                    Sequence([
                        FaceTowardsPosition(|s| s.get_field_center(), #{}, "Face Field"),
                        Kick("Clear!")
                    ], "Execute Clear"),
                    "Have ball?"
                )
            ], "Emergency Clear"),
            "Ball in penalty area"
        ),
        
        // Normal goalkeeper behavior
        GoToPosition(
            |s| calculate_goalkeeper_position(s),
            #{ heading: |s| get_goalkeeper_heading(s) },
            "Guard Goal"
        )
    ], "Goalkeeper");
}

fn calculate_goalkeeper_position(s) {
    let goal_pos = s.get_own_goal_position();
    if s.world.ball == () {
        return goal_pos + vec2(500.0, 0.0);  // Default position
    }
    
    let ball_pos = s.world.ball.position;
    let direction = (ball_pos - goal_pos).unit();
    return goal_pos + direction * 800.0;  // 800mm from goal
}

fn get_goalkeeper_heading(s) {
    if s.world.ball == () { return 0.0; }
    let goal_pos = s.get_own_goal_position();
    let ball_pos = s.world.ball.position;
    let dx = ball_pos.x - goal_pos.x;
    let dy = ball_pos.y - goal_pos.y;
    return dy.atan(dx);
}