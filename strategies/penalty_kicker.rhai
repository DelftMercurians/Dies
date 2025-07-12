fn build_penalty_kicker_tree() {
    return Semaphore(
        Sequence([
            // Move to penalty position
            GoToPosition(
                |s| s.get_opp_penalty_mark() + vec2(-300.0, 0.0),
                #{ heading: |s| get_heading_to_goal(s) },
                "Approach penalty"
            ),
            
            // Wait for setup
            Wait(1.5, "Wait for setup"),
            
            // Execute penalty
            FetchBall("Get ball"),
            Guard(|s| s.has_ball(),
                Sequence([
                    FaceTowardsPosition(|s| choose_penalty_target(s), #{}, "Aim penalty"),
                    Wait(0.8, "Final aim"),
                    Kick("Penalty shot!")
                ], "Execute penalty"),
                "Have ball?"
            )
        ], "Penalty sequence"),
        "penalty_kicker",
        1,
        "Penalty kicker"
    );
}

fn score_penalty_kicker(s) {
    let score = 90.0;
    
    // Prefer robots closer to penalty area
    let penalty_dist = s.distance_to_opp_penalty_area();
    score += (2000.0 - penalty_dist.min(2000.0)) / 25.0;
    
    // Consistent assignment based on robot hash
    score += s.player_id.hash_float() * 10.0;
    
    return score;
}

fn choose_penalty_target(s) {
    let goal_corners = s.get_opp_goal_corners();
    if goal_corners.len() >= 2 {
        if s.player_id.hash_float() > 0.5 {
            return goal_corners[0];
        } else {
            return goal_corners[1];
        }
    }
    return s.get_opp_goal_position();
}


fn get_heading_to_goal(s) {
    let player_pos = s.player.position;
    let goal_pos = s.get_opp_goal_position();
    return (goal_pos - player_pos).angle_to(vec2(1.0, 0.0));
}