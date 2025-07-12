fn build_kickoff_kicker_tree() {
    return Semaphore(
        Sequence([
            GoToPosition(|s| s.get_field_center(), #{}, "Move to center"),
            Wait(0.5, "Wait for start"),
            FetchBall("Get ball"),
            Guard(|s| s.has_ball(),
                Select([
                    // Try to pass if teammate available
                    Guard(|s| can_pass_to_teammate(s),
                        Sequence([
                            FaceTowardsPosition(|s| find_best_pass_target(s), #{}, "Aim pass"),
                            Kick("Kickoff pass!")
                        ], "Pass kickoff"),
                        "Can pass"
                    ),
                    // Otherwise kick toward goal
                    Sequence([
                        FaceTowardsPosition(|s| s.get_opp_goal_position(), #{}, "Face goal"),
                        Kick("Kickoff!")
                    ], "Direct kickoff")
                ], "Kickoff options"),
                "Have ball?"
            )
        ], "Kickoff sequence"),
        "kickoff_kicker",
        1,
        "Kickoff kicker"
    );
}

fn score_kickoff_kicker(s) {
    let score = 80.0;
    
    // Prefer robots closer to center
    let center_dist = s.distance_to_position(s.get_field_center());
    score += (1000.0 - center_dist.min(1000.0)) / 20.0;
    
    // Prefer robots not under pressure
    let nearby_opponents = s.get_opp_players_within_radius(s.player.position, 1000.0);
    if nearby_opponents.len() == 0 {
        score += 15.0;
    }
    
    return score;
}


fn can_pass_to_teammate(s) {
    if !s.has_ball() { return false; }
    
    let teammates = s.filter_own_players_by(|p| p.id != s.player_id);
    
    for teammate in teammates {
        let ray_result = s.cast_ray(s.world.ball.position, teammate.position);
        if !ray_result.hit {
            return true;
        }
    }
    return false;
}

fn find_best_pass_target(s) {
    let teammates = s.filter_own_players_by(|p| p.id != s.player_id);
    let ball_pos = s.world.ball.position;
    
    let best_target = s.get_opp_goal_position();
    let best_score = -1.0;
    
    for teammate in teammates {
        let ray_result = s.cast_ray(ball_pos, teammate.position);
        if !ray_result.hit {
            let goal_dist = teammate.position.distance_to(s.get_opp_goal_position());
            let score = 10000.0 - goal_dist;
            
            if score > best_score {
                best_score = score;
                best_target = teammate.position;
            }
        }
    }
    
    return best_target;
}
