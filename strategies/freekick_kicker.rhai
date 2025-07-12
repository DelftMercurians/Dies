fn build_free_kick_kicker_tree() {
    return Semaphore(
        Select([
            // Direct shot if close to goal
            Guard(|s| s.distance_to_position(s.get_opp_goal_position()) < 3000.0 && has_clear_shot(s),
                Sequence([
                    FetchBall("Get ball for free kick"),
                    Guard(|s| s.has_ball(),
                        Sequence([
                            FaceTowardsPosition(|s| find_optimal_shot_target(s), #{}, "Aim at goal"),
                            Wait(0.5, "Aim"),
                            Kick("Free kick shot!")
                        ], "Execute shot"),
                        "Have ball?"
                    )
                ], "Direct shot"),
                "Close shot opportunity"
            ),
            
            // Pass to teammate
            Guard(|s| can_pass_to_teammate(s),
                Sequence([
                    FetchBall("Get ball for free kick"),
                    Guard(|s| s.has_ball(),
                        Sequence([
                            FaceTowardsPosition(|s| find_best_pass_target(s), #{}, "Aim pass"),
                            Wait(0.3, "Aim pass"),
                            Kick("Free kick pass!")
                        ], "Execute pass"),
                        "Have ball?"
                    )
                ], "Pass option"),
                "Can pass to teammate"
            ),
            
            // Default: advance with ball
            Sequence([
                FetchBall("Get ball for free kick"),
                Guard(|s| s.has_ball(),
                    GoToPosition(
                        |s| calculate_advance_position(s),
                        #{ heading: |s| get_heading_to_goal(s), with_ball: true },
                        "Advance with ball"
                    ),
                    "Have ball?"
                )
            ], "Advance")
        ], "Free kick actions"),
        "free_kick_kicker",
        1,
        "Free kick kicker"
    );
}

fn score_free_kick_kicker(s) {
    let score = 80.0;
    
    // Prefer robots closest to ball
    let ball_dist = s.distance_to_ball();
    score += (1000.0 - ball_dist.min(1000.0)) / 10.0;
    
    // Prefer robots with good angle to goal
    if s.world.ball != () {
        let goal_pos = s.get_opp_goal_position();
        let ball_pos = s.world.ball.position;
        let ball_to_goal = (goal_pos - ball_pos).unit();
        let robot_to_ball = (ball_pos - s.player.position).unit();
        let angle_alignment = ball_to_goal.x * robot_to_ball.x + ball_to_goal.y * robot_to_ball.y;
        score += angle_alignment * 15.0;
    }
    
    return score;
}

fn has_clear_shot(s) {
    let shot_target = find_optimal_shot_target(s);
    let ray_result = s.cast_ray(s.player.position, shot_target);
    return !ray_result.hit;
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

fn find_optimal_shot_target(s) {
    let goal_corners = s.get_opp_goal_corners();
    if goal_corners.len() >= 2 {
        // Simple strategy: choose corner based on robot hash
        if s.player_id.hash_float() > 0.5 {
            return goal_corners[0];
        } else {
            return goal_corners[1];
        }
    }
    return s.get_opp_goal_position();
}

fn calculate_advance_position(s) {
    let player_pos = s.player.position;
    let goal_pos = s.get_opp_goal_position();
    let direction = (goal_pos - player_pos).unit();
    return player_pos + direction * 1000.0;
}

fn get_heading_to_goal(s) {
    let player_pos = s.player.position;
    let goal_pos = s.get_opp_goal_position();
    return (goal_pos - player_pos).angle_to(vec2(1.0, 0.0));
}
