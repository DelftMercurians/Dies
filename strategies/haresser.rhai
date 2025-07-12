fn build_harasser_tree() {
    return Select(
        |s| {
            // Find all threatening opponents in our half
            let threats = s.filter_opp_players_by(|p| {
                s.is_in_defensive_half(p.position) && 
                p.position.distance_to(s.get_own_goal_position()) < 4000.0
            });
            
            // Create behavior options for each threat
            let options = [];
            for opponent in threats {
                // Each option tries to mark this specific opponent
                options.push(
                    Semaphore(
                        GoToPosition(
                            calculate_harasser_position(s, opponent),
                            #{ heading: |s| get_defender_heading(s) },
                            "Mark opponent " + opponent.id.to_string()
                        ),
                        "tag_opponent_" + opponent.id.to_string(),
                        1  // Only one defender can mark each opponent
                    )
                );
            }
            
            // Add fallback behavior if no threats to mark
            if options.len() == 0 {
                options.push(GoToPosition(
                    |s| fallback_defender_position(s),
                    #{ heading: |s| get_defender_heading(s) },
                    "Defend position"
                ));
            }
            
            return options;
        },
        "Dynamic Harasser"
    );
}

fn calculate_harasser_position(s, target_opponent) {
    let goal_pos = s.get_own_goal_position();
    let opp_pos = target_opponent.position;
    
    let opp_to_goal = (goal_pos - opp_pos).unit();
    let marking_distance = 600.0;
    
    return opp_pos + opp_to_goal * marking_distance;
}

fn get_defender_heading(s) {
    if s.world.ball == () {
        return 0.0;
    }
    
    // Face the ball
    let ball_pos = s.world.ball.position;
    let my_pos = s.player.position;
    return (ball_pos - my_pos).angle_to(vec2(1.0, 0.0));
}

fn fallback_defender_position(s) {
    let goal_pos = s.get_own_goal_position();
    
    let ball_pos = s.world.ball.position;
    let ball_to_goal = goal_pos - ball_pos;
    
    // Position between ball and goal
    let defend_ratio = 0.4; // 40% of the way from ball to goal
    let base_pos = ball_pos + ball_to_goal * defend_ratio;
    
    return base_pos;
}