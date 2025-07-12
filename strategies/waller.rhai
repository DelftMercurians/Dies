fn build_waller_tree() {
    return Select([
        // Block immediate threats
        Guard(|s| is_ball_threatening(s),
            GoToPosition(
                |s| calculate_intercept_position(s),
                #{ heading: |s| get_defender_heading(s) },
                "Block threat"
            ),
            "Ball threatening"
        ),
        
        // Normal wall positioning
        GoToPosition(
            |s| calculate_wall_position(s, 1500.0),
            #{ heading: |s| get_defender_heading(s) },
            "Wall position"
        )
    ], "Waller");
}

fn is_ball_threatening(s) {
    if s.world.ball == () { return false; }
    
    let ball_vel = s.world.ball.velocity;
    let goal_pos = s.get_own_goal_position();
    let ball_pos = s.world.ball.position;
    
    // Check if ball is moving toward our goal
    let ball_to_goal = (goal_pos - ball_pos).unit();
    let vel_direction = ball_vel.unit();
    
    let moving_toward_goal = ball_to_goal.x * vel_direction.x + ball_to_goal.y * vel_direction.y > 0.5;
    let in_our_half = s.is_in_defensive_half(ball_pos);
    let moving_fast = ball_vel.norm() > 500.0;
    
    return moving_toward_goal && in_our_half && moving_fast;
}

// Calculate wall position between ball and goal
fn calculate_wall_position(s, wall_distance) {
    let ball_pos = s.world.ball.position;
    let goal_pos = s.get_own_goal_position();
    
    let ball_to_goal = (goal_pos - ball_pos).unit();
    return ball_pos + ball_to_goal * wall_distance;
}

// Calculate intercept position for moving ball
fn calculate_intercept_position(s) {
    if s.world.ball == () || s.world.ball.velocity.norm() < 100.0 {
        return s.world.ball.position;
    }
    
    let ball_pos = s.world.ball.position;
    let ball_vel = s.world.ball.velocity;
    let player_pos = s.player.position;
    
    // Simple intercept calculation
    let intercept_time = estimate_intercept_time(s, ball_pos, ball_vel, player_pos);
    let predicted_pos = ball_pos + ball_vel * intercept_time;
    
    // Constrain to defensive area
    return constrain_to_defensive_area(s, predicted_pos);
}

// Estimate time to intercept ball
fn estimate_intercept_time(s, ball_pos, ball_vel, player_pos) {
    let max_robot_speed = 3000.0; // mm/s
    
    // Iterative approximation
    let time_estimate = 0.5;
    for i in 0..5 {
        let predicted_ball = ball_pos + ball_vel * time_estimate;
        let distance = (predicted_ball - player_pos).norm();
        time_estimate = distance / max_robot_speed;
    }
    
    return time_estimate.min(3.0); // Cap at 3 seconds
}

// Constrain position to defensive area
fn constrain_to_defensive_area(s, pos) {
    let bounds = s.get_field_bounds();
    
    // Keep in own half with margin
    let x = pos.x.min(0.0);
    let y = pos.y.max(bounds.min_y + 200.0).min(bounds.max_y - 200.0);
    
    return vec2(x, y);
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
