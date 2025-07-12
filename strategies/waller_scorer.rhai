fn score_as_waller(s) {
    let ball_pos = s.world.ball.position;
    let goal_pos = s.get_own_goal_position();
    
    // Base score for defender role
    let score = 40.0;
    
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
    
    return score;
}

fn evaluate_waller_positioning(s, ball_pos, goal_pos) {
    let my_pos = s.player.position;
    
    // Check if we're between ball and goal
    let ball_to_goal = goal_pos - ball_pos;
    let ball_to_me = my_pos - ball_pos;
    
    // Project position onto ball-goal line
    let projection = (ball_to_me.x * ball_to_goal.x + ball_to_me.y * ball_to_goal.y) / ball_to_goal.norm();
    let projection_ratio = projection / ball_to_goal.norm();
    
    // Best position is about 30-70% between ball and goal
    if projection_ratio > 0.3 && projection_ratio < 0.7 {
        // Check lateral deviation
        let on_line_pos = ball_pos + ball_to_goal.unit() * projection;
        let deviation = (my_pos - on_line_pos).norm();
        
        return (1.0 - deviation / 1000.0).max(0.0);
    }
    
    return 0.0;
}

fn evaluate_ball_threat(s) {
    if s.world.ball == () { return 0.0; }
    
    let ball_pos = s.world.ball.position;
    let goal_pos = s.get_own_goal_position();
    
    // Distance factor
    let ball_goal_dist = ball_pos.distance_to(goal_pos);
    let dist_threat = (4000.0 - ball_goal_dist.min(4000.0)) / 4000.0;
    
    // Central position factor
    let central_threat = 1.0 - (ball_pos.y.abs() / 3000.0).min(1.0);
    
    // Ball in our half factor
    let half_factor = if s.is_in_defensive_half(ball_pos) { 1.0 } else { 0.3 };
    
    return dist_threat * central_threat * half_factor;
}
