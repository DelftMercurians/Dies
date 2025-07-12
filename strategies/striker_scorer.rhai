// Score for striker role assignment
fn score_striker(s) {
    let score = 50.0;
    
    // Prefer robots closer to ball
    let ball_dist = s.distance_to_ball();
    score += (2000.0 - ball_dist.min(2000.0)) / 20.0;
    
    // Prefer robots in attacking position
    if s.is_in_attacking_half(s.player.position) {
        score += 20.0;
    }
    
    // Higher score if we have the ball
    if s.has_ball() {
        score += 30.0;
    }
    
    // Slight preference based on robot hash for consistency
    score += s.player_id.hash_float() * 5.0;
    
    return score;
}

// Score for shooting vs passing decision
fn score_for_shooting(s) {
    if !s.has_ball() {
        return 0.0;
    }
    
    let goal_pos = s.get_opp_goal_position();
    let my_pos = s.player.position;
    let goal_dist = my_pos.distance_to(goal_pos);
    
    // Base score from distance (closer = better)
    let distance_score = (4000.0 - goal_dist.min(4000.0)) / 40.0;
    
    // Check shot quality
    let shot_quality = evaluate_shot_quality(s, my_pos, goal_pos);
    
    return distance_score * shot_quality;
}

fn score_for_passing(s) {
    if !s.has_ball() {
        return 0.0;
    }
    
    let teammates = s.filter_own_players_by(|p| p.id != s.player_id);
    if teammates.len() == 0 {
        return 0.0;
    }
    
    let best_pass_score = 0.0;
    
    for teammate in teammates {
        // Check if pass path is clear
        let pass_clear = evaluate_shot_quality(s, s.player.position, teammate.position);
        if pass_clear > 0.5 {
            // Evaluate teammate's position
            let teammate_shot_potential = evaluate_teammate_shot_potential(s, teammate);
            let pass_score = pass_clear * teammate_shot_potential;
            
            if pass_score > best_pass_score {
                best_pass_score = pass_score;
            }
        }
    }
    
    // Bonus if under pressure
    if is_under_heavy_pressure(s) {
        best_pass_score += 30.0;
    }
    
    return best_pass_score;
}

// Helper to evaluate teammate's shooting potential
fn evaluate_teammate_shot_potential(s, teammate) {
    let goal_pos = s.get_opp_goal_position();
    let teammate_goal_dist = teammate.position.distance_to(goal_pos);
    
    // Closer to goal = better
    let position_score = (3000.0 - teammate_goal_dist.min(3000.0)) / 30.0;
    
    // Check if teammate has clear shot
    let shot_quality = evaluate_shot_quality(s, teammate.position, goal_pos);
    
    return position_score * shot_quality;
}

fn is_under_heavy_pressure(s) {
    let nearby_opponents = s.get_opp_players_within_radius(s.player.position, 600.0);
    return nearby_opponents.len() >= 2;
}

// Evaluate shot quality from a position to a target
fn evaluate_shot_quality(s, from_pos, to_pos) {
    // Cast multiple rays to check for obstructions
    let ray_offsets = [
        vec2(0.0, 0.0),
        vec2(0.0, 50.0),
        vec2(0.0, -50.0),
        vec2(0.0, 100.0),
        vec2(0.0, -100.0)
    ];
    
    let clear_count = 0;
    for offset in ray_offsets {
        let target = to_pos + offset;
        let ray_result = s.cast_ray(from_pos, target);
        if !ray_result.hit {
            clear_count += 1;
        }
    }
    
    return clear_count / ray_offsets.len();
}

// Score for free kicker role
fn score_free_kicker(s) {
    let score = 60.0;
    
    // Prefer robots closest to ball
    let ball_dist = s.distance_to_ball();
    score += (1000.0 - ball_dist.min(1000.0)) / 10.0;
    
    // Prefer robots not under pressure
    let opp_nearby = s.get_opp_players_within_radius(s.player.position, 1000.0);
    if opp_nearby.len() == 0 {
        score += 20.0;
    }
    
    // Prefer robots with good kicking angle to goal
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