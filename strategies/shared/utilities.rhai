// Goal positions
fn get_opponent_goal() { return vec2(6000.0, 0.0); }
fn get_own_goal() { return vec2(-6000.0, 0.0); }

// Ball position helpers
fn get_ball_pos(s) {
    if s.world.ball == () {
        throw "Ball information not available in get_ball_pos.";
    }
    return s.world.ball.position;
}

fn get_ball_velocity(s) {
    if s.world.ball == () {
        return vec2(0.0, 0.0);
    }
    return s.world.ball.velocity;
}

// Helper to check if we're on the attacking team for this game state
fn we_are_attacking(s) {
    return s.world.game_state.us_operating;
}

// Distance calculations
fn distance_to_ball(s) {
    if s.world.ball == () { return 10000.0; }
    let player_pos = s.player().position;
    let ball_pos = s.world.ball.position;
    let dx = ball_pos.x - player_pos.x;
    let dy = ball_pos.y - player_pos.y;
    return (dx*dx + dy*dy).sqrt();
}

fn distance_to_own_goal(s) {
    let player_pos = s.player().position;
    let goal = get_own_goal();
    let dx = goal.x - player_pos.x;
    let dy = goal.y - player_pos.y;
    return (dx*dx + dy*dy).sqrt();
}

// Role scoring functions for dynamic role assignment
fn score_as_attacker(s) {
    let ball_dist = distance_to_ball(s);
    let base_score = 100.0 / (1.0 + ball_dist / 1000.0);
    
    // Bonus if ball is in opponent half
    if s.world.ball != () && s.world.ball.position.x > 0.0 {
        return base_score + 20.0;
    }
    return base_score;
}

fn score_as_supporter(s) {
    let ball_dist = distance_to_ball(s);
    let base_score = 50.0 / (1.0 + ball_dist / 2000.0);
    
    // Bonus if we're not closest to ball (let attacker get it)
    if ball_dist > 1500.0 {
        return base_score + 15.0;
    }
    return base_score;
}

fn score_as_defender(s) {
    let goal_dist = distance_to_own_goal(s);
    let base_score = 30.0 / (1.0 + goal_dist / 3000.0);
    
    // Higher score if ball is in our half
    if s.world.ball != () && s.world.ball.position.x < 0.0 {
        return base_score + 25.0;
    }
    return base_score;
}

// Dynamic positioning functions
fn get_supporter_pos(s) {
    if s.world.ball == () {
        return vec2(-2000.0, 0.0);
    }
    let ball_pos = s.world.ball.position;
    // Position 2m behind ball, ready for passes or to take over
    return vec2(ball_pos.x - 2000.0, ball_pos.y * 0.5);
}

fn get_defender_pos(s) {
    let own_goal = get_own_goal();
    let player_id_hash = hash_float(s.player_id);
    
    if s.world.ball == () {
        let spread_y = (player_id_hash - 0.5) * 3000.0;
        return vec2(own_goal.x + 2000.0, spread_y);
    }
    
    let ball_pos = s.world.ball.position;
    let ball_angle = ball_pos.y.atan(ball_pos.x - own_goal.x);
    let defend_radius = 2500.0;
    let defend_x = own_goal.x + defend_radius * ball_angle.cos();
    let defend_y = defend_radius * ball_angle.sin();
    
    return vec2(defend_x, defend_y);
}

fn get_heading_to_goal(s) {
    let player_pos = s.player().position;
    let goal = get_opponent_goal();
    let dx = goal.x - player_pos.x;
    let dy = goal.y - player_pos.y;
    return dy.atan(dx);
}

// Player-specific positioning functions
fn get_kickoff_supporter_pos(player_id) {
    // Position in own half, spread out, ready to support
    let player_hash = hash_float(player_id);
    let spread_x = -3000.0 + player_hash * 2000.0; // Between -3000 and -1000
    let spread_y = (player_hash - 0.5) * 4000.0;    // Between -2000 and 2000
    
    return vec2(spread_x, spread_y);
}

fn get_penalty_support_pos(player_id) {
    // Rule: All other players must be 1m behind ball
    let player_hash = hash_float(player_id);
    let spread_y = (player_hash - 0.5) * 6000.0; // Spread across field width
    
    return vec2(1000.0, spread_y); // 1m behind penalty mark
}

// Attacker helper functions
fn close_to_goal(s) {
    let player_pos = s.player().position;
    let goal = get_opponent_goal();
    let dist = (goal.x - player_pos.x).abs() + (goal.y - player_pos.y).abs();
    return dist < 3000.0;
}

fn advance_toward_goal(s) {
    let player_pos = s.player().position;
    let goal = get_opponent_goal();
    let advance_factor = 0.3;
    
    let target_x = player_pos.x + (goal.x - player_pos.x) * advance_factor;
    let target_y = player_pos.y + (goal.y - player_pos.y) * advance_factor;
    
    return vec2(target_x, target_y);
} 