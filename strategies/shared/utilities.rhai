// Updated utilities using modern RobotSituation API

// Legacy goal position functions kept for compatibility
fn opp_goal_center() { vec2(4500.0, 0.0) }  // Using standard SSL field dimensions
fn opp_goal_inner_center() { vec2(4500.0 + 180.0, 0.0) }  // Goal depth
fn opp_goal_top() { vec2(4500.0, 350.0) }  // Half goal width
fn opp_goal_bottom() { vec2(4500.0, -350.0) }

fn own_goal_center() { vec2(-4500.0, 0.0) }
fn own_goal_inner_center() { vec2(-4500.0 - 180.0, 0.0) }
fn own_goal_top() { vec2(-4500.0, 350.0) }
fn own_goal_bottom() { vec2(-4500.0, -350.0) }

// Ball position helpers - now using new API
fn get_ball_pos(s) {
    return s.world.ball.position;
}

fn get_ball_velocity(s) {
    return s.world.ball.velocity;
}

// Helper to check if we're on the attacking team for this game state
fn we_are_attacking(s) {
    return s.world.game_state.us_operating;
}

// Distance calculations - replaced with new API methods
fn distance_to_ball(s) {
    return s.distance_to_ball();
}

fn distance_to_own_goal(s) {
    return s.distance_to_position(s.get_own_goal_position());
}

fn distance_to_opp_goal(s) {
    return s.distance_to_position(s.get_opp_goal_position());
}

// Role scoring functions for role assignment
fn score_striker(s) {
    let score = 50.0;  // Base score
    
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

fn score_defender(s) {
    let score = 40.0;  // Lower base score than striker
    
    // Prefer robots closer to our goal
    let goal_dist = s.distance_to_position(s.get_own_goal_position());
    score += (3000.0 - goal_dist.min(3000.0)) / 30.0;
    
    // Higher score if ball is in our half
    if s.is_in_defensive_half(s.world.ball.position) {
        score += 25.0;
    }
    
    // Prefer robots not carrying ball
    if !s.has_ball() {
        score += 15.0;
    }
    
    return score;
}

fn score_support(s) {
    let score = 30.0;  // Base score
    
    // Prefer robots at medium distance from ball (not too close, not too far)
    let ball_dist = s.distance_to_ball();
    let optimal_dist = 1500.0;  // 1.5m from ball
    let dist_penalty = (ball_dist - optimal_dist).abs() / 100.0;
    score -= dist_penalty.min(20.0);
    
    // Slight randomization for role diversity
    score += s.player_id.hash_float() * 10.0;
    
    return score;
}

// Special role scoring functions
fn score_free_kicker(s) {
    let score = 60.0;  // High base score for special role
    
    // Prefer robots closest to ball
    let ball_dist = s.distance_to_ball();
    score += (1000.0 - ball_dist.min(1000.0)) / 10.0;
    
    // Prefer robots not under pressure
    let opp_nearby = s.get_opp_players_within_radius(s.player.position, 1000.0);
    if opp_nearby.len() == 0 {
        score += 20.0;
    }
    
    // Prefer robots with good kicking angle to goal
    let goal_pos = s.get_opp_goal_position();
    let ball_pos = s.world.ball.position;
    let ball_to_goal = (goal_pos - ball_pos).unit();
    let robot_to_ball = (ball_pos - s.player.position).unit();
    let angle_alignment = ball_to_goal.x * robot_to_ball.x + ball_to_goal.y * robot_to_ball.y;
    score += angle_alignment * 15.0;
    
    return score;
}

fn score_penalty_taker(s) {
    let score = 80.0;  // Very high base score for penalty taker
    
    // Prefer robots closer to penalty area
    let penalty_dist = s.distance_to_opp_penalty_area();
    score += (2000.0 - penalty_dist.min(2000.0)) / 25.0;
    
    // Consistent assignment based on robot hash
    score += s.player_id.hash_float() * 10.0;
    
    return score;
}

// Enhanced role scoring functions using new API
fn score_as_attacker(s) {
    let ball_dist = s.distance_to_ball();
    let base_score = 100.0 / (1.0 + ball_dist / 1000.0);
    
    // Bonus if ball is in opponent half
    if s.is_in_attacking_half(s.world.ball.position) {
        return base_score + 20.0;
    }
    
    // Bonus if robot has the ball
    if s.has_ball() {
        return base_score + 30.0;
    }
    
    return base_score;
}

fn score_as_supporter(s) {
    let ball_dist = s.distance_to_ball();
    let base_score = 50.0 / (1.0 + ball_dist / 2000.0);
    
    // Prefer not being closest to ball (let attacker get it)
    let closest_to_ball = s.closest_own_player_to_ball();
    if closest_to_ball != () && closest_to_ball.id != s.player_id {
        return base_score + 15.0;
    }
    
    return base_score;
}

fn score_as_defender(s) {
    let goal_dist = s.distance_to_position(s.get_own_goal_position());
    let base_score = 30.0 / (1.0 + goal_dist / 3000.0);
    
    // Higher score if ball is in our half
    if s.is_in_defensive_half(s.world.ball.position) {
        return base_score + 25.0;
    }
    
    // Prefer not having the ball (let attackers handle it)
    if !s.has_ball() {
        return base_score + 10.0;
    }
    
    return base_score;
}

// Position calculation functions using new API
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

fn calculate_defender_position(s) {
    let goal_pos = s.get_own_goal_position();
    if s.world.ball == () {
        let spread_y = (s.player_id.hash_float() - 0.5) * 3000.0;
        return goal_pos + vec2(2000.0, spread_y);
    }
    
    let ball_pos = s.world.ball.position;
    let ball_to_goal = goal_pos - ball_pos;
    let defend_pos = ball_pos + ball_to_goal.unit() * 1500.0;  // 1.5m toward goal from ball
    return defend_pos;
}

fn get_defender_heading(s) {
    if s.world.ball == () { return 0.0; }
    return get_heading_to_ball(s);
}

fn calculate_support_position(s) {
    if s.world.ball == () {
        return vec2(-2000.0, 0.0);
    }
    
    let ball_pos = s.world.ball.position;
    let goal_pos = s.get_opp_goal_position();
    
    // Position to support attack or defend as needed
    if s.is_in_attacking_half(ball_pos) {
        // Support attack from side
        let side_offset = (s.player_id.hash_float() - 0.5) * 3000.0;
        return ball_pos + vec2(-1000.0, side_offset);
    } else {
        // Support defense
        return ball_pos + vec2(-800.0, 0.0);
    }
}

fn calculate_advance_position(s) {
    let player_pos = s.player.position;
    let goal_pos = s.get_opp_goal_position();
    let advance_factor = 0.3;
    
    let target_x = player_pos.x + (goal_pos.x - player_pos.x) * advance_factor;
    let target_y = player_pos.y + (goal_pos.y - player_pos.y) * advance_factor;
    
    return vec2(target_x, target_y);
}

fn get_heading_to_goal(s) {
    let player_pos = s.player.position;
    let goal_pos = s.get_opp_goal_position();
    let dx = goal_pos.x - player_pos.x;
    let dy = goal_pos.y - player_pos.y;
    return dy.atan(dx);
}

fn get_heading_to_ball(s) {
    if s.world.ball == () { return 0.0; }
    let player_pos = s.player.position;
    let ball_pos = s.world.ball.position;
    let dx = ball_pos.x - player_pos.x;
    let dy = ball_pos.y - player_pos.y;
    return dy.atan(dx);
}

fn get_kickoff_support_position(s) {
    // Position in own half, spread out, ready to support
    let player_hash = s.player_id.hash_float();
    let spread_x = -3000.0 + player_hash * 2000.0; // Between -3000 and -1000
    let spread_y = (player_hash - 0.5) * 4000.0;    // Between -2000 and 2000
    
    return vec2(spread_x, spread_y);
}

fn get_penalty_support_position(s) {
    // Rule: All other players must be 1m behind ball
    let player_hash = s.player_id.hash_float();
    let spread_y = (player_hash - 0.5) * 6000.0; // Spread across field width
    
    return vec2(1000.0, spread_y); // 1m behind penalty mark
}

// Enhanced positioning functions using new API
fn get_supporter_pos(s) {
    if s.world.ball == () {
        return vec2(-2000.0, 0.0);
    }
    
    let ball_pos = s.world.ball.position;
    let goal_pos = s.get_opp_goal_position();
    
    // Position to support the attack - between ball and goal but offset
    let support_offset = vec2(-1500.0, (s.player_id.hash_float() - 0.5) * 2000.0);
    return ball_pos + support_offset;
}

fn get_defender_pos(s) {
    let own_goal = s.get_own_goal_position();
    
    if s.world.ball == () {
        // Default spread formation
        let spread_y = (s.player_id.hash_float() - 0.5) * 3000.0;
        return own_goal + vec2(2000.0, spread_y);
    }
    
    let ball_pos = s.world.ball.position;
    
    // Position between ball and goal for interception
    let ball_to_goal = (own_goal - ball_pos).unit();
    let defend_radius = 2000.0;
    
    return ball_pos + ball_to_goal * defend_radius;
}

// Enhanced helper functions using new API
fn close_to_goal(s) {
    return s.distance_to_position(s.get_opp_goal_position()) < 3000.0;
}

fn close_to_own_goal(s) {
    return s.distance_to_position(s.get_own_goal_position()) < 3000.0;
}

fn advance_toward_goal(s) {
    let player_pos = s.player.position;
    let goal_pos = s.get_opp_goal_position();
    let advance_factor = 0.3;
    
    let direction = (goal_pos - player_pos).unit();
    return player_pos + direction * 1000.0;  // Advance 1m toward goal
}

// Ball prediction helpers using new API
fn predict_ball_intercept_position(s, time_ahead) {
    if s.world.ball == () {
        return vec2(0.0, 0.0);
    }
    
    // Use the new prediction API if available, otherwise simple prediction
    return s.predict_ball_position(time_ahead);
}

// Team coordination helpers using new API
fn find_closest_teammate_to_ball(s) {
    return s.closest_own_player_to_ball();
}

fn find_closest_opponent_to_ball(s) {
    let ball_pos = s.world.ball.position;
    return s.closest_opp_player_to_position(ball_pos);
}

fn count_teammates_near_ball(s, radius) {
    let ball_pos = s.world.ball.position;
    return s.get_own_players_within_radius(ball_pos, radius).len();
}

fn count_opponents_near_ball(s, radius) {
    let ball_pos = s.world.ball.position;
    return s.get_opp_players_within_radius(ball_pos, radius).len();
}

// Field zone helpers using new API
fn is_ball_in_penalty_area(s) {
    if s.world.ball == () { return false; }
    return s.is_in_penalty_area(s.world.ball.position);
}

fn is_ball_in_own_penalty_area(s) {
    if s.world.ball == () { return false; }
    return s.is_in_own_penalty_area(s.world.ball.position);
}

fn is_ball_in_opp_penalty_area(s) {
    if s.world.ball == () { return false; }
    return s.is_in_opp_penalty_area(s.world.ball.position);
}

// Strategic position calculation helpers
fn calculate_wall_position(s, wall_distance) {
    let ball_pos = s.world.ball.position;
    let goal_pos = s.get_own_goal_position();
    
    let ball_to_goal = (goal_pos - ball_pos).unit();
    return ball_pos + ball_to_goal * wall_distance;
}

fn calculate_intercept_position(s) {
    // Calculate where to position to intercept ball movement
    if s.world.ball == () || s.world.ball.velocity.norm() < 100.0 {
        return s.world.ball.position;
    }
    
    let ball_pos = s.world.ball.position;
    let ball_vel = s.world.ball.velocity;
    let player_pos = s.player.position;
    
    // Simple intercept calculation - can be enhanced
    let intercept_time = 1.0; // Assume 1 second ahead
    return ball_pos + ball_vel * intercept_time;
}

// Game state detection helpers
fn is_ball_moving_fast(s) {
    if s.world.ball == () { return false; }
    return s.world.ball.velocity.norm() > 500.0; // 0.5 m/s threshold
}

fn is_ball_coming_toward_us(s) {
    if s.world.ball == () { return false; }
    let ball_vel = s.world.ball.velocity;
    let goal_pos = s.get_own_goal_position();
    let ball_pos = s.world.ball.position;
    
    let ball_to_goal = (goal_pos - ball_pos).unit();
    let vel_direction = ball_vel.unit();
    
    // Check if ball velocity is roughly toward our goal
    return ball_to_goal.x * vel_direction.x + ball_to_goal.y * vel_direction.y > 0.7;
} 