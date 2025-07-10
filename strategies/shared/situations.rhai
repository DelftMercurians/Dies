// Updated situation detection functions using modern RobotSituation API

// Player has ball check - using new API
fn i_have_ball(s) {
    return s.has_ball();
}

// Ball position checks - using new API
fn ball_in_our_half(s) {
    if s.world.ball == () { return false; }
    return s.is_in_defensive_half(s.world.ball.position);
}

fn ball_in_opponent_half(s) {
    if s.world.ball == () { return false; }
    return s.is_in_attacking_half(s.world.ball.position);
}

// Player role checks
fn is_goalkeeper(s) {
    return s.player_id == 0;
}

// Game state checks
fn we_are_attacking(s) {
    return s.world.game_state.us_operating;
}

// Game mode detection functions
fn is_kickoff_mode(s) {
    // Detect kickoff by ball being near center
    if s.world.ball == () { return false; }
    return s.is_in_center_circle(s.world.ball.position);
}

fn is_penalty_mode(s) {
    // Detect penalty by ball being near penalty mark
    if s.world.ball == () { return false; }
    let ball_pos = s.world.ball.position;
    let own_penalty_dist = s.distance_to_position(s.get_own_penalty_mark());
    let opp_penalty_dist = s.distance_to_position(s.get_opp_penalty_mark());
    return own_penalty_dist < 500.0 || opp_penalty_dist < 500.0;
}

fn is_normal_play_mode(s) {
    return !is_kickoff_mode(s) && !is_penalty_mode(s);
}

// Distance-based conditions - using new API
fn close_to_ball(s) {
    return s.distance_to_ball() < 500.0; // Within 500mm
}

fn very_close_to_ball(s) {
    return s.distance_to_ball() < 200.0; // Within 200mm
}

fn far_from_ball(s) {
    return s.distance_to_ball() > 2000.0; // Further than 2m
}

// Enhanced distance conditions using new API
fn close_to_own_goal(s) {
    return s.distance_to_position(s.get_own_goal_position()) < 1000.0;
}

fn close_to_opp_goal(s) {
    return s.distance_to_position(s.get_opp_goal_position()) < 2000.0;
}

fn in_center_circle(s) {
    return s.is_in_center_circle(s.player.position);
}

// Ball zone conditions using new API
fn ball_in_penalty_area(s) {
    if s.world.ball == () { return false; }
    return s.is_in_penalty_area(s.world.ball.position);
}

fn ball_in_own_penalty_area(s) {
    if s.world.ball == () { return false; }
    return s.is_in_own_penalty_area(s.world.ball.position);
}

fn ball_in_opp_penalty_area(s) {
    if s.world.ball == () { return false; }
    return s.is_in_opp_penalty_area(s.world.ball.position);
}

fn ball_in_center_circle(s) {
    if s.world.ball == () { return false; }
    return s.is_in_center_circle(s.world.ball.position);
}

// Enhanced team coordination conditions using new API
fn am_closest_to_ball(s) {
    let closest = s.closest_own_player_to_ball();
    return closest != () && closest.id == s.player_id;
}

fn am_furthest_from_ball(s) {
    let my_dist = s.distance_to_ball();
    let teammates = s.world.own_players;
    
    for teammate in teammates {
        if teammate.id != s.player_id {
            let teammate_dist = s.distance_to_position(teammate.position);
            if teammate_dist > my_dist {
                return false;
            }
        }
    }
    return true;
}

fn opponents_near_ball(s) {
    let ball_pos = s.world.ball.position;
    let nearby_opponents = s.get_opp_players_within_radius(ball_pos, 1000.0);
    return nearby_opponents.len() > 0;
}

fn teammates_near_ball(s) {
    let ball_pos = s.world.ball.position;
    let nearby_teammates = s.get_own_players_within_radius(ball_pos, 1000.0);
    return nearby_teammates.len() > 1; // More than just me
}

// Ball movement conditions using new API
fn ball_moving_fast(s) {
    if s.world.ball == () { return false; }
    return s.world.ball.velocity.norm() > 500.0; // 0.5 m/s threshold
}

fn ball_moving_toward_our_goal(s) {
    if s.world.ball == () { return false; }
    let ball_vel = s.world.ball.velocity;
    let goal_pos = s.get_own_goal_position();
    let ball_pos = s.world.ball.position;
    
    let ball_to_goal = (goal_pos - ball_pos).unit();
    let vel_direction = ball_vel.unit();
    
    // Check if ball velocity is roughly toward our goal
    return ball_to_goal.x * vel_direction.x + ball_to_goal.y * vel_direction.y > 0.7;
}

fn ball_moving_toward_opp_goal(s) {
    if s.world.ball == () { return false; }
    let ball_vel = s.world.ball.velocity;
    let goal_pos = s.get_opp_goal_position();
    let ball_pos = s.world.ball.position;
    
    let ball_to_goal = (goal_pos - ball_pos).unit();
    let vel_direction = ball_vel.unit();
    
    // Check if ball velocity is roughly toward opponent goal
    return ball_to_goal.x * vel_direction.x + ball_to_goal.y * vel_direction.y > 0.7;
}

// Enhanced strategic conditions
fn clear_shot_to_goal(s) {
    if s.world.ball == () { return false; }
    let ball_pos = s.world.ball.position;
    let goal_pos = s.get_opp_goal_position();
    
    // Use ray casting to check for clear path
    let ray_result = s.cast_ray(ball_pos, goal_pos);
    return ray_result.hit == false; // No obstacles
}

fn under_pressure(s) {
    let player_pos = s.player.position;
    let nearby_opponents = s.get_opp_players_within_radius(player_pos, 800.0);
    return nearby_opponents.len() > 0;
}

fn have_support(s) {
    let player_pos = s.player.position;
    let nearby_teammates = s.get_own_players_within_radius(player_pos, 1500.0);
    return nearby_teammates.len() >= 2; // At least one other teammate nearby
}

fn can_pass_to_teammate(s) {
    if !s.has_ball() { return false; }
    
    let ball_pos = s.world.ball.position;
    let teammates = s.world.own_players;
    
    for teammate in teammates {
        if teammate.id != s.player_id {
            let ray_result = s.cast_ray(ball_pos, teammate.position);
            if ray_result.hit == false {
                return true; // Found clear pass
            }
        }
    }
    return false;
}

// Positional advantage conditions
fn in_good_attacking_position(s) {
    let player_pos = s.player.position;
    return s.is_in_attacking_half(player_pos) && 
           s.distance_to_position(s.get_opp_goal_position()) < 4000.0;
}

fn in_good_defensive_position(s) {
    let player_pos = s.player.position;
    return s.is_in_defensive_half(player_pos) && 
           s.distance_to_position(s.get_own_goal_position()) < 3000.0;
}

fn should_retreat(s) {
    return ball_in_our_half(s) && 
           opponents_near_ball(s) && 
           !am_closest_to_ball(s);
}

fn should_advance(s) {
    return ball_in_opponent_half(s) && 
           (am_closest_to_ball(s) || s.has_ball()) &&
           !under_pressure(s);
} 