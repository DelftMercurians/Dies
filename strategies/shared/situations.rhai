// Player has ball check
fn i_have_ball(s) {
    return s.has_ball();
}

// Ball position checks
fn ball_in_our_half(s) {
    if s.world.ball == () { return false; }
    return s.world.ball.position.x < 0.0;
}

fn ball_in_opponent_half(s) {
    if s.world.ball == () { return false; }
    return s.world.ball.position.x > 0.0;
}

// Player role checks
fn is_goalkeeper(s) {
    return s.player_id.to_string() == "0";
}

// Game state checks
fn we_are_attacking(s) {
    return s.world.game_state.us_operating;
}

// Distance-based conditions
fn close_to_ball(s) {
    if s.world.ball == () { return false; }
    let player_pos = s.player().position;
    let ball_pos = s.world.ball.position;
    let dx = ball_pos.x - player_pos.x;
    let dy = ball_pos.y - player_pos.y;
    let dist_sq = dx*dx + dy*dy;
    return dist_sq < 500.0 * 500.0; // Within 500mm
}

fn very_close_to_ball(s) {
    if s.world.ball == () { return false; }
    let player_pos = s.player().position;
    let ball_pos = s.world.ball.position;
    let dx = ball_pos.x - player_pos.x;
    let dy = ball_pos.y - player_pos.y;
    let dist_sq = dx*dx + dy*dy;
    return dist_sq < 200.0 * 200.0; // Within 200mm
}

fn far_from_ball(s) {
    if s.world.ball == () { return true; }
    let player_pos = s.player().position;
    let ball_pos = s.world.ball.position;
    let dx = ball_pos.x - player_pos.x;
    let dy = ball_pos.y - player_pos.y;
    let dist_sq = dx*dx + dy*dy;
    return dist_sq > 2000.0 * 2000.0; // Further than 2m
} 