// Callbacks for dynamic arguments
fn get_ball_pos(s) {
    // s is RobotSituation
    // Note: The ball can be None (represented as '()' in Rhai).
    if s.world.ball == () {
        throw "Ball information not available in get_ball_pos.";
    }
    return s.world.ball.position;
}

fn get_heading_to_ball(s) {
    if s.world.ball == () {
        throw "Ball information not available in get_heading_to_ball.";
    }
    let player_pos = s.player().position;
    let ball_pos = s.world.ball.position;
    let dx = ball_pos.x - player_pos.x;
    let dy = ball_pos.y - player_pos.y;
    return dy.atan(dx);
}

fn get_supporter_pos(s) {
    if s.world.ball == () {
        throw "Ball information not available in get_supporter_pos.";
    }
    let ball_pos = s.world.ball.position;
    // Position self 2 meters behind the ball, relative to the center line.
    return vec2(ball_pos.x - 2000.0, ball_pos.y);
}

fn i_have_ball(s) {
    return s.has_ball();
}

fn build_player_bt(player_id) {
    let opponent_goal = vec2(6000.0, 0.0);
    let own_goal = vec2(-6000.0, 0.0);

    // Attacker behavior: get the ball and shoot
    let attacker_behavior = Sequence([
        FetchBall("Get Ball"),
        Guard(i_have_ball, 
            FaceTowardsPosition(opponent_goal, #{}, "Face Opponent Goal"),
            "Have ball?"
        ),
        Kick("Shoot!")
    ], "Attacker Sequence");

    // Supporter behavior: stay behind the ball, ready to receive a pass or take over.
    // Uses a dynamic target for GoToPosition and a dynamic heading.
    let supporter_behavior = GoToPosition(
        get_supporter_pos, 
        #{ heading: get_heading_to_ball },
        "Support Attacker"
    );

    // Defender behavior: stay near own goal. Position is varied by player ID to avoid clumping.
    let defender_y = (hash_float(player_id) - 0.5) * 4000.0; // spread between -2000 and 2000
    let defender_pos = vec2(own_goal.x + 1500.0, defender_y);
    let defender_behavior = GoToPosition(
        defender_pos,
        #{ heading: 0.0 }, // Face center
        "Defend"
    );

    // Main tree with role assignment using Semaphores
    let tree = Select([
        Semaphore(
            attacker_behavior,
            "attacker_role",
            1, // Only one attacker
            "Try Attacker Role"
        ),
        Semaphore(
            supporter_behavior,
            "supporter_role",
            1, // Only one supporter
            "Try Supporter Role"
        ),
        // Everyone else becomes a defender
        defender_behavior
    ], "Player Roles");

    return tree;
} 