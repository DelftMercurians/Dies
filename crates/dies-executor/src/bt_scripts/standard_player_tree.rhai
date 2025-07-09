// ============================================================================
// DIES BEHAVIOR TREE SCRIPT - COMPLETE GAME STATE HANDLING
// ============================================================================
// This script provides complete behavior trees for all major game states:
// - Play Tree: Normal gameplay including free kicks
// - Kickoff Tree: Kickoff situations (both offensive and defensive)
// - Penalty Tree: Penalty situations (both attacking and defending)
//
// All strategies are designed to be rule-compliant with RoboCup SSL 2025 rules.
// ============================================================================

// ============================================================================
// SHARED HELPER FUNCTIONS
// ============================================================================

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

// Goal positions
fn get_opponent_goal() { return vec2(6000.0, 0.0); }
fn get_own_goal() { return vec2(-6000.0, 0.0); }

// Helper to check if we're on the attacking team for this game state
fn we_are_attacking(s) {
    return s.world.game_state.us_operating;
}

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
    if ball_in_opponent_half(s) {
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
    if ball_in_our_half(s) {
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

// ============================================================================
// PLAY TREE - Normal gameplay including free kicks
// ============================================================================

fn build_play_bt(player_id) {
    // Goalkeeper behavior (player 0)
    if player_id.to_string() == "0" {
        return build_goalkeeper_behavior();
    }
    
    // Field players use dynamic role assignment
    return ScoringSelect([
        #{ 
            node: build_attacker_behavior(), 
            scorer: score_as_attacker 
        },
        #{ 
            node: build_supporter_behavior(), 
            scorer: score_as_supporter 
        },
        #{ 
            node: build_defender_behavior(), 
            scorer: score_as_defender 
        }
    ], 0.15, "Dynamic Role Assignment");
}

fn build_goalkeeper_behavior() {
    return Sequence([
        GoToPosition(
            vec2(-5500.0, 0.0),
            #{ heading: 0.0 },
            "Guard Goal"
        )
    ], "Goalkeeper");
}

fn build_attacker_behavior() {
    return Semaphore(
        Sequence([
            FetchBall("Get Ball"),
            Guard(i_have_ball,
                Select([
                    // Try to score if close to goal
                    Guard(close_to_goal,
                        Sequence([
                            FaceTowardsPosition(get_opponent_goal(), #{}, "Aim at Goal"),
                            Kick("Shoot!")
                        ], "Score Goal"),
                        "Close to goal?"
                    ),
                    // Otherwise, advance with ball
                    GoToPosition(
                        advance_toward_goal,
                        #{ heading: get_heading_to_goal, with_ball: true },
                        "Advance with Ball"
                    )
                ], "Ball Actions"),
                "Have ball?"
            )
        ], "Attacker Sequence"),
        "attacker_role",
        1,
        "Attacker Role"
    );
}

fn build_supporter_behavior() {
    return Semaphore(
        GoToPosition(
            get_supporter_pos,
            #{ heading: get_heading_to_goal },
            "Support Position"
        ),
        "supporter_role",
        1,
        "Supporter Role"
    );
}

fn build_defender_behavior() {
    return GoToPosition(
        get_defender_pos,
        #{ heading: 0.0 },
        "Defend Position"
    );
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

// ============================================================================
// KICKOFF TREE - Rule-compliant kickoff behavior
// ============================================================================

fn build_kickoff_bt(player_id) {
    return Select([
        // Try to be the kicker (only one allowed in center circle)
        Semaphore(
            build_kickoff_kicker_behavior(),
            "kickoff_kicker",
            1,
            "Kickoff Kicker"
        ),
        // Everyone else positions in own half, outside center circle
        build_kickoff_supporter_behavior(player_id)
    ], "Kickoff Roles");
}

fn build_kickoff_kicker_behavior() {
    return Sequence([
        // Move to center circle
        GoToPosition(
            vec2(0.0, 0.0),
            #{ heading: 0.0 },
            "Move to Center"
        ),
        // Wait for ball to be placed and game to start
        Wait(0.5, "Wait for Start"),
        // Kick the ball forward
        FetchBall("Get Ball"),
        Guard(i_have_ball,
            Sequence([
                FaceTowardsPosition(get_opponent_goal(), #{}, "Face Goal"),
                Kick("Kickoff!")
            ], "Execute Kickoff"),
            "Have ball?"
        )
    ], "Kickoff Kicker");
}

fn build_kickoff_supporter_behavior(player_id) {
    // Position in own half, spread out, ready to support
    let player_hash = hash_float(player_id);
    let spread_x = -3000.0 + player_hash * 2000.0; // Between -3000 and -1000
    let spread_y = (player_hash - 0.5) * 4000.0;    // Between -2000 and 2000
    
    return GoToPosition(
        vec2(spread_x, spread_y),
        #{ heading: 0.0 },
        "Kickoff Support Position"
    );
}

// ============================================================================
// PENALTY TREE - Rule-compliant penalty behavior
// ============================================================================

fn build_penalty_bt(player_id) {
    return Select([
        // Goalkeeper behavior (always player 0)
        Guard(is_goalkeeper,
            build_penalty_goalkeeper_behavior(),
            "Is goalkeeper?"
        ),
        
        // Try to be penalty taker (only one allowed)
        Semaphore(
            build_penalty_taker_behavior(),
            "penalty_taker",
            1,
            "Penalty Taker"
        ),
        
        // Everyone else stays back
        build_penalty_support_behavior(player_id)
    ], "Penalty Roles");
}

fn is_goalkeeper(s) {
    return s.player_id.to_string() == "0";
}

fn build_penalty_taker_behavior() {
    return Sequence([
        // Move to penalty mark area
        GoToPosition(
            vec2(2000.0, 0.0), // 2m from penalty mark
            #{ heading: 0.0 },
            "Approach Penalty Mark"
        ),
        
        // Wait for setup
        Wait(1.0, "Wait for Setup"),
        
        // Execute penalty
        FetchBall("Get Ball"),
        Guard(i_have_ball,
            Sequence([
                FaceTowardsPosition(get_opponent_goal(), #{}, "Aim at Goal"),
                Wait(0.5, "Aim"),
                Kick("Penalty Shot!")
            ], "Execute Penalty"),
            "Have ball?"
        )
    ], "Penalty Taker");
}

fn build_penalty_goalkeeper_behavior() {
    // Rule: Keeper must stay on goal line between posts
    return GoToPosition(
        vec2(-6000.0, 0.0),
        #{ heading: 0.0 },
        "Guard Goal Line"
    );
}

fn build_penalty_support_behavior(player_id) {
    // Rule: All other players must be 1m behind ball
    let player_hash = hash_float(player_id);
    let spread_y = (player_hash - 0.5) * 6000.0; // Spread across field width
    
    return GoToPosition(
        vec2(1000.0, spread_y), // 1m behind penalty mark
        #{ heading: 0.0 },
        "Penalty Support Position"
    );
}

// ============================================================================
// ENTRY POINTS - These are called by the engine
// ============================================================================

// Legacy entry point for backward compatibility
fn build_player_bt(player_id) {
    return build_play_bt(player_id);
} 