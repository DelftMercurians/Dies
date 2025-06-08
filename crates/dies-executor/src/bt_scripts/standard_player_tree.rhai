fn has_ball_condition(s) {
    // s is the RhaiRobotSituationView (a map)
    // print("RHAI: Checking has_ball_condition: " + s.has_ball);
    return s.has_ball;
}

fn ball_is_close_condition(s) {
    if s.ball == () { // Check if ball data is nil
        // print("RHAI: Ball data not available in ball_is_close_condition");
        return false;
    }
    let dx = s.player.pos_x - s.ball.pos_x;
    let dy = s.player.pos_y - s.ball.pos_y;
    let dist_sq = dx*dx + dy*dy;
    // print("RHAI: Ball distance squared: " + dist_sq + " for player " + s.player_id);
    return dist_sq < (500.0 * 500.0);
}

fn score_offensive_position(s) {
    // Dummy scorer, higher score is better
    // print("RHAI: Scoring offensive position for player " + s.player_id);
    if s.ball != () {
        // A simple score: closer to the ball is better. Max score 100.
        let dx = s.player.pos_x - s.ball.pos_x;
        let dy = s.player.pos_y - s.ball.pos_y;
        let dist = (dx*dx + dy*dy).sqrt();
        return 100.0 - dist * 10.0; // Scale distance effect
    }
    return 0.0;
}

fn score_defensive_position(s) {
    // Dummy scorer
    // print("RHAI: Scoring defensive position for player " + s.player_id);
    // Example: prefer player's x position to be positive, max score 50 + player.pos_x
    return 50.0 + s.player.pos_x;
}

fn build_player_bt(player_id) {
    let player_id_str = player_id.to_string();

    // Define some skills (using the globally registered skill constructors)
    let go_to_origin_skill = GoToPosition(0.0, 0.0);
    let go_to_positive_skill = GoToPosition(1000.0, 1000.0);
    let go_to_negative_skill = GoToPosition(-1000.0, -1000.0);

    // Build the tree
    // let tree = Select([
    //     // Sequence 1: If player has the ball, try to go to (1,1)
    //     Sequence([
    //         Guard(has_ball_condition, go_to_positive_skill, "Cond_HasBall"),
    //         go_to_positive_skill
    //     ], "Seq_HandleBall"),

    //     // Sequence 2: If ball is close (but not possessed), go to (-1,-1) to intercept
    //     Sequence([
    //         Guard(ball_is_close_condition, go_to_negative_skill, "Cond_BallClose"),
    //     ], "Seq_GetCloseBall"),

    //     // ScoringSelect Node: Choose between two tactical positions based on scores
    //     ScoringSelect(
    //         [
    //             #{ node: go_to_positive_skill, scorer: score_offensive_position },
    //             #{ node: go_to_negative_skill, scorer: score_defensive_position }
    //         ],
    //         10.0, // Hysteresis margin (scores are roughly in 0-100 range)
    //         "Scoring_StrategicPos"
    //     ),
        
    //     // Semaphore example: Only one player can attempt to go to origin via this semaphore
    //     Semaphore(
    //         go_to_origin_skill,
    //         "unique_origin_task", // Semaphore ID string
    //         1, // Max count (integer)
    //         "Sem_GoToOrigin"
    //     ),

    //     // Fallback: Go to origin by default if nothing else runs or succeeds
    //     go_to_origin_skill
    // ], "RootSelect_Player_" + player_id_str);
    let tree = Sequence([
        go_to_origin_skill,
        go_to_positive_skill,
        go_to_negative_skill,
    ], "RootSelect_Player_" + player_id_str);

    return tree;
} 