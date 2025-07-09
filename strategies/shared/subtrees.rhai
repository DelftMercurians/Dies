import "shared/utilities" as util;
import "shared/situations" as sit;

// Goalkeeper behavior subtree
fn build_goalkeeper_behavior() {
    return Sequence([
        GoToPosition(
            vec2(-5500.0, 0.0),
            #{ heading: 0.0 },
            "Guard Goal"
        )
    ], "Goalkeeper");
}

// Attacker behavior subtree
fn build_attacker_behavior() {
    return Semaphore(
        Sequence([
            FetchBall("Get Ball"),
            Guard(|s| sit::i_have_ball(s),
                Select([
                    // Try to score if close to goal
                    Guard(|s| util::close_to_goal(s),
                        Sequence([
                            FaceTowardsPosition(util::get_opponent_goal(), #{}, "Aim at Goal"),
                            Kick("Shoot!")
                        ], "Score Goal"),
                        "Close to goal?"
                    ),
                    // Otherwise, advance with ball
                    GoToPosition(
                        |s| util::advance_toward_goal(s),
                        #{ heading: |s| util::get_heading_to_goal(s), with_ball: true },
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

// Supporter behavior subtree
fn build_supporter_behavior() {
    return Semaphore(
        GoToPosition(
            |s| util::get_supporter_pos(s),
            #{ heading: |s| util::get_heading_to_goal(s) },
            "Support Position"
        ),
        "supporter_role",
        1,
        "Supporter Role"
    );
}

// Defender behavior subtree
fn build_defender_behavior() {
    return GoToPosition(
        |s| util::get_defender_pos(s),
        #{ heading: 0.0 },
        "Defend Position"
    );
}

// Kickoff kicker behavior subtree
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
        Guard(|s| sit::i_have_ball(s),
            Sequence([
                FaceTowardsPosition(util::get_opponent_goal(), #{}, "Face Goal"),
                Kick("Kickoff!")
            ], "Execute Kickoff"),
            "Have ball?"
        )
    ], "Kickoff Kicker");
}

// Penalty goalkeeper behavior subtree
fn build_penalty_goalkeeper_behavior() {
    // Rule: Keeper must stay on goal line between posts
    return GoToPosition(
        vec2(-6000.0, 0.0),
        #{ heading: 0.0 },
        "Guard Goal Line"
    );
}

// Penalty taker behavior subtree
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
        Guard(|s| sit::i_have_ball(s),
            Sequence([
                FaceTowardsPosition(util::get_opponent_goal(), #{}, "Aim at Goal"),
                Wait(0.5, "Aim"),
                Kick("Penalty Shot!")
            ], "Execute Penalty"),
            "Have ball?"
        )
    ], "Penalty Taker");
} 