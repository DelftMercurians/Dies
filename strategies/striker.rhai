import "striker/scorer" as scorer;
import "striker/positions" as pos;

fn build_striker_tree() {
    return Select([
        // Handle kickoff positioning - stay on our side
        Guard(|s| s.world.game_state.game_state == "Kickoff",
            build_kickoff_positioning_behavior(),
            "Kickoff positioning"
        ),
        
        // Normal striker behavior with zone allocation
        build_zone_based_striker_behavior()
    ], "Striker");
}

// Kickoff positioning for non-kicker strikers
fn build_kickoff_positioning_behavior() {
    return GoToPosition(
        |s| get_kickoff_striker_position(s),
        #{ heading: |s| get_heading_toward_ball(s) },
        "Kickoff positioning"
    );
}

// Zone-based striker behavior using ScoringSelect
fn build_zone_based_striker_behavior() {
    return ScoringSelect([
        #{
            node: Semaphore(
                build_striker_in_zone("top"),
                "striker_zone_top", 
                1,
                "Top Zone"
            ),
            scorer: |s| score_for_zone(s, "top")
        },
        #{
            node: Semaphore(
                build_striker_in_zone("middle"),
                "striker_zone_middle",
                1, 
                "Middle Zone"
            ),
            scorer: |s| score_for_zone(s, "middle")
        },
        #{
            node: Semaphore(
                build_striker_in_zone("bottom"),
                "striker_zone_bottom",
                1,
                "Bottom Zone"
            ),
            scorer: |s| score_for_zone(s, "bottom")
        }
    ], 0.1, "Choose Attack Zone");
}

// Striker behavior within a specific zone
fn build_striker_in_zone(zone) {
    return ScoringSelect([
        #{
            node: Guard(|s| s.has_ball(),
                build_ball_carrier_behavior(zone),
                "Have ball"
            ),
            scorer: |s| if s.has_ball() { 100.0 } else { 0.0 }
        },
        #{
            node: FetchBall("Get ball"),
            scorer: |s| {
                let ball_dist = s.distance_to_ball();
                if ball_dist < 1000.0 { 80.0 - ball_dist / 20.0 } else { 0.0 }
            }
        },
        #{
            node: GoToPosition(
                |s| pos::find_optimal_striker_position(s, zone),
                #{ heading: |s| s.world.ball.position.angle_to(s.player.position) },
                "Position in " + zone
            ),
            scorer: |s| 30.0
        }
    ], 0.2, "Zone " + zone + " Actions");
}

// Ball carrier decision making
fn build_ball_carrier_behavior(zone) {
    return ScoringSelect([
        #{
            node: build_shoot_sequence(),
            scorer: |s| scorer::score_for_shooting(s)
        },
        #{
            node: build_pass_sequence(),
            scorer: |s| scorer::score_for_passing(s)
        },
        #{
            node: build_dribble_sequence(zone),
            scorer: |s| 20.0 // Base dribble score
        }
    ], 5.0, "Ball Carrier Decision");
}

// Shooting sequence
fn build_shoot_sequence() {
    return Sequence([
        FaceTowardsPosition(
            |s| pos::find_optimal_shot_target(s),
            #{},
            "Aim at goal"
        ),
        Kick("Shoot!")
    ], "Execute Shot");
}

// Passing sequence
fn build_pass_sequence() {
    return Sequence([
        FaceTowardsPosition(
            |s| pos::find_best_pass_target(s),
            #{},
            "Aim pass"
        ),
        Kick("Pass!")
    ], "Execute Pass");
}

// Dribbling sequence
fn build_dribble_sequence(zone) {
    return GoToPosition(
        |s| pos::calculate_striker_advance_position(s),
        #{ 
            heading: |s| (s.get_opp_goal_position() - s.player.position).angle_to(vec2(1.0, 0.0)),
            with_ball: true
        },
        "Dribble in " + zone
    );
}

// Helper functions
fn score_for_zone(s, zone) {
    // Base score with some randomization for diversity
    let base_score = 50.0 + s.player_id.hash_float() * 20.0;
    
    // Prefer zones with fewer opponents
    let zone_bounds = pos::get_zone_bounds(zone, s.get_field_bounds());
    let zone_center = vec2(
        (zone_bounds.min_x + zone_bounds.max_x) / 2.0,
        (zone_bounds.min_y + zone_bounds.max_y) / 2.0
    );
    
    let opponents_in_zone = s.get_opp_players_within_radius(zone_center, 1500.0);
    let congestion_penalty = opponents_in_zone.len() as f64 * 10.0;
    
    return base_score - congestion_penalty;
}

fn get_kickoff_striker_position(s) {
    // Position in our half, outside center circle, ready to advance
    let player_hash = s.player_id.hash_float();
    
    // Spread strikers across our half
    let spread_x = -2000.0 + player_hash * 1500.0; // Between -2000 and -500
    let spread_y = (player_hash - 0.5) * 4000.0;    // Between -2000 and 2000
    
    // Ensure outside center circle (500mm radius + margin)
    let pos = vec2(spread_x, spread_y);
    if s.is_in_center_circle(pos) {
        // Move further back if in center circle
        return vec2(spread_x - 1000.0, spread_y);
    }
    
    return pos;
}

fn get_heading_toward_ball(s) {
    if s.world.ball == () { return 0.0; }
    let ball_pos = s.world.ball.position;
    let my_pos = s.player.position;
    return (ball_pos - my_pos).angle_to(vec2(1.0, 0.0));
} 