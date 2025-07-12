import "striker_scorer" as scorer;

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
                |s| find_optimal_striker_position(s, zone),
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
            |s| find_optimal_shot_target(s),
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
            |s| find_best_pass_target(s),
            #{},
            "Aim pass"
        ),
        Kick("Pass!")
    ], "Execute Pass");
}

// Dribbling sequence
fn build_dribble_sequence(zone) {
    return GoToPosition(
        |s| calculate_striker_advance_position(s),
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
    let zone_bounds = get_zone_bounds(zone, s.get_field_bounds());
    let zone_center = vec2(
        (zone_bounds.min_x + zone_bounds.max_x) / 2.0,
        (zone_bounds.min_y + zone_bounds.max_y) / 2.0
    );
    
    let opponents_in_zone = s.get_opp_players_within_radius(zone_center, 1500.0);
    let congestion_penalty = opponents_in_zone.len() * 10.0;
    
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

// Find optimal shot target on goal line
fn find_optimal_shot_target(s) {
    let goal_corners = s.get_opp_goal_corners();
    if goal_corners.len() < 2 {
        return s.get_opp_goal_position();
    }
    
    let top_corner = goal_corners[0];
    let bottom_corner = goal_corners[1];
    let goal_center = s.get_opp_goal_position();
    
    // Find opponent goalkeeper position
    let keeper_pos = find_opponent_keeper_position(s);
    
    // Find defenders near goal
    let goal_defenders = s.filter_opp_players_by(|p| {
        let dist_to_goal = p.position.distance_to(goal_center);
        return dist_to_goal < 2000.0;
    });
    
    // Divide goal line into segments and evaluate each
    let segments = 10;
    let best_score = -1.0;
    let best_target = goal_center;
    
    for i in 0..=segments {
        let t = i / segments;
        let target = bottom_corner.interpolate(top_corner, t);
        
        // Calculate shadow score (lower = less blocked)
        let shadow_score = 0.0;
        
        // Add keeper shadow
        if keeper_pos != () {
            let keeper_dist = target.distance_to(keeper_pos);
            shadow_score += 200.0 / (keeper_dist + 100.0);
        }
        
        // Add defender shadows
        for defender in goal_defenders {
            let defender_dist = target.distance_to(defender.position);
            shadow_score += 100.0 / (defender_dist + 100.0);
        }
        
        let segment_score = 1.0 / (shadow_score + 0.1);
        
        if segment_score > best_score {
            best_score = segment_score;
            best_target = target;
        }
    }
    
    return best_target;
}

// Find opponent keeper position
fn find_opponent_keeper_position(s) {
    let goal_center = s.get_opp_goal_position();
    
    // Look for opponent closest to their goal
    let keeper = s.find_opp_player_min_by(|p| {
        p.position.distance_to(goal_center)
    });
    
    if keeper != () && keeper.position.distance_to(goal_center) < 1000.0 {
        return keeper.position;
    }
    
    return ();
}

// Find optimal striker position within allocated zone
fn find_optimal_striker_position(s, zone) {
    let field_bounds = s.get_field_bounds();
    let goal_pos = s.get_opp_goal_position();
    
    // Define zone boundaries
    let zone_bounds = get_zone_bounds(zone, field_bounds);
    
    // Sample grid of points in zone
    let grid_size = 5;
    let best_pos = vec2(0.0, 0.0);
    let best_score = -1.0;
    
    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = zone_bounds.min_x + (zone_bounds.max_x - zone_bounds.min_x) * (i / (grid_size - 1));
            let y = zone_bounds.min_y + (zone_bounds.max_y - zone_bounds.min_y) * (j / (grid_size - 1));
            let pos = vec2(x, y);
            
            // Skip if too close to defense area
            if s.distance_to_opp_penalty_area() < 200.0 {
                continue;
            }
            
            // Calculate position score
            let score = evaluate_striker_position(s, pos);
            
            if score > best_score {
                best_score = score;
                best_pos = pos;
            }
        }
    }
    
    return best_pos;
}

// Helper to define zone boundaries
fn get_zone_bounds(zone, field_bounds) {
    let third_height = (field_bounds.max_y - field_bounds.min_y) / 3.0;
    
    if zone == "top" {
        return #{
            min_x: 0.0,
            max_x: field_bounds.max_x,
            min_y: field_bounds.min_y + 2.0 * third_height,
            max_y: field_bounds.max_y
        };
    } else if zone == "bottom" {
        return #{
            min_x: 0.0,
            max_x: field_bounds.max_x,
            min_y: field_bounds.min_y,
            max_y: field_bounds.min_y + third_height
        };
    } else {
        return #{
            min_x: 0.0,
            max_x: field_bounds.max_x,
            min_y: field_bounds.min_y + third_height,
            max_y: field_bounds.min_y + 2.0 * third_height
        };
    }
}

// Evaluate quality of a striker position
fn evaluate_striker_position(s, pos) {
    let score = 0.0;
    
    // Pass reception score
    if s.world.ball != () {
        let ball_pos = s.world.ball.position;
        let pass_quality = evaluate_pass_reception(s, pos, ball_pos);
        score += pass_quality * 50.0;
    }
    
    // Shot potential score
    let goal_pos = s.get_opp_goal_position();
    let shot_target = find_optimal_shot_target(s);
    let shot_quality = scorer::evaluate_shot_quality(s, pos, shot_target);
    score += shot_quality * 100.0;
    
    // Distance to goal (closer is better)
    let goal_dist = pos.distance_to(goal_pos);
    score += (4000.0 - goal_dist.min(4000.0)) / 40.0;
    
    // Space around position (more space = better)
    let nearby_opponents = s.get_opp_players_within_radius(pos, 1000.0);
    score += (3 - nearby_opponents.len().min(3)) * 20.0;
    
    return score;
}

// Evaluate pass reception quality at a position
fn evaluate_pass_reception(s, receiver_pos, passer_pos) {
    // Check if pass path is clear
    let ray_result = s.cast_ray(passer_pos, receiver_pos);
    if ray_result.hit {
        return 0.0;
    }
    
    // Check space around receiver
    let nearby_opponents = s.get_opp_players_within_radius(receiver_pos, 500.0);
    let space_score = 1.0 - (nearby_opponents.len() / 3.0).min(1.0);
    
    return space_score;
}

// Calculate advance position for striker
fn calculate_striker_advance_position(s) {
    let player_pos = s.player.position;
    let goal_pos = s.get_opp_goal_position();
    
    // Find optimal shot target
    let shot_target = find_optimal_shot_target(s);
    
    // Advance toward shot target, not just goal center
    let advance_direction = (shot_target - player_pos).unit();
    let advance_distance = 1000.0;
    
    let target_pos = player_pos + advance_direction * advance_distance;
    
    // Ensure we stay in field
    return constrain_to_field(s, target_pos);
}

// Constrain position to field boundaries
fn constrain_to_field(s, pos) {
    let bounds = s.get_field_bounds();
    
    let x = pos.x.max(bounds.min_x + 200.0).min(bounds.max_x - 200.0);
    let y = pos.y.max(bounds.min_y + 200.0).min(bounds.max_y - 200.0);
    
    return vec2(x, y);
}

// Find best pass target
fn find_best_pass_target(s) {
    let teammates = s.filter_own_players_by(|p| p.id != s.player_id);
    let ball_pos = s.world.ball.position;
    
    let best_target = s.get_opp_goal_position();
    let best_score = -1.0;
    
    for teammate in teammates {
        // Check if pass is possible
        let ray_result = s.cast_ray(ball_pos, teammate.position);
        if !ray_result.hit {
            // Evaluate teammate position
            let goal_dist = teammate.position.distance_to(s.get_opp_goal_position());
            let score = 10000.0 - goal_dist;
            
            // Bonus if teammate in good shooting position
            let shot_quality = scorer::evaluate_shot_quality(s, teammate.position, s.get_opp_goal_position());
            score += shot_quality * 2000.0;
            
            if score > best_score {
                best_score = score;
                best_target = teammate.position;
            }
        }
    }
    
    return best_target;
} 