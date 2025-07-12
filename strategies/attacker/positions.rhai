// Attacker-specific position calculations
import "attacker/scorer" as scorer;

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
        let t = i as f64 / segments as f64;
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
            let x = zone_bounds.min_x + (zone_bounds.max_x - zone_bounds.min_x) * (i as f64 / (grid_size - 1) as f64);
            let y = zone_bounds.min_y + (zone_bounds.max_y - zone_bounds.min_y) * (j as f64 / (grid_size - 1) as f64);
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
    score += (3 - nearby_opponents.len().min(3)) as f64 * 20.0;
    
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
    let space_score = 1.0 - (nearby_opponents.len() as f64 / 3.0).min(1.0);
    
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