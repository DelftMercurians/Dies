fn score_as_harasser(s) {
    let score = 40.0;
    
    // Find unmarked opponents in our half
    let unmarked_threats = find_unmarked_threats(s);
    if unmarked_threats.len() == 0 {
        return 0.0; // No need for harasser
    }
    
    // Score based on proximity to highest threat opponent
    let highest_threat = find_highest_threat_opponent(s, unmarked_threats);
    if highest_threat != () {
        let dist_to_threat = s.distance_to_position(highest_threat.position);
        score += (2000.0 - dist_to_threat.min(2000.0)) / 40.0;
        
        // Bonus if we're the closest defender
        if is_closest_defender_to(s, highest_threat.position) {
            score += 20.0;
        }
    }
    
    return score;
}

// Evaluate how threatening the ball position is
fn evaluate_ball_threat(s) {
    if s.world.ball == () { return 0.0; }
    
    let ball_pos = s.world.ball.position;
    let goal_pos = s.get_own_goal_position();
    
    // Distance factor
    let ball_goal_dist = ball_pos.distance_to(goal_pos);
    let dist_threat = (4000.0 - ball_goal_dist.min(4000.0)) / 4000.0;
    
    // Central position factor
    let central_threat = 1.0 - (ball_pos.y.abs() / 3000.0).min(1.0);
    
    // Ball in our half factor
    let half_factor = if s.is_in_defensive_half(ball_pos) { 1.0 } else { 0.3 };
    
    return dist_threat * central_threat * half_factor;
}

// Find unmarked opponent threats
fn find_unmarked_threats(s) {
    let opponents_in_our_half = s.filter_opp_players_by(|p| {
        s.is_in_defensive_half(p.position)
    });
    
    let unmarked = [];
    
    for opponent in opponents_in_our_half {
        if !is_opponent_marked(s, opponent) {
            unmarked.push(opponent);
        }
    }
    
    return unmarked;
}

// Check if opponent is already marked by a teammate
fn is_opponent_marked(s, opponent) {
    let marking_distance = 800.0;
    
    let nearby_teammates = s.get_own_players_within_radius(opponent.position, marking_distance);
    return nearby_teammates.len() > 0;
}

// Find highest threat opponent from list
fn find_highest_threat_opponent(s, opponents) {
    let best_opponent = ();
    let best_threat = -1.0;
    
    for opponent in opponents {
        let threat = evaluate_opponent_threat(s, opponent);
        if threat > best_threat {
            best_threat = threat;
            best_opponent = opponent;
        }
    }
    
    return best_opponent;
}

// Evaluate threat level of an opponent
fn evaluate_opponent_threat(s, opponent) {
    // Distance to our goal
    let goal_dist = opponent.position.distance_to(s.get_own_goal_position());
    let dist_threat = (3000.0 - goal_dist.min(3000.0)) / 3000.0;
    
    // Distance to ball
    let ball_dist = opponent.position.distance_to(s.world.ball.position);
    let ball_threat = (1500.0 - ball_dist.min(1500.0)) / 1500.0;
    
    // Central position is more threatening
    let central_threat = 1.0 - (opponent.position.y.abs() / 3000.0).min(1.0);
    
    return dist_threat * 0.5 + ball_threat * 0.3 + central_threat * 0.2;
}

// Check if we're the closest defender to a position
fn is_closest_defender_to(s, target_pos) {
    let my_dist = s.distance_to_position(target_pos);
    
    let teammates = s.filter_own_players_by(|p| {
        p.id != s.player_id && p.id != 0 // Exclude self and goalkeeper
    });
    
    for teammate in teammates {
        if teammate.position.distance_to(target_pos) < my_dist {
            return false;
        }
    }
    
    return true;
}
