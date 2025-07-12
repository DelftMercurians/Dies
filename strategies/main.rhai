import "keeper" as keeper;
import "striker" as striker;
import "defender/scorer" as defender_scorer;
import "striker_scorer" as striker_scorer;

fn main(game) {
    // Always need exactly one goalkeeper
    game.add_role("goalkeeper")
        .count(1)
        .score(|s| 100.0)
        .require(|s| s.player_id == 0)
        .behavior(|| keeper::build_goalkeeper_tree())
        .build();

    // Specialized roles based on game state
    switch game.game_state {
        "Kickoff" => {
            if game.us_operating {
                // Kickoff kicker - one robot to take the kickoff
                game.add_role("kickoff_kicker")
                    .count(1)
                    .score(|s| score_kickoff_kicker(s))
                    .exclude(|s| s.player_id == 0)
                    .behavior(|| build_kickoff_kicker_tree())
                    .build();
            }
        },
        "FreeKick" => {
            if game.us_operating {
                // Free kick kicker
                game.add_role("free_kick_kicker")
                    .count(1)
                    .score(|s| score_free_kick_kicker(s))
                    .exclude(|s| s.player_id == 0)
                    .behavior(|| build_free_kick_kicker_tree())
                    .build();
            } else {
                // Free kick interference - defend against opponent free kick
                game.add_role("free_kick_interference")
                    .min(1).max(2)
                    .score(|s| score_free_kick_interference(s))
                    .exclude(|s| s.player_id == 0)
                    .behavior(|| build_free_kick_interference_tree())
                    .build();
            }
        },
        "Penalty" => {
            if game.us_operating {
                // Penalty kicker
                game.add_role("penalty_kicker")
                    .count(1)
                    .score(|s| score_penalty_kicker(s))
                    .exclude(|s| s.player_id == 0)
                    .behavior(|| build_penalty_kicker_tree())
                    .build();
            }
        }
    }

    // Standard striker role (simplified without special game mode handling)
    game.add_role("striker")
        .max(1)
        .score(|s| striker_scorer::score_striker(s))
        .exclude(|s| s.player_id == 0)
        .behavior(|| striker::build_striker_tree())
        .build();

    // Dynamic waller role
    game.add_role("waller")
        .min(0).max(2)
        .score(|s| defender_scorer::score_as_waller(s))
        .exclude(|s| s.player_id == 0)
        .behavior(|| waller::build_waller_tree())
        .build();

    // Dynamic harasser role
    game.add_role("harasser")
        .min(0).max(2)
        .score(|s| defender_scorer::score_as_harasser(s))
        .exclude(|s| s.player_id == 0)
        .behavior(|| harasser::build_harasser_tree())
        .build();
}

// Specialized role scoring functions
fn score_kickoff_kicker(s) {
    let score = 80.0;
    
    // Prefer robots closer to center
    let center_dist = s.distance_to_position(s.get_field_center());
    score += (1000.0 - center_dist.min(1000.0)) / 20.0;
    
    // Prefer robots not under pressure
    let nearby_opponents = s.get_opp_players_within_radius(s.player.position, 1000.0);
    if nearby_opponents.len() == 0 {
        score += 15.0;
    }
    
    return score;
}

fn score_free_kick_kicker(s) {
    let score = 80.0;
    
    // Prefer robots closest to ball
    let ball_dist = s.distance_to_ball();
    score += (1000.0 - ball_dist.min(1000.0)) / 10.0;
    
    // Prefer robots with good angle to goal
    if s.world.ball != () {
        let goal_pos = s.get_opp_goal_position();
        let ball_pos = s.world.ball.position;
        let ball_to_goal = (goal_pos - ball_pos).unit();
        let robot_to_ball = (ball_pos - s.player.position).unit();
        let angle_alignment = ball_to_goal.x * robot_to_ball.x + ball_to_goal.y * robot_to_ball.y;
        score += angle_alignment * 15.0;
    }
    
    return score;
}

fn score_penalty_kicker(s) {
    let score = 90.0;
    
    // Prefer robots closer to penalty area
    let penalty_dist = s.distance_to_opp_penalty_area();
    score += (2000.0 - penalty_dist.min(2000.0)) / 25.0;
    
    // Consistent assignment based on robot hash
    score += s.player_id.hash_float() * 10.0;
    
    return score;
}

fn score_free_kick_interference(s) {
    let score = 70.0;
    
    // Prefer robots that can position between ball and our goal
    if s.world.ball != () {
        let ball_pos = s.world.ball.position;
        let goal_pos = s.get_own_goal_position();
        let my_pos = s.player.position;
        
        // Calculate positioning score
        let ball_to_goal = goal_pos - ball_pos;
        let ball_to_me = my_pos - ball_pos;
        let projection = (ball_to_me.x * ball_to_goal.x + ball_to_me.y * ball_to_goal.y) / ball_to_goal.norm();
        let projection_ratio = projection / ball_to_goal.norm();
        
        if projection_ratio > 0.2 && projection_ratio < 0.8 {
            score += 20.0;
        }
    }
    
    return score;
}

// Specialized role behavior builders
fn build_kickoff_kicker_tree() {
    return Semaphore(
        Sequence([
            GoToPosition(|s| s.get_field_center(), #{}, "Move to center"),
            Wait(0.5, "Wait for start"),
            FetchBall("Get ball"),
            Guard(|s| s.has_ball(),
                Select([
                    // Try to pass if teammate available
                    Guard(|s| can_pass_to_teammate(s),
                        Sequence([
                            FaceTowardsPosition(|s| find_best_pass_target(s), #{}, "Aim pass"),
                            Kick("Kickoff pass!")
                        ], "Pass kickoff"),
                        "Can pass"
                    ),
                    // Otherwise kick toward goal
                    Sequence([
                        FaceTowardsPosition(|s| s.get_opp_goal_position(), #{}, "Face goal"),
                        Kick("Kickoff!")
                    ], "Direct kickoff")
                ], "Kickoff options"),
                "Have ball?"
            )
        ], "Kickoff sequence"),
        "kickoff_kicker",
        1,
        "Kickoff kicker"
    );
}

fn build_free_kick_kicker_tree() {
    return Semaphore(
        Select([
            // Direct shot if close to goal
            Guard(|s| s.distance_to_position(s.get_opp_goal_position()) < 3000.0 && has_clear_shot(s),
                Sequence([
                    FetchBall("Get ball for free kick"),
                    Guard(|s| s.has_ball(),
                        Sequence([
                            FaceTowardsPosition(|s| find_optimal_shot_target(s), #{}, "Aim at goal"),
                            Wait(0.5, "Aim"),
                            Kick("Free kick shot!")
                        ], "Execute shot"),
                        "Have ball?"
                    )
                ], "Direct shot"),
                "Close shot opportunity"
            ),
            
            // Pass to teammate
            Guard(|s| can_pass_to_teammate(s),
                Sequence([
                    FetchBall("Get ball for free kick"),
                    Guard(|s| s.has_ball(),
                        Sequence([
                            FaceTowardsPosition(|s| find_best_pass_target(s), #{}, "Aim pass"),
                            Wait(0.3, "Aim pass"),
                            Kick("Free kick pass!")
                        ], "Execute pass"),
                        "Have ball?"
                    )
                ], "Pass option"),
                "Can pass to teammate"
            ),
            
            // Default: advance with ball
            Sequence([
                FetchBall("Get ball for free kick"),
                Guard(|s| s.has_ball(),
                    GoToPosition(
                        |s| calculate_advance_position(s),
                        #{ heading: |s| get_heading_to_goal(s), with_ball: true },
                        "Advance with ball"
                    ),
                    "Have ball?"
                )
            ], "Advance")
        ], "Free kick actions"),
        "free_kick_kicker",
        1,
        "Free kick kicker"
    );
}

fn build_penalty_kicker_tree() {
    return Semaphore(
        Sequence([
            // Move to penalty position
            GoToPosition(
                |s| s.get_opp_penalty_mark() + vec2(-300.0, 0.0),
                #{ heading: |s| get_heading_to_goal(s) },
                "Approach penalty"
            ),
            
            // Wait for setup
            Wait(1.5, "Wait for setup"),
            
            // Execute penalty
            FetchBall("Get ball"),
            Guard(|s| s.has_ball(),
                Sequence([
                    FaceTowardsPosition(|s| choose_penalty_target(s), #{}, "Aim penalty"),
                    Wait(0.8, "Final aim"),
                    Kick("Penalty shot!")
                ], "Execute penalty"),
                "Have ball?"
            )
        ], "Penalty sequence"),
        "penalty_kicker",
        1,
        "Penalty kicker"
    );
}

fn build_free_kick_interference_tree() {
    return GoToPosition(
        |s| calculate_free_kick_defense_position(s),
        #{ heading: |s| get_defender_heading(s) },
        "Free kick defense"
    );
}

// Helper functions
fn can_pass_to_teammate(s) {
    if !s.has_ball() { return false; }
    
    let teammates = s.filter_own_players_by(|p| p.id != s.player_id);
    
    for teammate in teammates {
        let ray_result = s.cast_ray(s.world.ball.position, teammate.position);
        if !ray_result.hit {
            return true;
        }
    }
    return false;
}

fn find_best_pass_target(s) {
    let teammates = s.filter_own_players_by(|p| p.id != s.player_id);
    let ball_pos = s.world.ball.position;
    
    let best_target = s.get_opp_goal_position();
    let best_score = -1.0;
    
    for teammate in teammates {
        let ray_result = s.cast_ray(ball_pos, teammate.position);
        if !ray_result.hit {
            let goal_dist = teammate.position.distance_to(s.get_opp_goal_position());
            let score = 10000.0 - goal_dist;
            
            if score > best_score {
                best_score = score;
                best_target = teammate.position;
            }
        }
    }
    
    return best_target;
}

fn has_clear_shot(s) {
    let shot_target = find_optimal_shot_target(s);
    let ray_result = s.cast_ray(s.player.position, shot_target);
    return !ray_result.hit;
}

fn find_optimal_shot_target(s) {
    let goal_corners = s.get_opp_goal_corners();
    if goal_corners.len() >= 2 {
        // Simple strategy: choose corner based on robot hash
        if s.player_id.hash_float() > 0.5 {
            return goal_corners[0];
        } else {
            return goal_corners[1];
        }
    }
    return s.get_opp_goal_position();
}

fn calculate_advance_position(s) {
    let player_pos = s.player.position;
    let goal_pos = s.get_opp_goal_position();
    let direction = (goal_pos - player_pos).unit();
    return player_pos + direction * 1000.0;
}

fn get_heading_to_goal(s) {
    let player_pos = s.player.position;
    let goal_pos = s.get_opp_goal_position();
    return (goal_pos - player_pos).angle_to(vec2(1.0, 0.0));
}

fn choose_penalty_target(s) {
    let goal_corners = s.get_opp_goal_corners();
    if goal_corners.len() >= 2 {
        if s.player_id.hash_float() > 0.5 {
            return goal_corners[0];
        } else {
            return goal_corners[1];
        }
    }
    return s.get_opp_goal_position();
}

fn calculate_free_kick_defense_position(s) {
    if s.world.ball == () {
        return s.get_own_goal_position() + vec2(1000.0, 0.0);
    }
    
    let ball_pos = s.world.ball.position;
    let goal_pos = s.get_own_goal_position();
    let ball_to_goal = (goal_pos - ball_pos).unit();
    
    // Position 700mm from ball toward goal (SSL rule: 500mm minimum)
    return ball_pos + ball_to_goal * 700.0;
}

fn get_defender_heading(s) {
    if s.world.ball == () { return 0.0; }
    let ball_pos = s.world.ball.position;
    let my_pos = s.player.position;
    return (ball_pos - my_pos).angle_to(vec2(1.0, 0.0));
}