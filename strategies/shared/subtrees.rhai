// Updated modular behavior subtrees using modern RobotSituation API
// These are the main behavior trees that handle all game modes

import "shared/utilities" as util;
import "shared/situations" as sit;

// Unified goalkeeper behavior that handles all game modes
fn build_goalkeeper_tree() {
    return Select([
        // Penalty mode behavior
        Guard(|s| sit::is_penalty_mode(s),
            GoToPosition(
                |s| s.get_own_goal_position(),
                #{ heading: 0.0 },
                "Penalty Defense"
            ),
            "Penalty mode"
        ),
        
        // Emergency ball clearing if ball is very close
        Guard(|s| sit::ball_in_own_penalty_area(s) && s.distance_to_ball() < 1000.0,
            Sequence([
                FetchBall("Clear Ball"),
                Guard(|s| s.has_ball(),
                    Sequence([
                        FaceTowardsPosition(|s| s.get_field_center(), #{}, "Face Field"),
                        Kick("Clear!")
                    ], "Execute Clear"),
                    "Have ball?"
                )
            ], "Emergency Clear"),
            "Ball in penalty area"
        ),
        
        // Normal goalkeeper behavior
        GoToPosition(
            |s| util::calculate_goalkeeper_position(s),
            #{ heading: |s| util::get_goalkeeper_heading(s) },
            "Guard Goal"
        )
    ], "Goalkeeper");
}

// Unified striker behavior that handles all game modes
fn build_striker_tree() {
    return Select([
        // Kickoff behavior when we're attacking
        Guard(|s| sit::is_kickoff_mode(s) && s.world.game_state.us_operating,
            Semaphore(
                Sequence([
                    GoToPosition(|s| s.get_field_center(), #{}, "Move to Center"),
                    Wait(0.5, "Wait for Start"),
                    FetchBall("Get Ball"),
                    Guard(|s| s.has_ball(),
                        Sequence([
                            FaceTowardsPosition(|s| s.get_opp_goal_position(), #{}, "Face Goal"),
                            Kick("Kickoff!")
                        ], "Execute Kickoff"),
                        "Have ball?"
                    )
                ], "Kickoff Sequence"),
                "kickoff_kicker",
                1,
                "Kickoff Kicker"
            ),
            "Kickoff attacking"
        ),
        
        // Penalty behavior
        Guard(|s| sit::is_penalty_mode(s) && s.world.game_state.us_operating,
            Semaphore(
                Sequence([
                    GoToPosition(|s| s.get_opp_penalty_mark() + vec2(-200.0, 0.0), #{}, "Approach Penalty Mark"),
                    Wait(1.0, "Wait for Setup"),
                    FetchBall("Get Ball"),
                    Guard(|s| s.has_ball(),
                        Sequence([
                            FaceTowardsPosition(|s| s.get_opp_goal_position(), #{}, "Aim at Goal"),
                            Wait(0.5, "Aim"),
                            Kick("Penalty Shot!")
                        ], "Execute Penalty"),
                        "Have ball?"
                    )
                ], "Penalty Sequence"),
                "penalty_taker",
                1,
                "Penalty Taker"
            ),
            "Penalty attacking"
        ),
        
        // Normal striker behavior
        Semaphore(
            Select([
                // Immediate shot if close and have ball
                Guard(|s| s.has_ball() && sit::close_to_opp_goal(s) && sit::clear_shot_to_goal(s),
                    Sequence([
                        FaceTowardsPosition(|s| s.get_opp_goal_position(), #{}, "Aim at Goal"),
                        Kick("Shoot!")
                    ], "Score Goal"),
                    "Clear shot opportunity"
                ),
                
                // Pass if under pressure and can pass
                Guard(|s| s.has_ball() && sit::under_pressure(s) && sit::can_pass_to_teammate(s),
                    Sequence([
                        FaceTowardsPosition(|s| find_best_pass_target(s), #{}, "Aim Pass"),
                        Kick("Pass!")
                    ], "Pass Ball"),
                    "Under pressure with pass option"
                ),
                
                // Get the ball
                FetchBall("Get Ball"),
                
                // Advance with ball
                Guard(|s| s.has_ball(),
                    GoToPosition(
                        |s| util::calculate_advance_position(s),
                        #{ heading: |s| util::get_heading_to_goal(s), with_ball: true },
                        "Advance with Ball"
                    ),
                    "Have ball?"
                )
            ], "Striker Actions"),
            "striker_role",
            1,
            "Striker Role"
        )
    ], "Striker");
}

// Unified defender behavior that handles all game modes
fn build_defender_tree() {
    return Select([
        // Kickoff support position
        Guard(|s| sit::is_kickoff_mode(s),
            GoToPosition(
                |s| util::get_kickoff_support_position(s),
                #{ heading: 0.0 },
                "Kickoff Support"
            ),
            "Kickoff mode"
        ),
        
        // Penalty support position  
        Guard(|s| sit::is_penalty_mode(s),
            GoToPosition(
                |s| util::get_penalty_support_position(s),
                #{ heading: 0.0 },
                "Penalty Support"
            ),
            "Penalty mode"
        ),
        
        // Block immediate threats
        Guard(|s| sit::ball_moving_toward_our_goal(s) && sit::ball_in_our_half(s),
            GoToPosition(
                |s| util::calculate_intercept_position(s),
                #{ heading: |s| util::get_heading_to_ball(s) },
                "Block Threat"
            ),
            "Ball threatening"
        ),
        
        // Wall positioning
        Guard(|s| sit::opponents_near_ball(s) && sit::ball_in_our_half(s),
            GoToPosition(
                |s| util::calculate_wall_position(s, 1500.0),
                #{ heading: |s| util::get_heading_to_ball(s) },
                "Wall Position"
            ),
            "Opponents attacking"
        ),
        
        // Normal defensive behavior
        GoToPosition(
            |s| util::calculate_defender_position(s),
            #{ heading: |s| util::get_defender_heading(s) },
            "Defend Position"
        )
    ], "Defender");
}

// Unified support behavior that handles all game modes
fn build_support_tree() {
    return Select([
        // Kickoff support position
        Guard(|s| sit::is_kickoff_mode(s),
            GoToPosition(
                |s| util::get_kickoff_support_position(s),
                #{ heading: 0.0 },
                "Kickoff Support"
            ),
            "Kickoff mode"
        ),
        
        // Penalty support position
        Guard(|s| sit::is_penalty_mode(s),
            GoToPosition(
                |s| util::get_penalty_support_position(s),
                #{ heading: 0.0 },
                "Penalty Support"
            ),
            "Penalty mode"
        ),
        
        // Help defend if ball moving toward our goal
        Guard(|s| sit::ball_moving_toward_our_goal(s) && sit::ball_in_our_half(s),
            GoToPosition(
                |s| calculate_defensive_support_position(s),
                #{ heading: |s| util::get_heading_to_ball(s) },
                "Defensive Support"
            ),
            "Ball threatening our goal"
        ),
        
        // Provide attacking support
        Guard(|s| sit::ball_in_opponent_half(s),
            GoToPosition(
                |s| calculate_attacking_support_position(s),
                #{ heading: |s| util::get_heading_to_goal(s) },
                "Attack Support"
            ),
            "Ball in attack"
        ),
        
        // Normal support behavior
        GoToPosition(
            |s| util::calculate_support_position(s),
            #{ heading: |s| util::get_heading_to_goal(s) },
            "Support Position"
        )
    ], "Support");
}

// Specialized legacy behavior components (kept for modularity)
fn build_goalkeeper_behavior() {
    return Select([
        // Emergency ball clearing if ball is very close
        Guard(|s| sit::ball_in_own_penalty_area(s) && s.distance_to_ball() < 1000.0,
            Sequence([
                FetchBall("Clear Ball"),
                Guard(|s| s.has_ball(),
                    Sequence([
                        FaceTowardsPosition(|s| s.get_field_center(), #{}, "Face Field"),
                        Kick("Clear!")
                    ], "Execute Clear"),
                    "Have ball?"
                )
            ], "Emergency Clear"),
            "Ball in penalty area"
        ),
        
        // Normal positioning
        GoToPosition(
            |s| util::calculate_goalkeeper_position(s),
            #{ heading: |s| util::get_goalkeeper_heading(s) },
            "Guard Goal"
        )
    ], "Goalkeeper");
}

fn build_attacker_behavior() {
    return Semaphore(
        Select([
            // Immediate shot if close and have ball
            Guard(|s| s.has_ball() && sit::close_to_opp_goal(s) && sit::clear_shot_to_goal(s),
                Sequence([
                    FaceTowardsPosition(|s| s.get_opp_goal_position(), #{}, "Aim at Goal"),
                    Kick("Shoot!")
                ], "Score Goal"),
                "Clear shot opportunity"
            ),
            
            // Pass if under pressure and can pass
            Guard(|s| s.has_ball() && sit::under_pressure(s) && sit::can_pass_to_teammate(s),
                Sequence([
                    FaceTowardsPosition(|s| find_best_pass_target(s), #{}, "Aim Pass"),
                    Kick("Pass!")
                ], "Pass Ball"),
                "Under pressure with pass option"
            ),
            
            // Intercept ball if it's moving
            Guard(|s| sit::ball_moving_fast(s) && sit::am_closest_to_ball(s),
                InterceptBall("Intercept"),
                "Ball moving and closest"
            ),
            
            // Get ball if close
            Guard(|s| sit::close_to_ball(s),
                FetchBall("Get Ball"),
                "Close to ball"
            ),
            
            // Advance with ball toward goal
            Guard(|s| s.has_ball(),
                GoToPosition(
                    |s| util::calculate_advance_position(s),
                    #{ heading: |s| util::get_heading_to_goal(s), with_ball: true },
                    "Advance with Ball"
                ),
                "Have ball"
            ),
            
            // Move to support position
            GoToPosition(
                |s| calculate_attacking_support_position(s),
                #{ heading: |s| util::get_heading_to_goal(s) },
                "Attack Support"
            )
        ], "Attacker Actions"),
        "attacker_role",
        1,
        "Attacker Role"
    );
}

fn build_supporter_behavior() {
    return Select([
        // Help defend if ball moving toward our goal
        Guard(|s| sit::ball_moving_toward_our_goal(s) && sit::ball_in_our_half(s),
            GoToPosition(
                |s| calculate_defensive_support_position(s),
                #{ heading: |s| util::get_heading_to_ball(s) },
                "Defensive Support"
            ),
            "Ball threatening our goal"
        ),
        
        // Provide attacking support
        Guard(|s| sit::ball_in_opponent_half(s),
            GoToPosition(
                |s| calculate_attacking_support_position(s),
                #{ heading: |s| util::get_heading_to_goal(s) },
                "Attack Support"
            ),
            "Ball in attack"
        ),
        
        // Default support position
        GoToPosition(
            |s| util::get_supporter_pos(s),
            #{ heading: |s| util::get_heading_to_goal(s) },
            "Support Position"
        )
    ], "Supporter");
}

fn build_defender_behavior() {
    return Select([
        // Block immediate threats
        Guard(|s| sit::ball_moving_toward_our_goal(s) && sit::ball_in_our_half(s),
            GoToPosition(
                |s| util::calculate_intercept_position(s),
                #{ heading: |s| util::get_heading_to_ball(s) },
                "Block Threat"
            ),
            "Ball threatening"
        ),
        
        // Wall positioning
        Guard(|s| sit::opponents_near_ball(s) && sit::ball_in_our_half(s),
            GoToPosition(
                |s| util::calculate_wall_position(s, 1500.0),
                #{ heading: |s| util::get_heading_to_ball(s) },
                "Wall Position"
            ),
            "Opponents attacking"
        ),
        
        // Default defensive position
        GoToPosition(
            |s| util::get_defender_pos(s),
            #{ heading: |s| util::get_defender_heading(s) },
            "Defend Position"
        )
    ], "Defender");
}

// Specialized kickoff behaviors using new API
fn build_kickoff_kicker_behavior() {
    return Sequence([
        // Move to center circle
        GoToPosition(
            |s| s.get_field_center(),
            #{ heading: |s| util::get_heading_to_goal(s) },
            "Move to Center"
        ),
        
        // Wait for game start
        Wait(0.5, "Wait for Start"),
        
        // Execute kickoff
        FetchBall("Get Ball"),
        Guard(|s| s.has_ball(),
            Select([
                // Pass to teammate if available
                Guard(|s| sit::can_pass_to_teammate(s),
                    Sequence([
                        FaceTowardsPosition(|s| find_best_pass_target(s), #{}, "Aim Pass"),
                        Kick("Kickoff Pass!")
                    ], "Pass Kickoff"),
                    "Can pass"
                ),
                
                // Otherwise kick toward goal
                Sequence([
                    FaceTowardsPosition(|s| s.get_opp_goal_position(), #{}, "Face Goal"),
                    Kick("Kickoff!")
                ], "Direct Kickoff")
            ], "Kickoff Options"),
            "Have ball?"
        )
    ], "Kickoff Kicker");
}

// Penalty behaviors using new API
fn build_penalty_goalkeeper_behavior() {
    return Select([
        // Try to intercept if ball is coming
        Guard(|s| sit::ball_moving_toward_our_goal(s) && s.distance_to_ball() < 2000.0,
            InterceptBall("Intercept Penalty"),
            "Ball coming"
        ),
        
        // Stay on goal line
        GoToPosition(
            |s| s.get_own_goal_position(),
            #{ heading: 0.0 },
            "Guard Goal Line"
        )
    ], "Penalty Goalkeeper");
}

fn build_penalty_taker_behavior() {
    return Sequence([
        // Move to penalty area
        GoToPosition(
            |s| s.get_opp_penalty_mark() + vec2(-200.0, 0.0), // 200mm behind mark
            #{ heading: |s| util::get_heading_to_goal(s) },
            "Approach Penalty"
        ),
        
        // Wait for setup
        Wait(1.0, "Wait for Setup"),
        
        // Execute penalty
        FetchBall("Get Ball"),
        Guard(|s| s.has_ball(),
            Sequence([
                // Choose corner based on goalkeeper position
                FaceTowardsPosition(|s| choose_penalty_target(s), #{}, "Aim Penalty"),
                Wait(0.3, "Final Aim"),
                Kick("Penalty Shot!")
            ], "Execute Penalty"),
            "Have ball?"
        )
    ], "Penalty Taker");
}

// Helper functions for the subtrees using new API
fn calculate_attacking_support_position(s) {
    if s.world.ball == () {
        return vec2(0.0, 0.0);
    }
    
    let ball_pos = s.world.ball.position;
    let goal_pos = s.get_opp_goal_position();
    
    // Position to receive passes or provide width
    let side_multiplier = if s.player_id.hash_float() > 0.5 { 1.0 } else { -1.0 };
    let width_offset = vec2(0.0, 2000.0 * side_multiplier);
    let depth_offset = vec2(-1000.0, 0.0); // Stay behind ball
    
    return ball_pos + width_offset + depth_offset;
}

fn calculate_defensive_support_position(s) {
    let ball_pos = s.world.ball.position;
    let goal_pos = s.get_own_goal_position();
    
    // Position between ball and goal
    let ball_to_goal = goal_pos - ball_pos;
    let support_ratio = 0.6; // 60% of the way from ball to goal
    
    return ball_pos + ball_to_goal * support_ratio;
}

fn find_best_pass_target(s) {
    let teammates = s.world.own_players;
    let ball_pos = s.world.ball.position;
    
    // Find teammate closest to opponent goal with clear pass
    let best_target = s.get_opp_goal_position();
    let best_score = -1.0;
    
    for teammate in teammates {
        if teammate.id != s.player_id {
            let ray_result = s.cast_ray(ball_pos, teammate.position);
            if ray_result.hit == false {
                // Clear pass available
                let goal_dist = s.distance_to_position(teammate.position);
                let score = 10000.0 - goal_dist; // Closer to goal is better
                if score > best_score {
                    best_score = score;
                    best_target = teammate.position;
                }
            }
        }
    }
    
    return best_target;
}

fn choose_penalty_target(s) {
    let goal_corners = s.get_opp_goal_corners();
    
    // Simple strategy: choose corner opposite to where goalkeeper is leaning
    if goal_corners.len() >= 2 {
        // Use hash to consistently choose same corner for this robot
        if s.player_id.hash_float() > 0.5 {
            return goal_corners[0];
        } else {
            return goal_corners[1];
        }
    }
    
    // Fallback to goal center
    return s.get_opp_goal_position();
} 