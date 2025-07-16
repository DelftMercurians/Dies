use dies_core::{Angle, Vector2};
use dies_executor::behavior_tree_api::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// Example of state that can be persisted across ticks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoningState {
    pub current_zone: String,
    pub zone_entry_time: Instant,
    pub last_target_position: Vector2,
    pub stability_counter: u32,
    pub zone_preference_score: f64,
}

impl ZoningState {
    pub fn new(zone: String, target: Vector2) -> Self {
        Self {
            current_zone: zone,
            zone_entry_time: Instant::now(),
            last_target_position: target,
            stability_counter: 0,
            zone_preference_score: 0.0,
        }
    }

    pub fn update_zone(&mut self, new_zone: String, target: Vector2) {
        if self.current_zone != new_zone {
            self.current_zone = new_zone;
            self.zone_entry_time = Instant::now();
            self.stability_counter = 0;
        } else {
            self.stability_counter += 1;
        }
        self.last_target_position = target;
    }

    pub fn time_in_zone(&self) -> std::time::Duration {
        self.zone_entry_time.elapsed()
    }

    pub fn is_stable(&self) -> bool {
        self.stability_counter > 30 // 30 ticks = ~1.5 seconds at 20Hz
    }
}

// Example of a stateful zoning function that persists state between ticks
fn stateful_zoning_callback(
    situation: &RobotSituation,
    state: Option<&ZoningState>,
) -> (Vector2, Option<ZoningState>) {
    let best_zone = determine_best_zone(situation);
    let target_position = get_zone_target_position(situation, &best_zone);

    // Check if we should commit to the current zone or allow transitions
    let new_state = match state {
        Some(existing_state) => {
            let mut updated_state = existing_state.clone();

            // Only change zones if we've been in the current zone for a minimum time
            // AND the new zone is significantly better
            let time_in_zone = existing_state.time_in_zone().as_secs_f64();
            let zone_score = score_zone_for_player(situation, &best_zone);
            let current_zone_score = score_zone_for_player(situation, &existing_state.current_zone);

            let should_change_zone = time_in_zone > 2.0 && // Minimum 2 seconds in zone
                zone_score > current_zone_score + 20.0; // Significant improvement required

            if should_change_zone {
                updated_state.update_zone(best_zone, target_position);
            } else {
                // Stay in current zone but update target position
                let current_target =
                    get_zone_target_position(situation, &existing_state.current_zone);
                updated_state.update_zone(existing_state.current_zone.clone(), current_target);
            }

            updated_state.zone_preference_score = zone_score;
            Some(updated_state)
        }
        None => {
            // First time - create new state
            Some(ZoningState::new(best_zone, target_position))
        }
    };

    // Return the target position for the committed zone
    let actual_target = if let Some(ref state) = new_state {
        state.last_target_position
    } else {
        target_position
    };

    (actual_target, new_state)
}

// Stateful heading callback that can maintain preferred orientation
fn stateful_heading_callback(
    situation: &RobotSituation,
    state: Option<&ZoningState>,
) -> (Angle, Option<ZoningState>) {
    let ball_heading = if let Some(ball) = &situation.world.ball {
        let ball_pos = ball.position.xy();
        let my_pos = situation.player_data().position;
        Angle::between_points(my_pos, ball_pos)
    } else {
        Angle::from_radians(0.0)
    };

    // If we have state and we're stable in a zone, use a smoother heading transition
    let heading = if let Some(existing_state) = state {
        if existing_state.is_stable() {
            // Use smooth interpolation towards ball when stable
            ball_heading // For now, just return ball heading
        } else {
            // Quick response when transitioning between zones
            ball_heading
        }
    } else {
        ball_heading
    };

    // Don't modify state in heading callback - that should be managed by position callback
    (heading, state.cloned())
}

// Example of how to use the stateful continuous node in a behavior tree
pub fn build_stateful_striker_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            // Handle kickoff positioning - stay on our side
            guard_node()
                .description("Kickoff positioning")
                .condition(|s| {
                    s.game_state_is_one_of(&[
                        dies_core::GameState::PrepareKickoff,
                        dies_core::GameState::Kickoff,
                    ])
                })
                .then(
                    go_to_position(get_kickoff_striker_position)
                        .with_heading(get_heading_toward_ball)
                        .description("Kickoff positioning")
                        .build(),
                )
                .build(),
        )
        .add(
            // Ball pickup with committing guard
            committing_guard_node()
                .description("Ball pickup opportunity")
                .when(|s| should_pickup_ball(s))
                .until(|_| false)
                .commit_to(
                    semaphore_node()
                        .do_then(fetch_and_shoot())
                        .semaphore_id("striker_ball_pickup".to_string())
                        .max_entry(1)
                        .build(),
                )
                .build(),
        )
        .add(
            // NEW: Stateful zoning that persists zone decisions
            stateful_continuous("stateful_zoning")
                .with_stateful_position(stateful_zoning_callback)
                .with_stateful_heading(stateful_heading_callback)
                .build(),
        )
        .description("Stateful Striker")
        .build()
        .into()
}

// Helper functions (would typically be in utils or elsewhere)
fn determine_best_zone(situation: &RobotSituation) -> String {
    // Simple zone determination based on field position
    if let Some(ball) = &situation.world.ball {
        if ball.position.y > 1000.0 {
            "top".to_string()
        } else if ball.position.y < -1000.0 {
            "bottom".to_string()
        } else {
            "middle".to_string()
        }
    } else {
        "middle".to_string()
    }
}

fn get_zone_target_position(situation: &RobotSituation, zone: &str) -> Vector2 {
    if let Some(field) = &situation.world.field_geom {
        let half_width = field.field_width / 2.0;
        let third_height = field.field_width / 3.0;

        let y_center = match zone {
            "top" => half_width - third_height / 2.0,
            "bottom" => -half_width + third_height / 2.0,
            _ => 0.0, // middle
        };

        // Position in attacking third
        let x = field.field_length / 4.0;
        Vector2::new(x, y_center)
    } else {
        // Default positions
        match zone {
            "top" => Vector2::new(2000.0, 2000.0),
            "bottom" => Vector2::new(2000.0, -2000.0),
            _ => Vector2::new(2000.0, 0.0),
        }
    }
}

fn score_zone_for_player(situation: &RobotSituation, zone: &str) -> f64 {
    let base_score = 50.0;
    let zone_center = get_zone_target_position(situation, zone);

    // Distance penalty
    let distance_penalty = (situation.player_data().position - zone_center).norm() / 100.0;

    // Congestion penalty
    let congestion = situation
        .get_opp_players_within_radius(zone_center, 1500.0)
        .len() as f64;

    base_score - distance_penalty - (congestion * 15.0)
}

fn get_kickoff_striker_position(s: &RobotSituation) -> Vector2 {
    let player_hash = s.player_id_hash() - 0.5;
    let spread_x = -500.0;
    let spread_y = if player_hash < 0.0 {
        -1200.0 - player_hash * 1000.0
    } else {
        1200.0 + player_hash * 1000.0
    };
    Vector2::new(spread_x, spread_y)
}

fn get_heading_toward_ball(s: &RobotSituation) -> Angle {
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        let my_pos = s.player_data().position;
        Angle::between_points(my_pos, ball_pos)
    } else {
        Angle::from_radians(0.0)
    }
}

fn should_pickup_ball(s: &RobotSituation) -> bool {
    if s.game_state_is_not(dies_core::GameState::Run) {
        return false;
    }
    if !s.can_touch_ball() {
        return false;
    }

    let Some(ball) = s.world.ball.as_ref() else {
        return false;
    };

    let closest_opponent_dist = s
        .world
        .opp_players
        .iter()
        .map(|p| (ball.position.xy() - p.position).norm())
        .min_by(|a, b| a.partial_cmp(b).unwrap());

    ball.position.x > -100.0 && closest_opponent_dist.map(|d| d > 300.0).unwrap_or(true)
}

fn fetch_and_shoot() -> BehaviorNode {
    // Placeholder - would use actual fetch_and_shoot implementation
    go_to_position(|s| {
        s.world
            .ball
            .as_ref()
            .map(|b| b.position.xy())
            .unwrap_or_default()
    })
    .build()
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zoning_state_persistence() {
        let initial_target = Vector2::new(1000.0, 500.0);
        let mut state = ZoningState::new("middle".to_string(), initial_target);

        // Test zone stability
        assert!(!state.is_stable());

        // Update same zone multiple times
        for _ in 0..35 {
            state.update_zone("middle".to_string(), initial_target);
        }

        assert!(state.is_stable());
        assert_eq!(state.current_zone, "middle");

        // Test zone transition
        let new_target = Vector2::new(1000.0, 1500.0);
        state.update_zone("top".to_string(), new_target);

        assert!(!state.is_stable());
        assert_eq!(state.current_zone, "top");
        assert_eq!(state.last_target_position, new_target);
    }
}
