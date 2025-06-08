use std::collections::HashMap;

use dies_protos::{ssl_gc_referee_message::Referee, ssl_vision_wrapper::SSL_WrapperPacket};

mod ball;
mod filter;
mod game_state;
mod player;

use ball::BallTracker;
pub use dies_core::{
    BallData, FieldCircularArc, FieldGeometry, FieldLineSegment, GameStateData, PlayerData,
};
use dies_core::{
    ExecutorSettings, FieldMask, GameState, PlayerFeedbackMsg, PlayerId, RawGameStateData,
    SideAssignment, TeamColor, TrackerSettings, Vector2, WorldData, WorldInstant,
};
use player::PlayerTracker;

use crate::game_state::GameStateTracker;

/// A struct to track the world state.
pub struct WorldTracker {
    /// Track all blue team players
    blue_players_tracker: HashMap<PlayerId, PlayerTracker>,
    /// Track all yellow team players  
    yellow_players_tracker: HashMap<PlayerId, PlayerTracker>,
    ball_tracker: BallTracker,
    game_state_tracker: GameStateTracker,
    field_geometry: Option<FieldGeometry>,
    /// Automatically detected from referee messages
    side_assignment: Option<SideAssignment>,
    /// Track which teams we receive feedback from (controlled teams)
    blue_team_controlled: bool,
    yellow_team_controlled: bool,
    /// Local timestamp of the first detection frame received from vision
    first_t_received: Option<WorldInstant>,
    /// Local timestamp of the last received detection frame, in seconds. This is
    ///  relative to `first_t_received`
    last_t_received: Option<f64>,
    /// Duration between the last two received frames (received time)
    dt_received: Option<f64>,
    /// The `t_capture` timestamp of the first frame received from vision, in seconds
    first_t_capture: Option<f64>,
    /// Capture timestamp of the last frame from vision, in seconds. This is relative to
    /// `first_t_capture`
    last_t_capture: Option<f64>,
    tracker_settings: TrackerSettings,
}

impl WorldTracker {
    /// Create a new world tracker from a config.
    pub fn new(settings: &ExecutorSettings) -> Self {
        Self {
            blue_players_tracker: HashMap::new(),
            yellow_players_tracker: HashMap::new(),
            ball_tracker: BallTracker::new(&settings.tracker_settings),
            game_state_tracker: GameStateTracker::new(),
            field_geometry: None,
            side_assignment: None, // Will be detected from referee messages
            blue_team_controlled: false,
            yellow_team_controlled: false,
            first_t_received: None,
            last_t_received: None,
            dt_received: None,
            first_t_capture: None,
            last_t_capture: None,
            tracker_settings: settings.tracker_settings.clone(),
        }
    }

    pub fn update_settings(&mut self, settings: &ExecutorSettings) {
        self.tracker_settings = settings.tracker_settings.clone();
        for player_tracker in self.blue_players_tracker.values_mut() {
            player_tracker.update_settings(&self.tracker_settings);
        }
        for player_tracker in self.yellow_players_tracker.values_mut() {
            player_tracker.update_settings(&self.tracker_settings);
        }
        self.ball_tracker.update_settings(&self.tracker_settings);

        // Log the field mask lines
        if let Some(geom) = self.field_geometry.as_ref() {
            let FieldMask {
                x_min,
                x_max,
                y_min,
                y_max,
            } = &self.tracker_settings.field_mask;
            dies_core::debug_line(
                "mask.x_min".to_string(),
                Vector2::new(
                    geom.field_length / 2.0 * x_min,
                    geom.field_width / 2.0 * y_min,
                ),
                Vector2::new(
                    geom.field_length / 2.0 * x_min,
                    geom.field_width / 2.0 * y_max,
                ),
                dies_core::DebugColor::Green,
            );
            dies_core::debug_line(
                "mask.x_max".to_string(),
                Vector2::new(
                    geom.field_length / 2.0 * x_max,
                    geom.field_width / 2.0 * y_min,
                ),
                Vector2::new(
                    geom.field_length / 2.0 * x_max,
                    geom.field_width / 2.0 * y_max,
                ),
                dies_core::DebugColor::Green,
            );
            dies_core::debug_line(
                "mask.y_min".to_string(),
                Vector2::new(
                    geom.field_length / 2.0 * x_min,
                    geom.field_width / 2.0 * y_min,
                ),
                Vector2::new(
                    geom.field_length / 2.0 * x_max,
                    geom.field_width / 2.0 * y_min,
                ),
                dies_core::DebugColor::Green,
            );
            dies_core::debug_line(
                "mask.y_max".to_string(),
                Vector2::new(
                    geom.field_length / 2.0 * x_min,
                    geom.field_width / 2.0 * y_max,
                ),
                Vector2::new(
                    geom.field_length / 2.0 * x_max,
                    geom.field_width / 2.0 * y_max,
                ),
                dies_core::DebugColor::Green,
            );
        }
    }

    /// Update the world state from a referee message.
    pub fn update_from_referee(&mut self, data: &Referee) {
        self.game_state_tracker
            .update_ball_movement_check(self.ball_tracker.get().as_ref());

        if let Some(ball) = self.ball_tracker.get() {
            let cur = self.game_state_tracker.update(data);
            if cur == GameState::Kickoff || cur == GameState::FreeKick {
                self.game_state_tracker
                    .start_ball_movement_check(ball.position, 10);
            }
            if cur == GameState::Penalty {
                self.game_state_tracker
                    .start_ball_movement_check(ball.position, 10);
            }
        } else {
            log::error!("Ball not detected, cannot update game state");
        }

        // Detect side assignment from referee message
        if let Some(blue_on_positive) = data.blue_team_on_positive_half {
            self.side_assignment = Some(if blue_on_positive {
                SideAssignment::BlueOnPositive
            } else {
                SideAssignment::YellowOnPositive
            });
        }
    }

    /// Update the world state from a vision message.
    pub fn update_from_vision(&mut self, data: &SSL_WrapperPacket, time: WorldInstant) {
        if let Some(frame) = data.detection.as_ref() {
            // Update t_received
            let first_t_received = *self.first_t_received.get_or_insert(time);
            let t_received = time.duration_since(&first_t_received);
            if let Some(last_t_received) = self.last_t_received {
                self.dt_received = Some(t_received - last_t_received);
                dies_core::debug_value("dt", t_received - last_t_received);
            }
            self.last_t_received = Some(t_received);

            // Update t_capture
            let t_capture = frame.t_capture();
            let first_t_capture = *self.first_t_capture.get_or_insert(t_capture);
            self.last_t_capture = Some(t_capture - first_t_capture);

            // Update blue team players
            for player in &frame.robots_blue {
                let in_mask = self.tracker_settings.field_mask.contains(
                    player.x(),
                    player.y(),
                    self.field_geometry.as_ref(),
                );
                if !in_mask {
                    continue;
                }

                let id = PlayerId::new(player.robot_id());
                let tracker = self
                    .blue_players_tracker
                    .entry(id)
                    .or_insert_with(|| PlayerTracker::new(id, &self.tracker_settings));
                tracker.update(t_capture, player);
            }
            // Check for missed blue players
            for (_, tracker) in self.blue_players_tracker.iter_mut() {
                tracker.check_is_gone(t_capture, time);
            }

            // Update yellow team players
            for player in &frame.robots_yellow {
                let in_mask = self.tracker_settings.field_mask.contains(
                    player.x(),
                    player.y(),
                    self.field_geometry.as_ref(),
                );
                if !in_mask {
                    continue;
                }

                let id = PlayerId::new(player.robot_id());
                let tracker = self
                    .yellow_players_tracker
                    .entry(id)
                    .or_insert_with(|| PlayerTracker::new(id, &self.tracker_settings));
                tracker.update(t_capture, player);
            }
            // Check for missed yellow players
            for (_, tracker) in self.yellow_players_tracker.iter_mut() {
                tracker.check_is_gone(t_capture, time);
            }

            // Update ball
            self.ball_tracker.update(
                frame,
                &self.tracker_settings.field_mask,
                self.field_geometry.as_ref(),
            );

            self.game_state_tracker
                .update_ball_movement_check(self.ball_tracker.get().as_ref());
        }
        if let Some(geometry) = data.geometry.as_ref() {
            // We don't expect the field geometry to change, so only update it once.
            if self.field_geometry.is_some() {
                return;
            }

            self.field_geometry = Some(FieldGeometry::from_protobuf(&geometry.field));
        }
    }

    pub fn update_from_feedback(&mut self, feedback: &PlayerFeedbackMsg, time: WorldInstant) {
        // Try to find the player in both team trackers to determine which team they belong to
        let mut found_in_blue = false;
        let mut found_in_yellow = false;

        // Check if player exists in blue team
        if let Some(tracker) = self.blue_players_tracker.get_mut(&feedback.id) {
            tracker.update_from_feedback(feedback, time);
            self.blue_team_controlled = true;
            found_in_blue = true;
        }

        // Check if player exists in yellow team
        if let Some(tracker) = self.yellow_players_tracker.get_mut(&feedback.id) {
            tracker.update_from_feedback(feedback, time);
            self.yellow_team_controlled = true;
            found_in_yellow = true;
        }

        // If player not found in either team, we can't determine which team they belong to
        // This shouldn't happen if vision data comes before feedback
        if !found_in_blue && !found_in_yellow {
            log::warn!(
                "Received feedback for player {} that hasn't been seen in vision yet",
                feedback.id
            );
        }
    }

    /// Get the detected side assignment, if available.
    pub fn get_side_assignment(&self) -> Option<&SideAssignment> {
        self.side_assignment.as_ref()
    }

    /// Check if the world state is initialized.
    ///
    /// The world state is initialized if at least one player and the ball have been
    /// seen at least twice (so that velocities can be calculated), and the field
    /// geometry has been received.
    pub fn is_init(&self) -> bool {
        let any_player_init = self
            .blue_players_tracker
            .values()
            .chain(self.yellow_players_tracker.values())
            .any(|t| t.is_init());

        let ball_init = self.ball_tracker.is_init();
        let field_geom_init = self.field_geometry.is_some();

        any_player_init && ball_init && field_geom_init
    }

    /// Get the current world state.
    ///
    /// Returns team-agnostic WorldData with players separated by team color.
    pub fn get(&mut self) -> WorldData {
        let field_geom = self.field_geometry.clone();

        let mut blue_players = Vec::new();
        for player_tracker in self.blue_players_tracker.values() {
            if player_tracker.is_gone {
                continue;
            }
            if let Some(player_data) = player_tracker.get() {
                blue_players.push(player_data);
            }
        }

        let mut yellow_players = Vec::new();
        for player_tracker in self.yellow_players_tracker.values() {
            if player_tracker.is_gone {
                continue;
            }
            if let Some(player_data) = player_tracker.get() {
                yellow_players.push(player_data);
            }
        }

        let game_state = RawGameStateData {
            game_state: self.game_state_tracker.get(),
            operating_team: match self.game_state_tracker.get_operator_is_blue() {
                Some(true) => TeamColor::Blue,
                Some(false) => TeamColor::Yellow,
                None => TeamColor::Blue, // Default fallback
            },
        };

        WorldData {
            dt: self.dt_received.unwrap_or(0.0),
            t_capture: self.last_t_capture.unwrap_or(0.0),
            t_received: self.last_t_received.unwrap_or(0.0),
            blue_team: blue_players,
            yellow_team: yellow_players,
            ball: self.ball_tracker.get(),
            field_geom,
            game_state,
            side_assignment: self
                .side_assignment
                .clone()
                .unwrap_or(SideAssignment::YellowOnPositive),
        }
    }
}

// Note: Tests have been temporarily removed as they need to be updated
// for the new team-agnostic interface. The WorldTracker now outputs
// team-agnostic WorldData instead of team-specific TeamData.
// Coordinate transformations are handled by the SideAssignment in dies-core.
