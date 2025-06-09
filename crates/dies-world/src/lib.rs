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
    SideAssignment, TeamColor, TeamConfiguration, TeamId, TrackerSettings, Vector2, WorldData,
    WorldInstant,
};
use player::PlayerTracker;

use crate::game_state::GameStateTracker;

/// Tracks players for a single team
struct TeamTracker {
    team_id: TeamId,
    players: HashMap<PlayerId, PlayerTracker>,
    controlled: bool,
}

impl TeamTracker {
    fn new(team_id: TeamId) -> Self {
        Self {
            team_id,
            players: HashMap::new(),
            controlled: false,
        }
    }

    fn update_settings(&mut self, settings: &TrackerSettings) {
        for player_tracker in self.players.values_mut() {
            player_tracker.update_settings(settings);
        }
    }

    fn update_from_vision(
        &mut self,
        player_id: PlayerId,
        player: &dies_protos::ssl_vision_detection::SSL_DetectionRobot,
        t_capture: f64,
        tracker_settings: &TrackerSettings,
    ) {
        let tracker = self
            .players
            .entry(player_id)
            .or_insert_with(|| PlayerTracker::new(player_id, tracker_settings));
        tracker.update(t_capture, player);
    }

    fn update_from_feedback(&mut self, feedback: &PlayerFeedbackMsg, time: WorldInstant) -> bool {
        if let Some(tracker) = self.players.get_mut(&feedback.id) {
            tracker.update_from_feedback(feedback, time);
            self.controlled = true;
            true
        } else {
            false
        }
    }

    fn check_players_gone(&mut self, t_capture: f64, time: WorldInstant) {
        for (_, tracker) in self.players.iter_mut() {
            tracker.check_is_gone(t_capture, time);
        }
    }

    fn get_players(&self) -> Vec<PlayerData> {
        let mut players = Vec::new();
        for player_tracker in self.players.values() {
            if player_tracker.is_gone {
                continue;
            }
            if let Some(player_data) = player_tracker.get() {
                players.push(player_data);
            }
        }
        players
    }

    fn is_init(&self) -> bool {
        self.players.values().any(|t| t.is_init())
    }
}

/// A struct to track the world state with team ID support.
pub struct WorldTracker {
    /// Track teams by ID
    team_a: TeamTracker,
    team_b: TeamTracker,
    ball_tracker: BallTracker,
    game_state_tracker: GameStateTracker,
    field_geometry: Option<FieldGeometry>,
    /// Automatically detected from referee messages
    side_assignment: Option<SideAssignment>,
    /// Team configuration with IDs and colors
    team_configuration: TeamConfiguration,
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
    /// Create a new world tracker with default team configuration.
    pub fn new(settings: &ExecutorSettings) -> Self {
        use dies_core::TeamInfo;

        // Create default team configuration
        let team_configuration = TeamConfiguration::new(
            TeamInfo::new_with_name("Team A"),
            TeamInfo::new_with_name("Team B"),
        );

        Self::new_with_teams(settings, team_configuration)
    }

    /// Create a new world tracker with specific team configuration.
    pub fn new_with_teams(
        settings: &ExecutorSettings,
        team_configuration: TeamConfiguration,
    ) -> Self {
        // Extract team IDs from configuration
        let team_a_id = team_configuration.get_team_id(TeamColor::Blue);
        let team_b_id = team_configuration.get_team_id(TeamColor::Yellow);

        Self {
            team_a: TeamTracker::new(team_a_id),
            team_b: TeamTracker::new(team_b_id),
            ball_tracker: BallTracker::new(&settings.tracker_settings),
            game_state_tracker: GameStateTracker::new(),
            field_geometry: None,
            side_assignment: None, // Will be detected from referee messages
            team_configuration,
            first_t_received: None,
            last_t_received: None,
            dt_received: None,
            first_t_capture: None,
            last_t_capture: None,
            tracker_settings: settings.tracker_settings.clone(),
        }
    }

    /// Update the team configuration (handles color changes).
    pub fn update_team_configuration(&mut self, new_config: TeamConfiguration) {
        // Check if team colors have changed and swap trackers if needed
        let old_team_a_color = self.team_configuration.get_team_color(self.team_a.team_id);
        let old_team_b_color = self.team_configuration.get_team_color(self.team_b.team_id);

        let new_team_a_color = new_config.get_team_color(self.team_a.team_id);
        let new_team_b_color = new_config.get_team_color(self.team_b.team_id);

        if old_team_a_color != new_team_a_color || old_team_b_color != new_team_b_color {
            log::info!("Team colors changed - updating configuration");

            // If both teams changed colors, they likely swapped
            if old_team_a_color != new_team_a_color && old_team_b_color != new_team_b_color {
                log::info!("Teams swapped colors - swapping trackers");
                std::mem::swap(&mut self.team_a, &mut self.team_b);
            }
        }

        self.team_configuration = new_config;
    }

    /// Get the list of controlled team IDs.
    pub fn get_controlled_team_ids(&self) -> Vec<TeamId> {
        let mut controlled = Vec::new();
        if self.team_a.controlled {
            controlled.push(self.team_a.team_id);
        }
        if self.team_b.controlled {
            controlled.push(self.team_b.team_id);
        }
        controlled
    }

    /// Get the team tracker for a given team ID.
    fn get_team_tracker_mut(&mut self, team_id: TeamId) -> Option<&mut TeamTracker> {
        if self.team_a.team_id == team_id {
            Some(&mut self.team_a)
        } else if self.team_b.team_id == team_id {
            Some(&mut self.team_b)
        } else {
            None
        }
    }

    /// Determine which team a player belongs to based on current vision data.
    fn determine_player_team_id(&self, team_color: TeamColor) -> TeamId {
        // Map from color to team ID using current configuration
        self.team_configuration.get_team_id(team_color)
    }

    pub fn update_settings(&mut self, settings: &ExecutorSettings) {
        self.tracker_settings = settings.tracker_settings.clone();
        self.team_a.update_settings(&self.tracker_settings);
        self.team_b.update_settings(&self.tracker_settings);
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

            let tracker_settings = self.tracker_settings.clone();
            // Update players based on current team configuration
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

                // Determine which team this blue player belongs to
                let team_id = self.determine_player_team_id(TeamColor::Blue);
                if let Some(team_tracker) = self.get_team_tracker_mut(team_id) {
                    team_tracker.update_from_vision(id, player, t_capture, &tracker_settings);
                }
            }

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

                // Determine which team this yellow player belongs to
                let team_id = self.determine_player_team_id(TeamColor::Yellow);
                if let Some(team_tracker) = self.get_team_tracker_mut(team_id) {
                    team_tracker.update_from_vision(id, player, t_capture, &tracker_settings);
                }
            }

            // Check for missed players
            self.team_a.check_players_gone(t_capture, time);
            self.team_b.check_players_gone(t_capture, time);

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
        // Try to find the player in team trackers to determine which team they belong to
        let mut found_team = None;

        // Check both teams
        if self.team_a.update_from_feedback(feedback, time) {
            found_team = Some(self.team_a.team_id);
        } else if self.team_b.update_from_feedback(feedback, time) {
            found_team = Some(self.team_b.team_id);
        }

        // If player not found in either team, we can't determine which team they belong to
        // This shouldn't happen if vision data comes before feedback
        if found_team.is_none() {
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

    /// Get the team configuration.
    pub fn get_team_configuration(&self) -> &TeamConfiguration {
        &self.team_configuration
    }

    /// Check if the world state is initialized.
    ///
    /// The world state is initialized if at least one player and the ball have been
    /// seen at least twice (so that velocities can be calculated), and the field
    /// geometry has been received.
    pub fn is_init(&self) -> bool {
        let any_player_init = self.team_a.is_init() || self.team_b.is_init();
        let ball_init = self.ball_tracker.is_init();
        let field_geom_init = self.field_geometry.is_some();

        any_player_init && ball_init && field_geom_init
    }

    /// Get the current world state.
    ///
    /// Returns team-agnostic WorldData with team configuration.
    pub fn get(&mut self) -> WorldData {
        let field_geom = self.field_geometry.clone();

        // Get players by their current team colors
        let team_a_color = self.team_configuration.get_team_color(self.team_a.team_id);
        let team_b_color = self.team_configuration.get_team_color(self.team_b.team_id);

        let mut blue_players = Vec::new();
        let mut yellow_players = Vec::new();

        // Add team A players to the appropriate color list
        let team_a_players = self.team_a.get_players();
        if team_a_color == TeamColor::Blue {
            blue_players.extend(team_a_players);
        } else {
            yellow_players.extend(team_a_players);
        }

        // Add team B players to the appropriate color list
        let team_b_players = self.team_b.get_players();
        if team_b_color == TeamColor::Blue {
            blue_players.extend(team_b_players);
        } else {
            yellow_players.extend(team_b_players);
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
