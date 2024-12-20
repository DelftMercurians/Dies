use std::collections::HashMap;

use dies_protos::{
    ssl_gc_referee_message::Referee, ssl_vision_detection::SSL_DetectionRobot,
    ssl_vision_wrapper::SSL_WrapperPacket,
};

mod ball;
mod filter;
mod game_state;
pub mod geom;
mod player;
mod tracker_settings;
mod world_frame;

use tracker_settings::{FieldMask, TrackerSettings};
pub use world_frame::*;

use crate::game_state::GameStateTracker;
use ball::BallTracker;
use dies_core::{PlayerFeedbackMsg, PlayerId, Vector2};
pub use geom::{FieldCircularArc, FieldGeometry, FieldLineSegment};
use player::PlayerTracker;

/// A struct to track the world state.
pub struct WorldTracker {
    /// Player trackers for each team
    blue_players: HashMap<PlayerId, PlayerTracker>,
    blue_controlled: bool,
    yellow_players: HashMap<PlayerId, PlayerTracker>,
    yellow_controlled: bool,

    side_assignment: SideAssignment,

    ball_tracker: BallTracker,
    game_state_tracker: GameStateTracker,
    field_geometry: Option<FieldGeometry>,

    /// Time tracking for frame synchronization
    first_t_received: Option<WorldInstant>,
    last_t_received: Option<f64>,
    dt_received: Option<f64>,
    first_t_capture: Option<f64>,
    last_t_capture: Option<f64>,

    tracker_settings: TrackerSettings,
}

impl WorldTracker {
    /// Create a new world tracker from settings.
    pub fn new(initial_side_assignment: SideAssignment, settings: &TrackerSettings) -> Self {
        Self {
            blue_controlled: false,
            yellow_controlled: false,
            side_assignment: initial_side_assignment,
            blue_players: HashMap::new(),
            yellow_players: HashMap::new(),
            ball_tracker: BallTracker::new(settings),
            game_state_tracker: GameStateTracker::new(),
            field_geometry: None,
            first_t_received: None,
            last_t_received: None,
            dt_received: None,
            first_t_capture: None,
            last_t_capture: None,
            tracker_settings: settings.clone(),
        }
    }

    pub fn reset(&mut self) {
        self.blue_players.clear();
        self.yellow_players.clear();
        self.ball_tracker = BallTracker::new(&self.tracker_settings);
        self.game_state_tracker = GameStateTracker::new();
        self.field_geometry = None;
        self.first_t_received = None;
        self.last_t_received = None;
        self.dt_received = None;
        self.first_t_capture = None;
        self.last_t_capture = None;
    }

    pub fn set_controlled(&mut self, blue: bool, yellow: bool) {
        if self.blue_controlled != blue || self.yellow_controlled != yellow {
            self.blue_controlled = blue;
            self.yellow_controlled = yellow;
            self.reset();
        }
    }

    pub fn update_settings(&mut self, settings: &TrackerSettings) {
        self.tracker_settings = settings.clone();

        // Update settings for all player trackers
        for player in self.blue_players.values_mut() {
            player.update_settings(&self.tracker_settings);
        }
        for player in self.yellow_players.values_mut() {
            player.update_settings(&self.tracker_settings);
        }

        self.ball_tracker.update_settings(&self.tracker_settings);

        // Log the field mask lines for debugging
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
            dies_core::debug_line(
                "mask.x_min".to_string(),
                Vector2::new(
                    geom.field_length / 2.0 * x_min,
                    geom.field_width / 2.0 * y_min,
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
        // Look for side assignment changes
        let blue_on_positive = data.blue_team_on_positive_half();
        if blue_on_positive && self.side_assignment != SideAssignment::BluePositive {
            self.side_assignment = SideAssignment::BluePositive;
            log::info!("Blue team on positive half");
        } else if !blue_on_positive && self.side_assignment != SideAssignment::YellowPositive {
            self.side_assignment = SideAssignment::YellowPositive;
            log::info!("Yellow team on positive half");
        }

        self.game_state_tracker
            .update_ball_movement_check(self.ball_tracker.get().as_ref());

        if let Some(ball) = self.ball_tracker.get() {
            let cur = self.game_state_tracker.update(data);
            // Handle special game states that require ball movement monitoring
            match cur {
                GameStateType::Kickoff | GameStateType::FreeKick => {
                    self.game_state_tracker
                        .start_ball_movement_check(ball.position, 10);
                }
                GameStateType::Penalty => {
                    self.game_state_tracker
                        .start_ball_movement_check(ball.position, 10);
                }
                _ => {}
            }
        } else {
            log::error!("Ball not detected, cannot update game state");
        }
    }

    /// Update the world state from a vision message.
    pub fn update_from_vision(&mut self, data: &SSL_WrapperPacket, time: WorldInstant) {
        if let Some(frame) = data.detection.as_ref() {
            // Update timing information
            let first_t_received = *self.first_t_received.get_or_insert(time);
            let t_received = time.duration_since(&first_t_received);

            if let Some(last_t_received) = self.last_t_received {
                self.dt_received = Some(t_received - last_t_received);
                dies_core::debug_value("dt", t_received - last_t_received);
            }
            self.last_t_received = Some(t_received);

            let t_capture = frame.t_capture();
            let first_t_capture = *self.first_t_capture.get_or_insert(t_capture);
            self.last_t_capture = Some(t_capture - first_t_capture);

            // Update blue team players
            self.update_team_players(frame.robots_blue.iter(), Team::Blue, t_capture, time);

            // Update yellow team players
            self.update_team_players(frame.robots_yellow.iter(), Team::Yellow, t_capture, time);

            // Update ball
            self.ball_tracker.update(
                frame,
                &self.tracker_settings.field_mask,
                self.field_geometry.as_ref(),
            );

            self.game_state_tracker
                .update_ball_movement_check(self.ball_tracker.get().as_ref());
        }

        // Update field geometry if provided
        if let Some(geometry) = data.geometry.as_ref() {
            // We don't expect the field geometry to change, so only update it once
            if self.field_geometry.is_none() {
                self.field_geometry = Some(FieldGeometry::from_protobuf(&geometry.field));
            }
        }
    }

    /// Helper method to update players for a specific team
    fn update_team_players<'a, I>(
        &mut self,
        players: I,
        team: Team,
        t_capture: f64,
        time: WorldInstant,
    ) where
        I: Iterator<Item = &'a SSL_DetectionRobot>,
    {
        let (players_map, is_controlled) = match team {
            Team::Blue => (&mut self.blue_players, self.blue_controlled),
            Team::Yellow => (&mut self.yellow_players, self.yellow_controlled),
        };

        // Update detected players
        for player in players {
            // Skip players outside the field mask
            if !self.tracker_settings.field_mask.contains(
                player.x(),
                player.y(),
                self.field_geometry.as_ref(),
            ) {
                continue;
            }

            let id = PlayerId::new(player.robot_id());
            if let std::collections::hash_map::Entry::Vacant(e) = players_map.entry(id) {
                // Create new tracker for previously unseen player
                e.insert(PlayerTracker::new(
                    id,
                    &self.tracker_settings,
                    is_controlled,
                ));
            }

            if let Some(tracker) = players_map.get_mut(&id) {
                tracker.update(t_capture, player);
            }
        }

        // Check for missing players
        for (_, tracker) in players_map.iter_mut() {
            tracker.check_is_gone(t_capture, time);
        }
    }

    /// Update player feedback for controlled robots
    pub fn update_from_feedback(
        &mut self,
        feedback: &PlayerFeedbackMsg,
        team: Team,
        time: WorldInstant,
    ) {
        let (players_map, is_controlled) = match team {
            Team::Blue => (&mut self.blue_players, self.blue_controlled),
            Team::Yellow => (&mut self.yellow_players, self.yellow_controlled),
        };

        // Only process feedback for controlled players
        if is_controlled {
            let tracker = players_map
                .entry(feedback.id)
                .or_insert_with(|| PlayerTracker::new(feedback.id, &self.tracker_settings, true));
            tracker.update_from_feedback(feedback, time);
        } else {
            log::warn!(
                "Received feedback for uncontrolled player: {:?}",
                feedback.id
            );
        }
    }

    /// Check if the world state is initialized
    pub fn is_init(&self) -> bool {
        let any_player_init = self
            .blue_players
            .values()
            .chain(self.yellow_players.values())
            .any(|t| t.is_init());

        let ball_init = self.ball_tracker.is_init();
        let field_geom_init = self.field_geometry.is_some();

        any_player_init && ball_init && field_geom_init
    }

    /// Get the current world state
    pub fn get(&mut self) -> WorldFrame {
        // Collect active players for each team
        let blue_team = self
            .blue_players
            .values()
            .filter(|p| !p.is_gone)
            .filter_map(|p| p.get())
            .collect();

        let yellow_team = self
            .yellow_players
            .values()
            .filter(|p| !p.is_gone)
            .filter_map(|p| p.get())
            .collect();

        WorldFrame {
            dt: self.dt_received.unwrap_or(0.0),
            t_capture: self.last_t_capture.unwrap_or(0.0),
            t_received: self.last_t_received.unwrap_or(0.0),
            blue_team,
            yellow_team,
            ball: self.ball_tracker.get(),
            field_geom: self.field_geometry.clone(),
            current_game_state: self.game_state_tracker.get(),
            side_assignment: self.side_assignment,
        }
    }
}
