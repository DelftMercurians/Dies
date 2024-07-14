use std::collections::HashMap;

use dies_protos::ssl_gc_referee_message::Referee;

use dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;

mod ball;
mod coord_utils;
mod filter;
mod game_state;
mod player;

use crate::game_state::GameStateTracker;
use ball::BallTracker;
pub use dies_core::{
    BallData, FieldCircularArc, FieldGeometry, FieldLineSegment, GameStateData, PlayerData,
};
use dies_core::{
    ExecutorSettings, GameState, PlayerFeedbackMsg, PlayerId, PlayerModel, TrackerSettings,
    WorldData, WorldInstant,
};
use player::PlayerTracker;

const IS_DIV_A: bool = false;

/// A struct to track the world state.
pub struct WorldTracker {
    /// Whether our team color is blue
    is_blue: bool,
    /// The sign of the enemy goal's x coordinate in ssl-vision coordinates. Used for
    /// converting coordinates.
    play_dir_x: f64,
    own_players_tracker: HashMap<PlayerId, PlayerTracker>,
    opp_players_tracker: HashMap<PlayerId, PlayerTracker>,
    ball_tracker: BallTracker,
    game_state_tracker: GameStateTracker,
    field_geometry: Option<FieldGeometry>,
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
    player_model: PlayerModel,
}

impl WorldTracker {
    /// Create a new world tracker from a config.
    pub fn new(settings: &ExecutorSettings) -> Self {
        Self {
            is_blue: settings.tracker_settings.is_blue,
            play_dir_x: settings.tracker_settings.initial_opp_goal_x,
            own_players_tracker: HashMap::new(),
            opp_players_tracker: HashMap::new(),
            ball_tracker: BallTracker::new(&settings.tracker_settings),
            game_state_tracker: GameStateTracker::new(settings.tracker_settings.initial_opp_goal_x),
            field_geometry: None,
            first_t_received: None,
            last_t_received: None,
            dt_received: None,
            first_t_capture: None,
            last_t_capture: None,
            tracker_settings: settings.tracker_settings.clone(),
            player_model: settings.into(),
        }
    }

    pub fn update_settings(&mut self, settings: &ExecutorSettings) {
        let tracker_settings = &settings.tracker_settings;
        assert_eq!(
            tracker_settings.initial_opp_goal_x, self.play_dir_x,
            "Changing the initial opponent goal x coordinate is not supported"
        );
        assert_eq!(
            tracker_settings.is_blue, self.is_blue,
            "Changing the team color is not supported"
        );

        self.player_model = settings.into();
        self.tracker_settings = tracker_settings.clone();
        for player_tracker in self.own_players_tracker.values_mut() {
            player_tracker.update_settings(tracker_settings);
        }
        for player_tracker in self.opp_players_tracker.values_mut() {
            player_tracker.update_settings(tracker_settings);
        }
        self.ball_tracker.update_settings(tracker_settings);
    }

    /// Update the sign of the enemy goal's x coordinate (in ssl-vision coordinates).
    pub fn set_play_dir_x(&mut self, sign: f64) {
        self.play_dir_x = sign.signum();
        self.ball_tracker.set_play_dir_x(self.play_dir_x);
        self.game_state_tracker.set_play_dir_x(self.play_dir_x);
        for player_tracker in self.own_players_tracker.values_mut() {
            player_tracker.set_play_dir_x(self.play_dir_x);
        }
        for player_tracker in self.opp_players_tracker.values_mut() {
            player_tracker.set_play_dir_x(self.play_dir_x);
        }
    }

    pub fn get_play_dir_x(&self) -> f64 {
        self.play_dir_x
    }

    /// Update the world state from a referee message.
    pub fn update_from_referee(&mut self, data: &Referee) {
        self.game_state_tracker
            .update_ball_movement_check(self.ball_tracker.get().as_ref());

        let cur = self.game_state_tracker.update(data);
        if cur == GameState::Kickoff || cur == GameState::FreeKick {
            let timeout = if IS_DIV_A { 10 } else { 5 };
            self.game_state_tracker
                .start_ball_movement_check(self.ball_tracker.get().unwrap().position, timeout);
        }
        if cur == GameState::Penalty {
            self.game_state_tracker
                .start_ball_movement_check(self.ball_tracker.get().unwrap().position, 10);
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
            }
            self.last_t_received = Some(t_received);

            // Update t_capture
            let t_capture = frame.t_capture();
            let first_t_capture = *self.first_t_capture.get_or_insert(t_capture);
            self.last_t_capture = Some(t_capture - first_t_capture);

            // Update players
            let (blue_trackers, yellow_tracker) = if self.is_blue {
                (&mut self.own_players_tracker, &mut self.opp_players_tracker)
            } else {
                (&mut self.opp_players_tracker, &mut self.own_players_tracker)
            };

            // Blue players
            for player in data.detection.robots_blue.iter() {
                let id = PlayerId::new(player.robot_id());
                let tracker = blue_trackers
                    .entry(id)
                    .or_insert_with(|| PlayerTracker::new(id, &self.tracker_settings));
                tracker.update(t_capture, player);
            }

            // Yellow players
            for player in data.detection.robots_yellow.iter() {
                let id = PlayerId::new(player.robot_id());
                let tracker = yellow_tracker
                    .entry(id)
                    .or_insert_with(|| PlayerTracker::new(id, &self.tracker_settings));
                tracker.update(t_capture, player);
            }

            // Update ball
            self.ball_tracker.update(frame);
        }
        if let Some(geometry) = data.geometry.as_ref() {
            // We don't expect the field geometry to change, so only update it once.
            if self.field_geometry.is_some() {
                return;
            }

            self.field_geometry = Some(FieldGeometry::from_protobuf(&geometry.field));
        }
    }

    pub fn update_from_feedback(&mut self, feedback: &PlayerFeedbackMsg) {
        if let Some(player_tracker) = self.own_players_tracker.get_mut(&feedback.id) {
            player_tracker.update_from_feedback(feedback);
        }
    }

    /// Check if the world state is initialized.
    ///
    /// The world state is initialized if at least one player and the ball have been
    /// seen at least twice (so that velocities can be calculated), and the field
    /// geometry has been received.
    pub fn is_init(&self) -> bool {
        let any_player_init = self
            .own_players_tracker
            .values()
            .chain(self.opp_players_tracker.values())
            .any(|t| t.is_init());

        let ball_init = self.ball_tracker.is_init();
        let field_geom_init = self.field_geometry.is_some();

        any_player_init && ball_init && field_geom_init
    }

    /// Get the current world state.
    ///
    /// Returns `None` if the world state is not initialized (see
    /// [`WorldTracker::is_init`]).
    pub fn get(&mut self) -> WorldData {
        let field_geom = if let Some(v) = &self.field_geometry {
            Some(v)
        } else {
            None
        };

        let mut own_players = Vec::new();
        for player_tracker in self.own_players_tracker.values() {
            if let Some(player_data) = player_tracker.get() {
                own_players.push(player_data);
            }
        }

        let mut opp_players = Vec::new();
        for player_tracker in self.opp_players_tracker.values() {
            if let Some(player_data) = player_tracker.get() {
                opp_players.push(player_data);
            }
        }

        let game_state = GameStateData {
            game_state: self.game_state_tracker.get(),
            us_operating: match self.game_state_tracker.get_operator_is_blue() {
                Some(true) => self.is_blue,
                Some(false) => !self.is_blue,
                None => true,
            },
        };

        WorldData {
            dt: self.dt_received.unwrap_or(0.0),
            t_capture: self.last_t_capture.unwrap_or(0.0),
            t_received: self.last_t_received.unwrap_or(0.0),
            own_players: own_players.into_iter().collect(),
            opp_players: opp_players.into_iter().collect(),
            ball: self.ball_tracker.get(),
            field_geom: field_geom.cloned(),
            current_game_state: game_state,
            player_model: self.player_model.clone(),
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use dies_protos::{
        ssl_gc_referee_message::referee::Command,
        ssl_vision_detection::{SSL_DetectionBall, SSL_DetectionFrame, SSL_DetectionRobot},
        ssl_vision_geometry::{SSL_GeometryData, SSL_GeometryFieldSize},
    };
    use std::time::Duration;

    #[test]
    fn test_init() {
        let mut tracker = WorldTracker::new(&ExecutorSettings {
            tracker_settings: TrackerSettings {
                is_blue: true,
                initial_opp_goal_x: 1.0,
                ..Default::default()
            },
            ..Default::default()
        });

        // First detection frame
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(1.0);
        // Add ball
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(0.0);
        ball.set_y(0.0);
        ball.set_z(0.0);
        frame.balls.push(ball.clone());
        // Add player
        let mut player = SSL_DetectionRobot::new();
        player.set_robot_id(1);
        player.set_x(100.0);
        player.set_y(200.0);
        player.set_orientation(0.0);
        frame.robots_blue.push(player.clone());
        let mut packet_detection = SSL_WrapperPacket::new();
        packet_detection.detection = Some(frame.clone()).into();

        // Add field geometry
        let mut geom = SSL_GeometryData::new();
        let mut field = SSL_GeometryFieldSize::new();
        field.set_field_length(9000);
        field.set_field_width(6000);
        field.set_goal_width(1000);
        field.set_goal_depth(200);
        field.set_boundary_width(300);
        geom.field = Some(field).into();
        let mut packet_geom = SSL_WrapperPacket::new();
        packet_geom.geometry = Some(geom).into();

        tracker.update_from_vision(&packet_detection, WorldInstant::now_real());
        assert!(!tracker.is_init());

        tracker.update_from_vision(&packet_geom, WorldInstant::now_real());
        tracker.update_from_feedback(&PlayerFeedbackMsg::empty(PlayerId::new(1)));
        assert!(!tracker.is_init());

        // Second detection frame
        frame.set_t_capture(2.0);
        frame.robots_blue.get_mut(0).unwrap().set_x(200.0);
        let mut packet_detection = SSL_WrapperPacket::new();
        packet_detection.detection = Some(frame).into();

        tracker.update_from_vision(&packet_detection, WorldInstant::now_real());
        assert!(tracker.is_init());

        let data = tracker.get();

        // Check player
        assert!(data.own_players.len() == 1);
        assert!(data.opp_players.is_empty());
        assert!(data.own_players[0].raw_position.x == 200.0);
        assert!(data.own_players[0].raw_position.y == 200.0);

        // Check ball
        assert!(data.ball.is_some());
        let ball = data.ball.unwrap();
        assert!(ball.raw_position[0].x == 0.0);
        assert!(ball.raw_position[0].y == 0.0);

        // Check field geometry
        let field_geom = data.field_geom.unwrap();
        assert!(field_geom.field_length == 9000.0);
        assert!(field_geom.field_width == 6000.0);
        assert!(field_geom.goal_width == 1000.0);
        assert!(field_geom.goal_depth == 200.0);
        assert!(field_geom.boundary_width == 300.0);
    }

    pub struct RefereeBuilder {
        /// for simplicity, this can be enhanced later
        pub command: Command,
    }

    impl RefereeBuilder {
        pub fn new(command: Command) -> RefereeBuilder {
            RefereeBuilder { command }
        }

        #[allow(dead_code)]
        pub fn command(mut self, command: Command) -> Self {
            self.command = command;
            self
        }

        pub fn build(self) -> Referee {
            let mut referee = Referee::new();
            referee.set_command(self.command);
            referee
        }
    }

    pub struct Director;
    impl Director {
        pub fn process_commands(commands: Vec<Command>) -> Vec<Referee> {
            commands
                .into_iter()
                .map(|command| RefereeBuilder::new(command).build())
                .collect()
        }
    }

    #[test]
    fn test_game_state_tracker_simple() {
        let mut tracker = WorldTracker::new(&ExecutorSettings {
            tracker_settings: TrackerSettings {
                is_blue: true,
                initial_opp_goal_x: 1.0,
                ..Default::default()
            },
            ..Default::default()
        });
        let messages = Director::process_commands(vec![
            Command::HALT,
            Command::STOP,
            Command::STOP,
            Command::FORCE_START,
        ]);
        tracker.update_from_referee(&messages[0]);
        assert_eq!(tracker.game_state_tracker.get(), GameState::Halt);
        tracker.update_from_referee(&messages[1]);
        assert_eq!(tracker.game_state_tracker.get(), GameState::Stop);
        tracker.update_from_referee(&messages[2]);
        assert_eq!(tracker.game_state_tracker.get(), GameState::Stop);
        tracker.update_from_referee(&messages[3]);
        assert_eq!(tracker.game_state_tracker.get(), GameState::Run);
    }

    #[test]
    fn test_game_state_tracker_freekick() {
        let mut tracker = WorldTracker::new(&ExecutorSettings {
            tracker_settings: TrackerSettings {
                is_blue: true,
                initial_opp_goal_x: 1.0,
                ..Default::default()
            },
            ..Default::default()
        });
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(0.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(1.0);
        ball.set_y(2.0);
        ball.set_z(3.0);
        frame.balls.push(ball.clone());
        tracker.ball_tracker.update(&frame);
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(1.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(2.0);
        ball.set_y(4.0);
        ball.set_z(6.0);
        frame.balls.push(ball.clone());
        tracker.ball_tracker.update(&frame);
        let messages = Director::process_commands(vec![
            Command::STOP,
            Command::DIRECT_FREE_YELLOW,
            Command::DIRECT_FREE_YELLOW,
            Command::STOP,
        ]);

        tracker.update_from_referee(&messages[0]);
        assert_eq!(tracker.game_state_tracker.get(), GameState::Stop);
        tracker.update_from_referee(&messages[1]);
        assert_eq!(tracker.game_state_tracker.get(), GameState::FreeKick);

        std::thread::sleep(Duration::from_secs(6));
        tracker.update_from_referee(&messages[2]);
        assert_eq!(tracker.game_state_tracker.get(), GameState::Run);
        tracker.update_from_referee(&messages[3]);
        assert_eq!(tracker.game_state_tracker.get(), GameState::Stop);
    }

    #[test]
    fn test_game_penalty_stop_early() {
        let mut tracker = WorldTracker::new(&ExecutorSettings {
            tracker_settings: TrackerSettings {
                is_blue: true,
                initial_opp_goal_x: 1.0,
                ..Default::default()
            },
            ..Default::default()
        });
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(0.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(1.0);
        ball.set_y(2.0);
        ball.set_z(3.0);
        frame.balls.push(ball.clone());
        tracker.ball_tracker.update(&frame);
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(1.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(2.0);
        ball.set_y(4.0);
        ball.set_z(6.0);
        frame.balls.push(ball.clone());
        tracker.ball_tracker.update(&frame);
        let messages = Director::process_commands(vec![
            Command::PREPARE_PENALTY_YELLOW,
            Command::NORMAL_START,
            Command::STOP,
        ]);

        tracker.update_from_referee(&messages[0]);
        assert_eq!(tracker.game_state_tracker.get(), GameState::PreparePenalty);
        tracker.update_from_referee(&messages[1]);
        assert_eq!(tracker.game_state_tracker.get(), GameState::Penalty);

        std::thread::sleep(Duration::from_secs(5));

        assert_eq!(tracker.game_state_tracker.get(), GameState::Penalty);

        tracker.update_from_referee(&messages[2]);
        assert_eq!(tracker.game_state_tracker.get(), GameState::Stop);

        std::thread::sleep(Duration::from_secs(6));

        tracker.update_from_referee(&messages[2]);
        assert_eq!(tracker.game_state_tracker.get(), GameState::Stop);
    }
}
