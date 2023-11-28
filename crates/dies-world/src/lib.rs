use dies_protos::ssl_gc_referee_message::Referee;

use dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;
mod ball;
mod coord_utils;
mod game_state;
mod player;

use crate::game_state::GameStateTracker;
use ball::BallTracker;
pub use dies_core::{BallData, FieldCircularArc, FieldGeometry, FieldLineSegment, PlayerData};
use dies_core::{GameState, WorldData};
use player::PlayerTracker;

/// The number of players with unique ids in a single team.
///
/// Might be higher than the number of players on the field at a time so there should
/// be a safe margin.
const MAX_PLAYERS: usize = 15;
const IS_DIV_A: bool = false;

/// A struct to configure the world tracker.
#[derive(Clone, Debug)]
pub struct WorldConfig {
    /// Whether our team color is blue
    pub is_blue: bool,
    /// The initial sign of the enemy goal's x coordinate in ssl-vision coordinates.
    pub initial_opp_goal_x: f32,
}

/// A struct to track the world state.
pub struct WorldTracker {
    /// Whether our team color is blue
    is_blue: bool,
    /// The sign of the enemy goal's x coordinate in ssl-vision coordinates. Used for
    /// converting coordinates.
    play_dir_x: f32,
    own_players_tracker: Vec<Option<PlayerTracker>>,
    opp_players_tracker: Vec<Option<PlayerTracker>>,
    ball_tracker: BallTracker,
    game_state_tracker: GameStateTracker,
    field_geometry: Option<FieldGeometry>,
}

impl WorldTracker {
    /// Create a new world tracker from a config.
    pub fn new(config: WorldConfig) -> Self {
        Self {
            is_blue: config.is_blue,
            play_dir_x: config.initial_opp_goal_x,
            own_players_tracker: vec![None; MAX_PLAYERS],
            opp_players_tracker: vec![None; MAX_PLAYERS],
            ball_tracker: BallTracker::new(config.initial_opp_goal_x),
            game_state_tracker: GameStateTracker::new(),
            field_geometry: None,
        }
    }

    /// Update the sign of the enemy goal's x coordinate (in ssl-vision coordinates).
    pub fn set_play_dir_x(&mut self, sign: f32) {
        self.play_dir_x = sign.signum();
        self.ball_tracker.set_play_dir_x(self.play_dir_x);
        for player_tracker in self.own_players_tracker.iter_mut() {
            if let Some(player_tracker) = player_tracker.as_mut() {
                player_tracker.set_play_dir_x(self.play_dir_x);
            }
        }
        for player_tracker in self.opp_players_tracker.iter_mut() {
            if let Some(player_tracker) = player_tracker.as_mut() {
                player_tracker.set_play_dir_x(self.play_dir_x);
            }
        }
    }

    /// Update the world state from a referee message.
    pub fn update_from_referee(&mut self, data: &Referee) {
        self.game_state_tracker
            .update_ball_movement_check(self.ball_tracker.get());
        let cur = self.game_state_tracker.update(&data.command());
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

    /// Update the world state from a protobuf message.
    pub fn update_from_protobuf(&mut self, data: &SSL_WrapperPacket) {
        if let Some(frame) = data.detection.as_ref() {
            let t_capture = frame.t_capture();

            // Update players
            let (blue_trackers, yellow_tracker) = if self.is_blue {
                (&mut self.own_players_tracker, &mut self.opp_players_tracker)
            } else {
                (&mut self.opp_players_tracker, &mut self.own_players_tracker)
            };

            // Blue players
            for player in data.detection.robots_blue.iter() {
                let id = player.robot_id();
                if id as usize >= MAX_PLAYERS {
                    log::error!("Player id {} is too high", id);
                    continue;
                }

                if blue_trackers[id as usize].is_none() {
                    blue_trackers[id as usize] = Some(PlayerTracker::new(id, self.play_dir_x));
                }

                if let Some(tracker) = blue_trackers[id as usize].as_mut() {
                    tracker.update(t_capture, player);
                }
            }

            // Yellow players
            for player in data.detection.robots_yellow.iter() {
                let id = player.robot_id();
                if id as usize >= MAX_PLAYERS {
                    log::error!("Player id {} is too high", id);
                    continue;
                }

                if yellow_tracker[id as usize].is_none() {
                    yellow_tracker[id as usize] = Some(PlayerTracker::new(id, self.play_dir_x));
                }

                if let Some(tracker) = yellow_tracker[id as usize].as_mut() {
                    tracker.update(t_capture, player);
                }
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
            log::debug!("Received field geometry: {:?}", self.field_geometry);
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
            .iter()
            .chain(self.opp_players_tracker.iter())
            .any(|t| t.as_ref().map(|t| t.is_init()).unwrap_or(false));

        let ball_init = self.ball_tracker.is_init();
        let field_geom_init = self.field_geometry.is_some();

        any_player_init && ball_init && field_geom_init
    }

    /// Get the current world state.
    ///
    /// Returns `None` if the world state is not initialized (see
    /// [`WorldTracker::is_init`]).
    pub fn get(&self) -> Option<WorldData> {
        let field_geom = if let Some(v) = &self.field_geometry {
            v
        } else {
            log::warn!("Tried to get world state before field geometry was initialized");
            return None;
        };

        let mut own_players = Vec::new();
        for player_tracker in self.own_players_tracker.iter() {
            if let Some(player_data) = player_tracker.as_ref().and_then(|t| t.get()) {
                own_players.push(player_data);
            }
        }

        let mut opp_players = Vec::new();
        for player_tracker in self.opp_players_tracker.iter() {
            if let Some(player_data) = player_tracker.as_ref().and_then(|t| t.get()) {
                opp_players.push(player_data);
            }
        }

        let ball = if let Some(ball_data) = self.ball_tracker.get() {
            ball_data
        } else {
            log::warn!("Tried to get world state before ball was initialized");
            return None;
        };

        Some(WorldData {
            own_players: own_players.into_iter().cloned().collect(),
            opp_players: opp_players.into_iter().cloned().collect(),
            ball: ball.clone(),
            field_geom: field_geom.clone(),
            current_game_state: self.game_state_tracker.get_game_state(),
        })
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
    fn test_no_data() {
        let tracker = WorldTracker::new(WorldConfig {
            is_blue: true,
            initial_opp_goal_x: 1.0,
        });

        assert!(!tracker.is_init());
        assert!(tracker.get().is_none());
    }

    #[test]
    fn test_init() {
        let mut tracker = WorldTracker::new(WorldConfig {
            is_blue: true,
            initial_opp_goal_x: 1.0,
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

        tracker.update_from_protobuf(&packet_detection);
        assert!(!tracker.is_init());

        tracker.update_from_protobuf(&packet_geom);
        assert!(!tracker.is_init());

        // Second detection frame
        frame.set_t_capture(2.0);
        frame.robots_blue.get_mut(0).unwrap().set_x(200.0);
        let mut packet_detection = SSL_WrapperPacket::new();
        packet_detection.detection = Some(frame).into();

        tracker.update_from_protobuf(&packet_detection);
        assert!(tracker.is_init());

        let data = tracker.get().unwrap();

        // Check player
        assert!(data.own_players.len() == 1);
        assert!(data.opp_players.is_empty());
        assert!(data.own_players[0].position.x == 200.0);
        assert!(data.own_players[0].position.y == 200.0);

        // Check ball
        assert!(data.ball.position.x == 0.0);
        assert!(data.ball.position.y == 0.0);

        // Check field geometry
        assert!(data.field_geom.field_length == 9000);
        assert!(data.field_geom.field_width == 6000);
        assert!(data.field_geom.goal_width == 1000);
        assert!(data.field_geom.goal_depth == 200);
        assert!(data.field_geom.boundary_width == 300);
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
        let mut tracker = WorldTracker::new(WorldConfig {
            is_blue: true,
            initial_opp_goal_x: 1.0,
        });
        let messages = Director::process_commands(vec![
            Command::HALT,
            Command::STOP,
            Command::STOP,
            Command::FORCE_START,
        ]);
        tracker.update_from_referee(&messages[0]);
        assert_eq!(tracker.game_state_tracker.get_game_state(), GameState::Halt);
        tracker.update_from_referee(&messages[1]);
        assert_eq!(tracker.game_state_tracker.get_game_state(), GameState::Stop);
        tracker.update_from_referee(&messages[2]);
        assert_eq!(tracker.game_state_tracker.get_game_state(), GameState::Stop);
        tracker.update_from_referee(&messages[3]);
        assert_eq!(tracker.game_state_tracker.get_game_state(), GameState::Run);
    }

    #[test]
    fn test_game_state_tracker_freekick() {
        let mut tracker = WorldTracker::new(WorldConfig {
            is_blue: true,
            initial_opp_goal_x: 1.0,
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
        assert_eq!(tracker.game_state_tracker.get_game_state(), GameState::Stop);
        tracker.update_from_referee(&messages[1]);
        assert_eq!(
            tracker.game_state_tracker.get_game_state(),
            GameState::FreeKick
        );

        std::thread::sleep(Duration::from_secs(6));
        tracker.update_from_referee(&messages[2]);
        assert_eq!(tracker.game_state_tracker.get_game_state(), GameState::Run);
        tracker.update_from_referee(&messages[3]);
        assert_eq!(tracker.game_state_tracker.get_game_state(), GameState::Stop);
    }

    #[test]
    fn test_game_penalty_stop_early() {
        let mut tracker = WorldTracker::new(WorldConfig {
            is_blue: true,
            initial_opp_goal_x: 1.0,
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
        assert_eq!(
            tracker.game_state_tracker.get_game_state(),
            GameState::PreparePenalty
        );
        tracker.update_from_referee(&messages[1]);
        assert_eq!(
            tracker.game_state_tracker.get_game_state(),
            GameState::Penalty
        );

        std::thread::sleep(Duration::from_secs(5));

        assert_eq!(
            tracker.game_state_tracker.get_game_state(),
            GameState::Penalty
        );

        tracker.update_from_referee(&messages[2]);
        assert_eq!(tracker.game_state_tracker.get_game_state(), GameState::Stop);

        std::thread::sleep(Duration::from_secs(6));

        tracker.update_from_referee(&messages[2]);
        assert_eq!(tracker.game_state_tracker.get_game_state(), GameState::Stop);
    }
}
