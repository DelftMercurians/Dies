use std::{cell::Cell, rc::Rc};

use serde::Serialize;

use dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;

mod ball;
mod coord_utils;
mod geom;
mod player;

use ball::BallTracker;
use player::PlayerTracker;

pub use ball::BallData;
pub use geom::{FieldCircularArc, FieldGeometry, FieldLineSegment};
pub use player::PlayerData;

/// The number of players with unique ids in a single team.
///
/// Might be higher than the number of players on the field at a time so there should
/// be a safe margin.
const MAX_PLAYERS: usize = 15;

/// A struct to store the world state from a single frame.
#[derive(Serialize, Clone, Debug)]
pub struct WorldData<'a> {
    own_players: Vec<&'a PlayerData>,
    opp_players: Vec<&'a PlayerData>,
    ball: &'a BallData,
    field_geom: &'a FieldGeometry,
}

/// A struct to configure the world tracker.
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
    opp_goal_x_sign: Rc<Cell<f32>>,
    own_players_tracker: Vec<Option<PlayerTracker>>,
    opp_players_tracker: Vec<Option<PlayerTracker>>,
    ball_tracker: BallTracker,
    field_geometry: Option<FieldGeometry>,
}

impl WorldTracker {
    /// Create a new world tracker from a config.
    pub fn new(config: WorldConfig) -> Self {
        let opp_goal_x_sign = Rc::new(Cell::new(config.initial_opp_goal_x.signum()));
        Self {
            is_blue: config.is_blue,
            opp_goal_x_sign: opp_goal_x_sign.clone(),
            own_players_tracker: vec![None; MAX_PLAYERS],
            opp_players_tracker: vec![None; MAX_PLAYERS],
            ball_tracker: BallTracker::new(opp_goal_x_sign),
            field_geometry: None,
        }
    }

    /// Update the sign of the enemy goal's x coordinate (in ssl-vision coordinates).
    pub fn set_opp_goal_x_sign(&self, sign: f32) {
        self.opp_goal_x_sign.set(sign.signum());
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
                    blue_trackers[id as usize] =
                        Some(PlayerTracker::new(id, self.opp_goal_x_sign.clone()));
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
                    yellow_tracker[id as usize] =
                        Some(PlayerTracker::new(id, self.opp_goal_x_sign.clone()));
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
    /// The world state is initialized if all players and the ball have been seen at
    /// least twice (so that velocities can be calculated), and the field geometry has
    /// been received.
    pub fn is_init(&self) -> bool {
        let own_players_init = 
            // Check whether we have any players at all
            if self
                .own_players_tracker
                .iter()
                .any(|tracker| tracker.is_some())
            {
                // Check whether the players we have are initialized
                self.own_players_tracker
                    .iter()
                    .all(|tracker| tracker.as_ref().map(|t| t.is_init()).unwrap_or(true))
            } else {
                false
            };

        let opp_players_init =
            // Check whether we have any players at all
            if self
                .opp_players_tracker
                .iter()
                .any(|tracker| tracker.is_some())
            {
                // Check whether the players we have are initialized
                self.opp_players_tracker
                    .iter()
                    .all(|tracker| tracker.as_ref().map(|t| t.is_init()).unwrap_or(true))
            } else {
                false
            };

        let ball_init = self.ball_tracker.is_init();
        let field_geom_init = self.field_geometry.is_some();

        own_players_init && opp_players_init && ball_init && field_geom_init
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

        let mut own_players = Vec::with_capacity(11);
        for player_tracker in self.own_players_tracker.iter() {
            if let Some(player_data) = player_tracker.as_ref().and_then(|t| t.get()) {
                own_players.push(player_data);
            } else {
                log::warn!("Tried to get world state before all own players were initialized");
                return None;
            }
        }

        let mut opp_players = Vec::with_capacity(11);
        for player_tracker in self.opp_players_tracker.iter() {
            if let Some(player_data) = player_tracker.as_ref().and_then(|t| t.get()) {
                opp_players.push(player_data);
            } else {
                log::warn!("Tried to get world state before all opp players were initialized");
                return None;
            }
        }

        let ball = if let Some(ball_data) = self.ball_tracker.get() {
            ball_data
        } else {
            log::warn!("Tried to get world state before ball was initialized");
            return None;
        };

        Some(WorldData {
            own_players,
            opp_players,
            ball,
            field_geom,
        })
    }
}

#[cfg(test)]
mod test {
    use dies_protos::ssl_vision_detection::{SSL_DetectionBall, SSL_DetectionFrame};

    use super::*;

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
    fn test_ball_only_update() {
        let mut tracker = WorldTracker::new(WorldConfig {
            is_blue: true,
            initial_opp_goal_x: 1.0,
        });

        let mut frame = SSL_WrapperPacket::new();
        let mut detection = SSL_DetectionFrame::new();
        detection.set_t_capture(0.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(1.0);
        ball.set_y(2.0);
        ball.set_z(3.0);
        detection.balls.push(ball.clone());
        frame.detection.as_mut().replace(&mut detection);

        tracker.update_from_protobuf(&frame);
        assert!(!tracker.is_init());
        assert!(tracker.get().is_none());
    }
}
