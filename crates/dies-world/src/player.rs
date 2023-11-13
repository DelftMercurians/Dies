use std::{cell::Cell, rc::Rc};

use dies_protos::ssl_vision_detection::SSL_DetectionRobot;
use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

use crate::coord_utils::to_dies_coords2;

/// A struct to store the player state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PlayerData {
    /// Unix timestamp of the recorded frame from which this data was extracted (in
    /// seconds). This is the time that ssl-vision received the frame.
    pub timestamp: f64,
    /// The player's unique id
    pub id: u32,
    /// Position of the player in mm, in dies coordinates
    pub position: Vector2<f32>,
    /// Velocity of the player in mm/s, in dies coordinates
    pub velocity: Vector2<f32>,
    /// Orientation of the player [-pi, pi]
    pub orientation: f32,
    /// Angular speed of the player (in rad/s)
    pub angular_speed: f32,
}

/// Tracker for a single player.
#[derive(Clone, Debug)]
pub struct PlayerTracker {
    /// Player's unique id
    id: u32,
    /// The sign of the enemy goal's x coordinate in ssl-vision coordinates. Used for
    /// converting coordinates.
    opp_goal_x_sign: Rc<Cell<f32>>,
    /// Whether the tracker has been initialized (i.e. the player has been detected at
    /// least twice)
    is_init: bool,
    /// Last recorded data (for caching)
    last_data: Option<PlayerData>,
}

impl PlayerTracker {
    pub fn new(id: u32, opp_goal_x_sign: Rc<Cell<f32>>) -> PlayerTracker {
        PlayerTracker {
            id,
            opp_goal_x_sign,
            last_data: None,
            is_init: false,
        }
    }

    pub fn is_init(&self) -> bool {
        self.is_init
    }

    /// Update the tracker with a new frame.
    pub fn update(&mut self, t_capture: f64, player: &SSL_DetectionRobot) {
        let current_position = to_dies_coords2(player.x(), player.y(), self.opp_goal_x_sign.get());
        let current_orientation = player.orientation();

        if let Some(last_data) = &self.last_data {
            let last_update_time = last_data.timestamp;
            let last_pos = last_data.position;
            let last_orientation = last_data.orientation;
            let dt = (t_capture - last_update_time) as f32;
            let velocity = (current_position - last_pos) / dt;
            let omega = (current_orientation - last_orientation) / dt;

            self.last_data = Some(PlayerData {
                timestamp: t_capture,
                id: self.id,
                position: current_position,
                velocity,
                orientation: current_orientation,
                angular_speed: omega,
            });

            self.is_init = true;
        } else {
            self.last_data = Some(PlayerData {
                timestamp: t_capture,
                id: self.id,
                position: current_position,
                velocity: Vector2::zeros(),
                orientation: current_orientation,
                angular_speed: 0.0,
            });
        }
    }

    pub fn get(&self) -> Option<&PlayerData> {
        if self.is_init {
            self.last_data.as_ref()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_no_player() {
        let opp_goal_x_sign = Rc::new(Cell::new(1.0));
        let tracker = PlayerTracker::new(1, opp_goal_x_sign.clone());

        assert!(!tracker.is_init());
        assert!(tracker.get().is_none());
    }

    #[test]
    fn test_no_data_after_one_update() {
        let opp_goal_x_sign = Rc::new(Cell::new(1.0));
        let mut tracker = PlayerTracker::new(1, opp_goal_x_sign.clone());

        let mut player = SSL_DetectionRobot::new();
        player.set_x(100.0);
        player.set_y(200.0);
        player.set_orientation(0.0);

        tracker.update(0.0, &player);
        assert!(!tracker.is_init());
        assert!(tracker.get().is_none());
    }

    #[test]
    fn test_basic_update() {
        let opp_goal_x_sign = Rc::new(Cell::new(1.0));
        let mut tracker = PlayerTracker::new(1, opp_goal_x_sign.clone());

        let mut player = SSL_DetectionRobot::new();
        player.set_x(100.0);
        player.set_y(200.0);
        player.set_orientation(0.0);

        tracker.update(0.0, &player);
        assert!(!tracker.is_init());

        tracker.update(1.0, &player);
        assert!(tracker.is_init());

        let data = tracker.get().unwrap();
        assert_eq!(data.id, 1);
        assert_eq!(data.position, Vector2::new(100.0, 200.0));
        assert_eq!(data.velocity, Vector2::zeros());
        assert_eq!(data.orientation, 0.0);
        assert_eq!(data.angular_speed, 0.0);

        player.set_x(200.0);
        player.set_y(300.0);
        player.set_orientation(1.0);

        tracker.update(2.0, &player);
        assert!(tracker.is_init());

        let data = tracker.get().unwrap();
        assert_eq!(data.id, 1);
        assert_eq!(data.position, Vector2::new(200.0, 300.0));
        assert_eq!(data.velocity, Vector2::new(100.0, 100.0));
        assert_eq!(data.orientation, 1.0);
        assert_eq!(data.angular_speed, 1.0);
    }

    #[test]
    fn test_x_flip() {
        let opp_goal_x_sign = Rc::new(Cell::new(-1.0));
        let mut tracker = PlayerTracker::new(1, opp_goal_x_sign.clone());

        let mut player = SSL_DetectionRobot::new();
        player.set_x(100.0);
        player.set_y(200.0);
        player.set_orientation(0.0);

        tracker.update(0.0, &player);
        assert!(!tracker.is_init());

        tracker.update(1.0, &player);
        assert!(tracker.is_init());

        let data = tracker.get().unwrap();
        assert_eq!(data.id, 1);
        assert_eq!(data.position, Vector2::new(-100.0, 200.0));
        assert_eq!(data.velocity, Vector2::zeros());
        assert_eq!(data.orientation, 0.0);
        assert_eq!(data.angular_speed, 0.0);

        player.set_x(200.0);
        player.set_y(300.0);
        player.set_orientation(1.0);

        tracker.update(2.0, &player);
        assert!(tracker.is_init());

        let data = tracker.get().unwrap();
        assert_eq!(data.id, 1);
        assert_eq!(data.position, Vector2::new(-200.0, 300.0));
        assert_eq!(data.velocity, Vector2::new(-100.0, 100.0));
        assert_eq!(data.orientation, 1.0);
        assert_eq!(data.angular_speed, 1.0);
    }
}
