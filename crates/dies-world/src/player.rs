use dies_core::PlayerData;
use dies_protos::ssl_vision_detection::SSL_DetectionRobot;
use nalgebra::Vector2;

use crate::coord_utils::to_dies_coords2;

/// Tracker for a single player.
#[derive(Clone, Debug)]
pub struct PlayerTracker {
    /// Player's unique id
    id: u32,
    /// The sign of the enemy goal's x coordinate in ssl-vision coordinates. Used for
    /// converting coordinates.
    play_dir_x: f32,
    /// Whether the tracker has been initialized (i.e. the player has been detected at
    /// least twice)
    is_init: bool,
    /// Last recorded data (for caching)
    last_data: Option<PlayerData>,
}

impl PlayerTracker {
    /// Create a new PlayerTracker.
    pub fn new(id: u32, initial_play_dir_x: f32) -> PlayerTracker {
        PlayerTracker {
            id,
            play_dir_x: initial_play_dir_x,
            last_data: None,
            is_init: false,
        }
    }

    /// Set the sign of the enemy goal's x coordinate in ssl-vision coordinates.
    pub fn set_play_dir_x(&mut self, play_dir_x: f32) {
        if play_dir_x != self.play_dir_x {
            // Flip the x coordinate of the player's position, velocity and orientation
            if let Some(last_data) = &mut self.last_data {
                last_data.position.x *= -1.0;
                last_data.velocity.x *= -1.0;
                last_data.orientation *= -1.0;
            }
        }
        self.play_dir_x = play_dir_x;
    }

    /// Whether the tracker has been initialized (i.e. the player has been detected at
    /// least twice)
    pub fn is_init(&self) -> bool {
        self.is_init
    }

    /// Update the tracker with a new frame.
    pub fn update(&mut self, t_capture: f64, player: &SSL_DetectionRobot) {
        let current_position = to_dies_coords2(player.x(), player.y(), self.play_dir_x);
        let current_orientation = player.orientation() * self.play_dir_x;

        if let Some(last_data) = &self.last_data {
            let last_update_time = last_data.timestamp;
            let last_pos = last_data.position;
            let last_orientation = last_data.orientation;
            let dt = (t_capture - last_update_time) as f32;
            if dt > f32::EPSILON {
                let velocity = (current_position - last_pos) / (dt + f32::EPSILON);
                let omega = (current_orientation - last_orientation) / (dt + f32::EPSILON);

                self.last_data = Some(PlayerData {
                    timestamp: t_capture,
                    id: self.id,
                    position: current_position,
                    velocity,
                    orientation: current_orientation,
                    angular_speed: omega,
                });
                self.is_init = true;
            }
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
    use std::f32::consts::PI;

    use super::*;

    #[test]
    fn test_no_player() {
        let tracker = PlayerTracker::new(1, 1.0);

        assert!(!tracker.is_init());
        assert!(tracker.get().is_none());
    }

    #[test]
    fn test_no_data_after_one_update() {
        let mut tracker = PlayerTracker::new(1, 1.0);

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
        let mut tracker = PlayerTracker::new(1, 1.0);

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
        let mut tracker = PlayerTracker::new(1, -1.0);

        let mut player = SSL_DetectionRobot::new();
        let dir = PI / 2.0;
        player.set_x(100.0);
        player.set_y(200.0);
        player.set_orientation(dir);

        tracker.update(0.0, &player);
        assert!(!tracker.is_init());

        tracker.update(1.0, &player);
        assert!(tracker.is_init());

        let data = tracker.get().unwrap();
        assert_eq!(data.id, 1);
        assert_eq!(data.position, Vector2::new(-100.0, 200.0));
        assert_eq!(data.velocity, Vector2::zeros());
        assert_eq!(data.orientation, -dir);
        assert_eq!(data.angular_speed, 0.0);

        tracker.set_play_dir_x(1.0);

        player.set_x(200.0);
        player.set_y(300.0);
        player.set_orientation(-dir);

        tracker.update(2.0, &player);
        assert!(tracker.is_init());

        let data = tracker.get().unwrap();
        assert_eq!(data.id, 1);
        assert_eq!(data.position, Vector2::new(200.0, 300.0));
        assert_eq!(data.velocity, Vector2::new(100.0, 100.0));
        assert_eq!(data.orientation, -dir);
        assert_eq!(data.angular_speed, -PI);
    }
}
