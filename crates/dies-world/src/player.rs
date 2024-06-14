use dies_core::{Angle, PlayerData, PlayerId};
use dies_protos::ssl_vision_detection::SSL_DetectionRobot;
use nalgebra::{self as na, Vector2, Vector4};

use crate::{coord_utils::to_dies_coords2, filter::Kalman};

/// Tracker for a single player.
pub struct PlayerTracker {
    /// Player's unique id
    id: PlayerId,
    /// The sign of the enemy goal's x coordinate in ssl-vision coordinates. Used for
    /// converting coordinates.
    play_dir_x: f64,
    /// Whether the tracker has been initialized (i.e. the player has been detected at
    /// least twice)
    is_init: bool,
    /// Last recorded data (for caching)
    last_data: Option<PlayerData>,
    /// Kalman filter for the player's position and velocity
    filter: Option<Kalman<2, 4>>,
}

impl PlayerTracker {
    /// Create a new PlayerTracker.
    pub fn new(id: PlayerId, initial_play_dir_x: f64) -> PlayerTracker {
        PlayerTracker {
            id,
            play_dir_x: initial_play_dir_x,
            last_data: None,
            is_init: false,
            filter: None,
        }
    }

    /// Set the sign of the enemy goal's x coordinate in ssl-vision coordinates.
    pub fn set_play_dir_x(&mut self, play_dir_x: f64) {
        if play_dir_x != self.play_dir_x {
            // Flip the x coordinate of the player's position, velocity and yaw
            if let Some(last_data) = &mut self.last_data {
                last_data.position.x *= -1.0;
                last_data.velocity.x *= -1.0;
                last_data.yaw = last_data.yaw.inv();
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
        let current_yaw = Angle::from_radians(player.orientation() as f64 * self.play_dir_x);

        if let Some(filter) = &mut self.filter {
            let z = na::convert(Vector2::new(current_position.x, current_position.y));
            if let Some(x) = filter.update(z, t_capture, false) {
                let last_data = if let Some(last_data) = &mut self.last_data {
                    last_data
                } else {
                    self.last_data = Some(PlayerData {
                        timestamp: t_capture,
                        id: self.id,
                        raw_position: current_position,
                        position: na::convert(Vector2::new(x[0], x[2])),
                        velocity: na::convert(Vector2::new(x[1], x[3])),
                        yaw: current_yaw,
                        angular_speed: 0.0,
                    });
                    self.is_init = true;
                    self.last_data.as_mut().unwrap()
                };
                last_data.raw_position = current_position;
                last_data.position = na::convert(Vector2::new(x[0], x[2]));
                last_data.velocity = na::convert(Vector2::new(x[1], x[3]));
                last_data.yaw = current_yaw;
                last_data.angular_speed = (current_yaw - last_data.yaw).as_f64()
                    / ((t_capture - last_data.timestamp + std::f64::EPSILON) as f64);
                last_data.timestamp = t_capture;
            }
        } else {
            self.filter = Some(Kalman::<2, 4>::new_player_filter(
                0.1,
                0.1,
                2.0,
                Vector4::new(
                    current_position.x as f64,
                    0.0,
                    current_position.y as f64,
                    0.0,
                ),
                t_capture,
            ));
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
    use std::f64::consts::PI;

    use super::*;

    #[test]
    fn test_no_player() {
        let tracker = PlayerTracker::new(PlayerId::new(1), 1.0);

        assert!(!tracker.is_init());
        assert!(tracker.get().is_none());
    }

    #[test]
    fn test_no_data_after_one_update() {
        let mut tracker = PlayerTracker::new(PlayerId::new(1), 1.0);

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
        let mut tracker = PlayerTracker::new(PlayerId::new(1), 1.0);

        let mut player = SSL_DetectionRobot::new();
        player.set_x(100.0);
        player.set_y(200.0);
        player.set_orientation(0.0);

        tracker.update(0.0, &player);
        assert!(!tracker.is_init());

        tracker.update(1.0, &player);
        assert!(tracker.is_init());

        let data = tracker.get().unwrap();
        assert_eq!(data.id.as_u32(), 1);
        assert_eq!(data.position, Vector2::new(100.0, 200.0));
        assert_eq!(data.velocity, Vector2::zeros());
        assert_eq!(data.yaw, Angle::from_radians(0.0));
        assert_eq!(data.angular_speed, 0.0);

        player.set_x(200.0);
        player.set_y(300.0);
        player.set_orientation(1.0);

        tracker.update(2.0, &player);
        assert!(tracker.is_init());

        let data = tracker.get().unwrap();
        assert_eq!(data.id.as_u32(), 1);
        assert_eq!(data.position, Vector2::new(200.0, 300.0));
        assert_eq!(data.velocity, Vector2::new(100.0, 100.0));
        assert_eq!(data.yaw, Angle::from_radians(1.0));
        assert_eq!(data.angular_speed, 1.0);
    }

    #[test]
    fn test_x_flip() {
        let mut tracker = PlayerTracker::new(PlayerId::new(1), -1.0);

        let mut player = SSL_DetectionRobot::new();
        let dir = PI / 2.0;
        player.set_x(100.0);
        player.set_y(200.0);
        player.set_orientation(dir as f32);

        tracker.update(0.0, &player);
        assert!(!tracker.is_init());

        tracker.update(1.0, &player);
        assert!(tracker.is_init());

        let data = tracker.get().unwrap();
        assert_eq!(data.id.as_u32(), 1);
        assert_eq!(data.position, Vector2::new(-100.0, 200.0));
        assert_eq!(data.velocity, Vector2::zeros());
        assert_eq!(data.yaw, Angle::from_radians(-dir));
        assert_eq!(data.angular_speed, 0.0);

        tracker.set_play_dir_x(1.0);

        player.set_x(200.0);
        player.set_y(300.0);
        player.set_orientation(-dir as f32);

        tracker.update(2.0, &player);
        assert!(tracker.is_init());

        let data = tracker.get().unwrap();
        assert_eq!(data.id.as_u32(), 1);
        assert_eq!(data.position, Vector2::new(200.0, 300.0));
        assert_eq!(data.velocity, Vector2::new(100.0, 100.0));
        assert_eq!(data.yaw, Angle::from_radians(-dir));
        assert_eq!(data.angular_speed, -PI);
    }
}
