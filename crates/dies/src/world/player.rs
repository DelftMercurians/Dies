use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

use crate::protos::ssl_detection::SSL_DetectionFrame;

/// A struct to store the player state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PlayerData {
    // Whether the player is friendly -- on our team
    pub is_ours: bool,
    // Position of the player
    pub position: Vector2<f32>,
    // Velocity of the player
    pub velocity: Vector2<f32>,
    // Orientation of the player [-pi, pi]
    pub orientation: f32,
    // Angular speed of the player
    pub angular_speed: f32,
}

// Create tracker types for players and the ball.
#[derive(Serialize, Clone, Debug)]
pub struct PlayerTracker {
    is_ours: bool, // the blue ones are ours
    player_nr: usize,
    last_update_time: Option<f32>,
    last_position: Option<Vector2<f32>>,
    last_orientation: Option<f32>,
}

impl PlayerTracker {
    /// Create a new PlayerTracker.
    pub fn new(is_ours: bool, player_nr: usize) -> PlayerTracker {
        PlayerTracker {
            is_ours,
            player_nr,
            last_update_time: None,
            last_position: None,
            last_orientation: None,
        }
    }

    /// Update the tracker with a new frame.
    pub fn update(&mut self, frame: &SSL_DetectionFrame) -> Option<PlayerData> {
        let player_detection = if self.is_ours {
            frame.robots_blue.get(self.player_nr)
        } else {
            frame.robots_yellow.get(self.player_nr)
        }?;

        let current_time = frame.t_capture() as f32;
        // let current_position =
        //     Vector3::new(ball_detection.x(), ball_detection.y(), ball_detection.z()) / 1000.0;
        // QUESTION is Vector2 ok?
        // QUESTION is the /1000.0 ok?
        let current_position = Vector2::new(player_detection.x(), player_detection.y()) / 1000.0;

        // Compute the velocity of the ball.
        let velocity = if let (Some(last_position), Some(last_update_time)) =
            (self.last_position, self.last_update_time)
        {
            (current_position - last_position) / (current_time - last_update_time)
        } else {
            Vector2::zeros()
        };

        // Compute the orientation of the player.
        let orientation = player_detection.orientation();

        // Compute the angular speed of the player.
        let angular_speed = if let (Some(last_orientation), Some(last_update_time)) =
            (self.last_orientation, self.last_update_time)
        {
            (orientation - last_orientation) / (current_time - last_update_time)
        } else {
            0.0
        };

        // Update the internal state of the tracker.
        self.last_update_time = Some(current_time);
        self.last_position = Some(current_position.clone());

        // Construct and return a PlayerData instance.
        // QUESTION where should the flip happen?
        Some(PlayerData {
            is_ours: self.is_ours,
            position: current_position,
            velocity,
            orientation,
            angular_speed,
        })
    }
}

// QUESTION errors
#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::ssl_detection::{SSL_DetectionFrame, SSL_DetectionRobot};

    #[test]
    fn test_update_no_player() {
        let mut tracker = PlayerTracker::new(true, 0);
        let frame = SSL_DetectionFrame::new();

        let player_data = tracker.update(&frame);
        assert!(player_data.is_none());
    }

    #[test]
    fn test_update() {
        let mut tracker = PlayerTracker::new(true, 0);

        // 1st update
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(0.0);
        let mut robot = SSL_DetectionRobot::new();
        robot.set_x(1.0);
        robot.set_y(2.0);
        robot.set_orientation(0.0);
        frame.robots_blue.push(robot.clone());

        let player_data = tracker.update(&frame).unwrap();
        assert_eq!(player_data.position, Vector2::new(1.0, 2.0));
        assert_eq!(player_data.velocity, Vector2::zeros());

        // 2nd update
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(1.0);
        let mut robot = SSL_DetectionRobot::new();
        robot.set_x(2.0);
        robot.set_y(3.0);
        robot.set_orientation(0.0);
        frame.robots_blue.push(robot.clone());

        let player_data = tracker.update(&frame).unwrap();
        assert_eq!(player_data.position, Vector2::new(2.0, 3.0));
        assert_eq!(player_data.velocity, Vector2::new(1.0, 1.0));
    }
}
