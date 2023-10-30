use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use crate::protos::ssl_detection::SSL_DetectionFrame;

/// A struct to store the ball state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BallData {
    // Position of the ball
    pub position: Vector3<f32>,
    // Velocity of the ball
    pub velocity: Vector3<f32>,
}

/// Tracker for the ball.
#[derive(Serialize, Clone, Debug)]
pub struct BallTracker {
    last_update_time: Option<f32>,
    last_position: Option<Vector3<f32>>,
}

impl BallTracker {
    /// Create a new BallTracker.
    pub fn new() -> BallTracker {
        BallTracker {
            last_update_time: None,
            last_position: None,
        }
    }

    /// Update the tracker with a new frame.
    pub fn update(&mut self, frame: &SSL_DetectionFrame) -> Option<BallData> {
        let ball_detection = frame.balls.get(0)?;
        let current_time = frame.t_capture() as f32;
        let current_position =
            Vector3::new(ball_detection.x(), ball_detection.y(), ball_detection.z()) / 1000.0;

        // Compute the velocity of the ball.
        let velocity = if let (Some(last_position), Some(last_update_time)) =
            (self.last_position, self.last_update_time)
        {
            (current_position - last_position) / (current_time - last_update_time)
        } else {
            Vector3::zeros()
        };

        // Update the internal state of the tracker.
        self.last_update_time = Some(current_time);
        self.last_position = Some(current_position.clone());

        // Construct and return a BallData instance.
        Some(BallData {
            position: current_position,
            velocity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::ssl_detection::{SSL_DetectionBall, SSL_DetectionFrame};

    #[test]
    fn test_update_no_ball() {
        let mut tracker = BallTracker::new();
        let frame = SSL_DetectionFrame::new();

        let ball_data = tracker.update(&frame);
        assert!(ball_data.is_none());
    }

    #[test]
    fn test_update() {
        let mut tracker = BallTracker::new();

        // 1st update
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(0.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(1.0);
        ball.set_y(2.0);
        ball.set_z(3.0);
        frame.balls.push(ball.clone());

        let ball_data = tracker.update(&frame).unwrap();
        assert_eq!(ball_data.position, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(ball_data.velocity, Vector3::zeros());

        // 2nd update
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(1.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(2.0);
        ball.set_y(4.0);
        ball.set_z(6.0);
        frame.balls.push(ball.clone());

        let ball_data = tracker.update(&frame).unwrap();
        assert_eq!(ball_data.position, Vector3::new(2.0, 4.0, 6.0));
        assert_eq!(ball_data.velocity, Vector3::new(1.0, 2.0, 3.0));
    }
}
