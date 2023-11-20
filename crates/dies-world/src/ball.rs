use dies_core::BallData;
use nalgebra::Vector3;

use dies_protos::ssl_vision_detection::SSL_DetectionFrame;

use crate::coord_utils::to_dies_coords3;

/// Tracker for the ball.
#[derive(Debug)]
pub struct BallTracker {
    /// The sign of the enemy goal's x coordinate in ssl-vision coordinates. Used for
    /// converting coordinates.
    play_dir_x: f32,
    /// Whether the tracker has been initialized (i.e. the ball has been detected at
    /// least twice)
    is_init: bool,
    /// Last recorded data (for caching)
    last_data: Option<BallData>,
}

impl BallTracker {
    /// Create a new BallTracker.
    pub fn new(play_dir_x: f32) -> BallTracker {
        BallTracker {
            play_dir_x,
            is_init: false,
            last_data: None,
        }
    }

    /// Set the sign of the enemy goal's x coordinate in ssl-vision coordinates.
    pub fn set_play_dir_x(&mut self, play_dir_x: f32) {
        if play_dir_x != self.play_dir_x {
            // Flip the x coordinate of the ball's position and velocity
            if let Some(last_data) = &mut self.last_data {
                last_data.position.x *= -1.0;
                last_data.velocity.x *= -1.0;
            }
            self.play_dir_x = play_dir_x;
        }
    }

    /// Whether the tracker has been initialized (i.e. the ball has been detected at
    /// least twice)
    pub fn is_init(&self) -> bool {
        self.is_init
    }

    /// Update the tracker with a new frame.
    pub fn update(&mut self, frame: &SSL_DetectionFrame) {
        let ball_detection = frame.balls.get(0);
        if let Some(ball_detection) = ball_detection {
            let current_time = frame.t_capture();
            let current_position = to_dies_coords3(
                ball_detection.x(),
                ball_detection.y(),
                ball_detection.z(),
                self.play_dir_x,
            );

            if let Some(last_data) = &self.last_data {
                let last_position = last_data.position;
                let last_time = last_data.timestamp;
                let dt = (current_time - last_time) as f32;
                let velocity = (current_position - last_position) / dt;

                self.last_data = Some(BallData {
                    timestamp: current_time,
                    position: current_position,
                    velocity,
                });
                self.is_init = true;
            } else {
                log::debug!("Ball tracker received first data");
                self.last_data = Some(BallData {
                    timestamp: current_time,
                    position: current_position,
                    velocity: Vector3::zeros(),
                });
            };
        }
    }

    pub fn get(&self) -> Option<&BallData> {
        if self.is_init {
            self.last_data.as_ref()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_protos::ssl_vision_detection::{SSL_DetectionBall, SSL_DetectionFrame};

    #[test]
    fn test_update_no_ball() {
        let mut tracker = BallTracker::new(1.0);
        let frame = SSL_DetectionFrame::new();

        tracker.update(&frame);
        let ball_data = tracker.get();
        assert!(ball_data.is_none());
    }

    #[test]
    fn test_no_data_after_first_update() {
        let mut tracker = BallTracker::new(1.0);

        // 1st update
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(0.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(1.0);
        ball.set_y(2.0);
        ball.set_z(3.0);
        frame.balls.push(ball.clone());

        tracker.update(&frame);
        let ball_data = tracker.get();
        assert!(ball_data.is_none());
    }

    #[test]
    fn test_basic_update() {
        let mut tracker = BallTracker::new(1.0);

        // 1st update
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(0.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(1.0);
        ball.set_y(2.0);
        ball.set_z(3.0);
        frame.balls.push(ball.clone());

        tracker.update(&frame);

        // 2nd update
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(1.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(2.0);
        ball.set_y(4.0);
        ball.set_z(6.0);
        frame.balls.push(ball.clone());

        tracker.update(&frame);
        let ball_data = tracker.get().unwrap();
        assert_eq!(ball_data.position, Vector3::new(2.0, 4.0, 6.0));
        assert_eq!(ball_data.velocity, Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_x_flip() {
        let mut tracker = BallTracker::new(-1.0);

        // 1st update
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(0.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(1.0);
        ball.set_y(2.0);
        ball.set_z(3.0);
        frame.balls.push(ball.clone());

        tracker.update(&frame);

        // 2nd update
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(1.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(2.0);
        ball.set_y(4.0);
        ball.set_z(6.0);
        frame.balls.push(ball.clone());

        tracker.update(&frame);
        let ball_data = tracker.get().unwrap();
        assert_eq!(ball_data.position, Vector3::new(-2.0, 4.0, 6.0));
        assert_eq!(ball_data.velocity, Vector3::new(-1.0, 2.0, 3.0));
    }
}
