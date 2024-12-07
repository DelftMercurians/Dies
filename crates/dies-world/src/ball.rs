use dies_core::{debug_line, debug_value, Vector3};
use dies_protos::ssl_vision_detection::SSL_DetectionFrame;
use nalgebra::{SVector, Vector6};

use crate::{filter::MaybeKalman, geom::FieldGeometry, BallData, FieldMask, TrackerSettings};

/// Stored data for the ball from the last update.
#[derive(Debug)]
struct StoredData {
    timestamp: f64,
    raw_position: Vec<Vector3>,
    position: Vector3,
    velocity: Vector3,
}

/// Tracker for the ball.
#[derive(Debug)]
pub struct BallTracker {
    /// Kalman filter for the ball's position and velocity
    filter: MaybeKalman<3, 6>,

    /// Result of the last vision update
    last_detection: Option<StoredData>,

    misses: u32,
}

impl BallTracker {
    /// Create a new BallTracker.
    pub fn new(settings: &TrackerSettings) -> BallTracker {
        BallTracker {
            last_detection: None,
            filter: MaybeKalman::new(
                0.1,
                settings.ball_unit_transition_var,
                settings.ball_measurement_var,
            ),
            misses: 0,
        }
    }

    pub fn update_settings(&mut self, settings: &TrackerSettings) {
        if let Some(filter) = self.filter.as_mut() {
            filter.update_settings(
                settings.ball_unit_transition_var,
                settings.ball_measurement_var,
            );
        }
    }

    /// Whether the tracker has been initialized (i.e. the ball has been detected at least twice)
    pub fn is_init(&self) -> bool {
        self.last_detection.is_some()
    }

    /// Update the tracker with a new frame.
    pub fn update(
        &mut self,
        frame: &SSL_DetectionFrame,
        field_mask: &FieldMask,
        field_geom: Option<&FieldGeometry>,
    ) {
        // Log confidence values for debugging
        frame
            .balls
            .iter()
            .for_each(|b| debug_value("ball.confidence", b.confidence() as f64));

        // Filter and collect valid ball measurements
        let mut ball_measurements = frame
            .balls
            .iter()
            .filter(|ball| field_mask.contains(ball.x(), ball.y(), field_geom))
            .filter(|ball| !ball.has_confidence() || ball.confidence() > 0.6)
            .map(|ball| {
                (
                    Vector3::new(ball.x() as f64, ball.y() as f64, ball.z() as f64),
                    !ball.has_confidence() || ball.confidence.unwrap() < 0.9,
                )
            })
            .collect::<Vec<(Vector3, bool)>>();

        // Store raw positions for visualization
        let measured_positions = ball_measurements
            .iter()
            .map(|(pos, _)| pos)
            .cloned()
            .collect::<Vec<_>>();

        // Handle case with no valid measurements
        if ball_measurements.is_empty() {
            self.misses += 1;
            return;
        }
        self.misses = 0;

        // Sort measurements by confidence (less noisy ones first)
        ball_measurements.sort_by(|a, b| b.1.cmp(&a.1));
        let current_time = frame.t_capture();

        // Initialize tracker with first measurement if needed
        if self.last_detection.is_none() {
            self.last_detection = Some(StoredData {
                timestamp: current_time,
                position: ball_measurements[0].0,
                raw_position: measured_positions,
                velocity: Vector3::zeros(),
            });

            self.filter.init(
                Vector6::new(
                    ball_measurements[0].0.x,
                    0.0,
                    ball_measurements[0].0.y,
                    0.0,
                    ball_measurements[0].0.z,
                    0.0,
                ),
                current_time,
            );
            return;
        }

        // Update state with each measurement
        for (pos, _is_noisy) in ball_measurements.iter() {
            let pos_ov = SVector::<f64, 3>::new(pos.x, pos.y, pos.z);
            if let Some(z) = self
                .filter
                .as_mut()
                .unwrap()
                .update(pos_ov, current_time, false)
            {
                // Extract position and velocity from filter state
                let mut pos_v3 = Vector3::new(z[0], z[2], z[4]);
                let vel_v3 = Vector3::new(z[1], z[3], z[5]);

                // Handle ball bouncing by reflecting velocity
                if pos_v3.z < 0.0 {
                    pos_v3.z = 0.0;
                    self.filter.as_mut().unwrap().set_x(SVector::<f64, 6>::new(
                        pos_v3.x, vel_v3.x, pos_v3.y, vel_v3.y, pos_v3.z, -vel_v3.z,
                    ));
                }

                // Debug visualization
                debug_value("ball.speed", vel_v3.xy().norm());
                debug_line(
                    "ball.vel",
                    pos_v3.xy(),
                    pos_v3.xy() + vel_v3.xy(),
                    dies_core::DebugColor::Orange,
                );

                // Store updated state
                self.last_detection = Some(StoredData {
                    timestamp: current_time,
                    position: pos_v3,
                    raw_position: Vec::with_capacity(0),
                    velocity: vel_v3,
                });
            }
        }

        // Update raw positions for visualization
        if let Some(detection) = &mut self.last_detection {
            detection.raw_position = measured_positions;
        }
    }

    /// Get the current ball state
    pub fn get(&self) -> Option<BallData> {
        self.last_detection.as_ref().map(|data| BallData {
            timestamp: data.timestamp,
            raw_position: data.raw_position.clone(),
            position: data.position,
            velocity: data.velocity,
            detected: self.misses < 5,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_core::Vector2;
    use dies_protos::ssl_vision_detection::{SSL_DetectionBall, SSL_DetectionFrame};

    #[test]
    fn test_update_no_ball() {
        let mut tracker = BallTracker::new(&TrackerSettings::default());
        let frame = SSL_DetectionFrame::new();

        tracker.update(
            &frame,
            &FieldMask::default(),
            Some(&FieldGeometry::default()),
        );
        let ball_data = tracker.get();
        assert!(ball_data.is_none());
    }

    #[test]
    fn test_basic_update() {
        let mut tracker = BallTracker::new(&TrackerSettings::default());

        // First update with stationary ball
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(0.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(1.0);
        ball.set_y(2.0);
        ball.set_z(0.0);
        ball.set_confidence(0.99);
        frame.balls.push(ball.clone());

        tracker.update(
            &frame,
            &FieldMask::default(),
            Some(&FieldGeometry::default()),
        );
        assert!(tracker.is_init()); // initialized after single update

        // Second update with same position
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(1.0);
        ball.set_confidence(0.99);
        frame.balls.push(ball);

        tracker.update(
            &frame,
            &FieldMask::default(),
            Some(&FieldGeometry::default()),
        );

        let ball_data = tracker.get().unwrap();
        assert_eq!(ball_data.raw_position[0], Vector3::new(1.0, 2.0, 0.0));
        assert_eq!(ball_data.velocity.xy(), Vector2::zeros()); // Should have zero velocity
    }

    #[test]
    fn test_moving_ball() {
        let mut tracker = BallTracker::new(&TrackerSettings::default());

        // First update
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(0.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(0.0);
        ball.set_y(0.0);
        ball.set_z(0.0);
        ball.set_confidence(0.99);
        frame.balls.push(ball);

        tracker.update(
            &frame,
            &FieldMask::default(),
            Some(&FieldGeometry::default()),
        );

        // Second update - ball moved
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(1.0);
        let mut ball = SSL_DetectionBall::new();
        ball.set_x(1000.0); // 1m movement in 1 second = 1m/s velocity
        ball.set_y(0.0);
        ball.set_z(0.0);
        ball.set_confidence(0.99);
        frame.balls.push(ball);

        tracker.update(
            &frame,
            &FieldMask::default(),
            Some(&FieldGeometry::default()),
        );

        let ball_data = tracker.get().unwrap();
        assert!(ball_data.velocity.x > 0.0); // Should have positive x velocity
        assert_eq!(ball_data.velocity.y, 0.0); // Should have no y velocity
    }
}
