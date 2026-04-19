use dies_core::{debug_value, BallData, FieldGeometry, FieldMask, TrackerSettings, Vector3};
use dies_protos::ssl_vision_detection::SSL_DetectionFrame;
use nalgebra::{SVector, Vector6};

use crate::filter::{MaybeKalman, ParticleFilter, ParticleFilterConfig};

/// Stored data for the ball from the last update.
///
/// This type contains **vision coordinates**, meaning the x axis is unchanged -- it may point towards our or the enemy goal.
///
/// For documentation on the fields, see the `BallData` struct in `dies-core`.
#[derive(Debug)]
struct StoredData {
    timestamp: f64,
    raw_position: Vec<Vector3>,
    position: Vector3,
    velocity: Vector3,
}

/// Tracker for the ball with particle filter integration.
#[derive(Debug)]
pub struct BallTracker {
    /// Kalman filter for the ball's position and velocity (kept for compatibility/hybrid approach)
    filter: MaybeKalman<3, 6>,

    /// Particle filter for robust ball tracking
    particle_filter: ParticleFilter,

    /// Result of the last vision update
    last_detection: Option<StoredData>,

    /// Number of consecutive missed detections
    misses: u32,

    /// Use particle filter for state estimate (if true) or Kalman (if false)
    use_particle_filter: bool,
}

impl BallTracker {
    /// Create a new BallTracker with particle filter enabled.
    pub fn new(settings: &TrackerSettings) -> BallTracker {
        let particle_config = ParticleFilterConfig::default();

        BallTracker {
            last_detection: None,
            filter: MaybeKalman::new(
                0.1,
                settings.ball_unit_transition_var,
                settings.ball_measurement_var,
            ),
            particle_filter: ParticleFilter::new(particle_config),
            misses: 0,
            use_particle_filter: true,
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

    /// Whether the tracker has been initialized (i.e. the ball has been detected at
    /// least twice)
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
        frame
            .balls
            .iter()
            .for_each(|b| debug_value("ball.confidence", b.confidence() as f64));

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
        let measured_positions = ball_measurements
            .iter()
            .map(|(pos, _)| pos)
            .cloned()
            .collect::<Vec<_>>();

        let current_time = frame.t_capture();

        log::debug!(
            "BallTracker::update - ball_measurements: count={}, measured_positions: {:?}",
            ball_measurements.len(),
            measured_positions
        );

        if ball_measurements.is_empty() {
            self.misses += 1;

            // Continue prediction with particle filter even without measurements
            if self.particle_filter.is_initialized() && self.misses <= 5 {
                let dt = if let Some(last) = &self.last_detection {
                    current_time - last.timestamp
                } else {
                    1.0 / 60.0 // Default to 60 FPS if no previous detection
                };

                self.particle_filter.predict(dt);

                if let Some(pos) = self.particle_filter.get_mean_estimate() {
                    if let Some(vel) = self.particle_filter.get_mean_velocity() {
                        self.last_detection = Some(StoredData {
                            timestamp: current_time,
                            position: pos,
                            raw_position: measured_positions,
                            velocity: vel,
                        });
                    }
                }
            }
            return;
        }

        self.misses = 0;
        ball_measurements.sort_by(|a, b| b.1.cmp(&a.1));

        // If first detection or if we've lost the ball for too long, re-initialize filters
        let should_reinitialize = self.last_detection.is_none() || self.misses > 5;
        
        if should_reinitialize {
            let initial_pos = ball_measurements[0].0;
            let initial_vel = Vector3::zeros();

            self.particle_filter
                .initialize_from_measurement(initial_pos, initial_vel);

            self.filter.init(
                Vector6::new(initial_pos.x, 0.0, initial_pos.y, 0.0, initial_pos.z, 0.0),
                current_time,
            );

            self.last_detection = Some(StoredData {
                timestamp: current_time,
                position: initial_pos,
                raw_position: measured_positions,
                velocity: initial_vel,
            });
        } else {
            // Predict step with particle filter
            let dt = current_time - self.last_detection.as_ref().unwrap().timestamp;
            if dt > 0.0 {
                self.particle_filter.predict(dt);
            }

            // Update particle filter with measurements
            self.particle_filter.update(&measured_positions);

            // Also update Kalman filter for comparison/fallback
            for (pos, _is_noisy) in ball_measurements.iter() {
                let pos_ov = SVector::<f64, 3>::new(pos.x, pos.y, pos.z);
                let z = self
                    .filter
                    .as_mut()
                    .unwrap()
                    .update(pos_ov, current_time, false);
                if let Some(_) = z {
                    // Kalman update successful
                }
            }

            // Use particle filter for position/velocity estimate
            if self.use_particle_filter {
                if let Some(pos) = self.particle_filter.get_mean_estimate() {
                    if let Some(vel) = self.particle_filter.get_mean_velocity() {
                        let mut pos_v3 = pos;
                        let vel_v3 = vel;

                        // Clamp z to ground
                        if pos_v3.z < 0.0 {
                            pos_v3.z = 0.0;
                        }

                        log::debug!(
                            "BallTracker::update - particle filter: pos={:?}, vel={:?}, raw_pos_count={}",
                            pos_v3,
                            vel_v3,
                            measured_positions.len()
                        );

                        self.last_detection = Some(StoredData {
                            timestamp: current_time,
                            position: pos_v3,
                            raw_position: measured_positions,
                            velocity: vel_v3,
                        });
                    }
                }
            } else {
                // Fallback: use Kalman filter estimate
                if let Some(z) = self
                    .filter
                    .as_mut()
                    .unwrap()
                    .update(SVector::<f64, 3>::new(ball_measurements[0].0.x, ball_measurements[0].0.y, ball_measurements[0].0.z), current_time, false)
                {
                    let mut pos_v3 = Vector3::new(z[0], z[2], z[4]);
                    let vel_v3 = Vector3::new(z[1], z[3], z[5]);
                    if pos_v3.z < 0.0 {
                        pos_v3.z = 0.0;
                        self.filter.as_mut().unwrap().set_x(SVector::<f64, 6>::new(
                            pos_v3.x, vel_v3.x, pos_v3.y, vel_v3.y, pos_v3.z, -vel_v3.z,
                        ));
                    }
                    self.last_detection = Some(StoredData {
                        timestamp: current_time,
                        position: pos_v3,
                        raw_position: Vec::with_capacity(0),
                        velocity: vel_v3,
                    });
                }
            }
        }

        if let Some(last) = &self.last_detection {
            dies_core::debug_value("ball_speed", last.velocity.xy().norm());
        }
    }

    pub fn get(&self) -> Option<BallData> {
        self.last_detection.as_ref().map(|data| {
            log::debug!(
                "BallTracker::get - raw_position: len={}, pos={:?}, vel={:?}, detected={}, misses={}",
                data.raw_position.len(),
                data.position,
                data.velocity,
                self.misses < 5,
                self.misses
            );
            BallData {
                timestamp: data.timestamp,
                raw_position: data.raw_position.clone(),
                position: data.position,
                velocity: data.velocity,
                detected: self.misses < 5,
            }
        })
    }
}

// TOOD: FIX
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use dies_protos::ssl_vision_detection::{SSL_DetectionBall, SSL_DetectionFrame};

//     #[test]
//     fn test_update_no_ball() {
//         let mut tracker = BallTracker::new(&TrackerSettings::default());
//         let frame = SSL_DetectionFrame::new();

//         tracker.update(&frame);
//         let ball_data = tracker.get();
//         assert!(ball_data.is_none());
//     }

//     #[test]
//     fn test_basic_update() {
//         let mut tracker = BallTracker::new(&TrackerSettings::default());

//         // 1st update
//         let mut frame = SSL_DetectionFrame::new();
//         frame.set_t_capture(0.0);
//         let mut ball = SSL_DetectionBall::new();
//         ball.set_x(1.0);
//         ball.set_y(2.0);
//         ball.set_z(3.0);
//         frame.balls.push(ball.clone());

//         tracker.update(&frame);

//         // 2nd update
//         let mut frame = SSL_DetectionFrame::new();
//         frame.set_t_capture(1.0);
//         let mut ball = SSL_DetectionBall::new();
//         ball.set_x(2.0);
//         ball.set_y(4.0);
//         ball.set_z(6.0);
//         frame.balls.push(ball.clone());

//         tracker.update(&frame);
//         let ball_data = tracker.get().unwrap();
//         assert_eq!(ball_data.raw_position[0], Vector3::new(2.0, 4.0, 6.0));
//         assert!(ball_data.velocity.norm() > 0.0)
//     }

//     #[test]
//     fn test_x_flip() {
//         let settings = TrackerSettings {
//             initial_opp_goal_x: -1.0,
//             ..Default::default()
//         };
//         let mut tracker = BallTracker::new(&settings);

//         // 1st update
//         let mut frame = SSL_DetectionFrame::new();
//         frame.set_t_capture(0.0);
//         let mut ball = SSL_DetectionBall::new();
//         ball.set_x(1.0);
//         ball.set_y(2.0);
//         ball.set_z(3.0);
//         frame.balls.push(ball.clone());

//         tracker.update(&frame);

//         // 2nd update
//         let mut frame = SSL_DetectionFrame::new();
//         frame.set_t_capture(1.0);
//         let mut ball = SSL_DetectionBall::new();
//         ball.set_x(2.0);
//         ball.set_y(4.0);
//         ball.set_z(6.0);
//         frame.balls.push(ball.clone());

//         tracker.update(&frame);
//         let ball_data = tracker.get().unwrap();
//         assert_eq!(ball_data.raw_position[0], Vector3::new(-2.0, 4.0, 6.0));
//         assert!(ball_data.velocity.x < 0.0)
//     }
// }
