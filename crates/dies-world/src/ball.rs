use dies_core::{debug_line, debug_value, BallData, TrackerSettings, Vector3};
use nalgebra::{SVector, Vector6};

use dies_protos::ssl_vision_detection::SSL_DetectionFrame;

use crate::coord_utils::to_dies_coords3;
use crate::filter::MaybeKalman;

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

/// Tracker for the ball.
#[derive(Debug)]
pub struct BallTracker {
    /// The sign of the enemy goal's x coordinate in ssl-vision coordinates. Used for
    /// converting coordinates.
    play_dir_x: f64,

    /// Kalman filter for the ball's position and velocity
    filter: MaybeKalman<3, 6>,

    /// Result of the last vision update
    last_detection: Option<StoredData>,
}

impl BallTracker {
    /// Create a new BallTracker.
    pub fn new(settings: &TrackerSettings) -> BallTracker {
        BallTracker {
            play_dir_x: settings.initial_opp_goal_x,
            last_detection: None,
            filter: MaybeKalman::new(
                0.1,
                settings.ball_unit_transition_var,
                settings.ball_measurement_var,
            ),
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

    /// Set the sign of the enemy goal's x coordinate in ssl-vision coordinates.
    pub fn set_play_dir_x(&mut self, play_dir_x: f64) {
        if play_dir_x != self.play_dir_x {
            // Flip the x coordinate of the ball's position and velocity
            if let Some(last_data) = &mut self.last_detection {
                last_data.position.x *= -1.0;
                last_data.velocity.x *= -1.0;
            }
            self.play_dir_x = play_dir_x;
        }
    }

    /// Whether the tracker has been initialized (i.e. the ball has been detected at
    /// least twice)
    pub fn is_init(&self) -> bool {
        self.last_detection.is_some()
    }

    /// Update the tracker with a new frame.
    pub fn update(&mut self, frame: &SSL_DetectionFrame) {
        frame
            .balls
            .iter()
            .for_each(|b| debug_value("ball.confidence", b.confidence() as f64));

        let mut ball_measurements = frame
            .balls
            .iter()
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

        if ball_measurements.is_empty() {
            return;
        }
        ball_measurements.sort_by(|a, b| b.1.cmp(&a.1));
        let current_time = frame.t_capture();

        //if no last data, update with the first data
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
        } else {
            for (pos, is_noisy) in ball_measurements.iter() {
                let pos_ov = SVector::<f64, 3>::new(pos.x, pos.y, pos.z);
                let z = self
                    .filter
                    .as_mut()
                    .unwrap()
                    .update(pos_ov, current_time, false);
                if let Some(z) = z {
                    let mut pos_v3 = Vector3::new(z[0], z[2], z[4]);
                    let vel_v3 = Vector3::new(z[1], z[3], z[5]);
                    if pos_v3.z < 0.0 {
                        pos_v3.z = 0.0;
                        self.filter.as_mut().unwrap().set_x(SVector::<f64, 6>::new(
                            pos_v3.x,
                            vel_v3.x,
                            pos_v3.y,
                            vel_v3.y,
                            pos_v3.z,
                            -vel_v3.z,
                        ));
                    }
                    debug_line(
                        "ball.vel",
                        pos_v3.xy(),
                        pos_v3.xy() + vel_v3.xy(),
                        dies_core::DebugColor::Orange,
                    );
                    self.last_detection = Some(StoredData {
                        timestamp: current_time,
                        position: pos_v3,
                        raw_position: Vec::with_capacity(0),
                        velocity: vel_v3,
                    });
                }
            }
            self.last_detection.as_mut().unwrap().raw_position = measured_positions;
        }
    }

    pub fn get(&self) -> Option<BallData> {
        self.last_detection.as_ref().map(|data| BallData {
                timestamp: data.timestamp,
                raw_position: data
                    .raw_position
                    .iter()
                    .map(|v| to_dies_coords3(*v, self.play_dir_x))
                    .collect(),
                position: to_dies_coords3(data.position, self.play_dir_x),
                velocity: to_dies_coords3(data.velocity, self.play_dir_x),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_protos::ssl_vision_detection::{SSL_DetectionBall, SSL_DetectionFrame};

    #[test]
    fn test_update_no_ball() {
        let mut tracker = BallTracker::new(&TrackerSettings::default());
        let frame = SSL_DetectionFrame::new();

        tracker.update(&frame);
        let ball_data = tracker.get();
        assert!(ball_data.is_none());
    }

    #[test]
    fn test_basic_update() {
        let mut tracker = BallTracker::new(&TrackerSettings::default());

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
        assert_eq!(ball_data.raw_position[0], Vector3::new(2.0, 4.0, 6.0));
        assert!(ball_data.velocity.norm() > 0.0)
    }

    #[test]
    fn test_x_flip() {
        let settings = TrackerSettings {
            initial_opp_goal_x: -1.0,
            ..Default::default()
        };
        let mut tracker = BallTracker::new(&settings);

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
        assert_eq!(ball_data.raw_position[0], Vector3::new(-2.0, 4.0, 6.0));
        assert!(ball_data.velocity.x < 0.0)
    }

    fn generate_SSL_Wrapper(
        t: f64,
        xs: Vec<f64>,
        ys: Vec<f64>,
        zs: Vec<f64>,
    ) -> SSL_DetectionFrame {
        let mut frame = SSL_DetectionFrame::new();
        frame.set_t_capture(t);
        for i in 0..xs.len() {
            let mut ball = SSL_DetectionBall::new();
            ball.set_x((xs[i] * 1000.0) as f32);
            ball.set_y((ys[i] * 1000.0) as f32);
            ball.set_z((zs[i] * 1000.0) as f32);
            ball.confidence = Some(1.0);
            frame.balls.push(ball);
        }
        frame
    }
}
