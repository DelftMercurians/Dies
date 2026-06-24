use dies_core::{debug_value, BallData, FieldGeometry, FieldMask, TrackerSettings, Vector3};
use dies_protos::ssl_vision_detection::SSL_DetectionFrame;
use nalgebra::{SVector, Vector6};

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
    /// Kalman filter for the ball's position and velocity
    filter: MaybeKalman<3, 6>,

    /// Last reported ball state. **Sticky**: once the ball has been seen it is
    /// never cleared, only overwritten by a newer accepted measurement. This is
    /// what `get()` reports, so we never stop emitting a ball position just
    /// because vision lost sight of it (e.g. a robot covering the ball). The
    /// `detected` flag on the returned `BallData` carries freshness instead.
    last_detection: Option<StoredData>,

    /// Consecutive frames without an accepted measurement. Drives the `detected`
    /// freshness flag and triggers a filter re-acquisition after a long dropout.
    misses: u32,

    /// Filter variances, kept so the filter can be rebuilt when the active track
    /// is reset (`MaybeKalman::init` is a no-op once initialized).
    init_var: f64,
    transition_var: f64,
    measurement_var: f64,

    /// Minimum vision confidence to accept a detection.
    confidence_threshold: f64,
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
            init_var: 0.1,
            transition_var: settings.ball_unit_transition_var,
            measurement_var: settings.ball_measurement_var,
            confidence_threshold: settings.ball_confidence_threshold,
        }
    }

    pub fn update_settings(&mut self, settings: &TrackerSettings) {
        self.transition_var = settings.ball_unit_transition_var;
        self.measurement_var = settings.ball_measurement_var;
        self.confidence_threshold = settings.ball_confidence_threshold;
        if let Some(filter) = self.filter.as_mut() {
            filter.update_settings(
                settings.ball_unit_transition_var,
                settings.ball_measurement_var,
            );
        }
    }

    /// Reset the Kalman filter so the next confident detection re-initializes it
    /// cleanly, anywhere on the field, instead of being gated against a stale
    /// position. The reported position (`last_detection`) is **kept** — only the
    /// active filter track is dropped — and its velocity is zeroed since we no
    /// longer have evidence the ball is moving.
    fn reset_filter(&mut self) {
        self.filter = MaybeKalman::new(self.init_var, self.transition_var, self.measurement_var);
        if let Some(data) = self.last_detection.as_mut() {
            data.velocity = Vector3::zeros();
        }
    }

    /// Whether the ball has been seen at least once. Sticky: stays true after the
    /// first sighting (see `last_detection`).
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
            .filter(|ball| {
                !ball.has_confidence() || ball.confidence() as f64 >= self.confidence_threshold
            })
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

        // Gate against the *active filter track* (not the sticky reported
        // position): with a live track we trust nearby measurements; once the
        // track has been reset after a long dropout there is no reference and we
        // re-acquire freely.
        let gate_ref = if self.filter.is_init() {
            self.last_detection.as_ref().map(|d| d.position)
        } else {
            None
        };

        // Pick the measurement to feed the filter:
        //  - with an active track: the one nearest the current estimate, but
        //    only if it's within the gate radius (rejects a far-away blob from
        //    yanking the track);
        //  - without a track: the highest-confidence candidate (preferred), so a
        //    real ball wins over a low-confidence phantom at acquisition.
        const GATE_RADIUS: f64 = 500.0; // mm
        let accepted = match gate_ref {
            Some(prev) => ball_measurements
                .iter()
                .map(|(pos, _)| *pos)
                .filter(|pos| (pos - prev).norm() < GATE_RADIUS)
                .min_by(|a, b| {
                    let da = (a - prev).norm();
                    let db = (b - prev).norm();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                }),
            None => {
                // `false` (confidence >= 0.9) sorts first.
                ball_measurements.sort_by(|a, b| a.1.cmp(&b.1));
                ball_measurements.first().map(|(pos, _)| *pos)
            }
        };

        let pos = match accepted {
            Some(pos) => pos,
            None => {
                // No usable measurement this frame.
                self.misses += 1;
                // After a long dropout, reset the *filter* so it can re-acquire
                // the ball wherever it next appears. The reported position is kept
                // (we never stop emitting a ball position) and just goes stale.
                const RESET_FILTER_AFTER_MISSES: u32 = 30; // ~0.5s at 60Hz
                if self.misses == RESET_FILTER_AFTER_MISSES + 1 {
                    self.reset_filter();
                }
                return;
            }
        };
        self.misses = 0;

        if !self.filter.is_init() {
            self.last_detection = Some(StoredData {
                timestamp: current_time,
                position: pos,
                raw_position: measured_positions,
                velocity: Vector3::zeros(),
            });
            self.filter.init(
                Vector6::new(pos.x, 0.0, pos.y, 0.0, pos.z, 0.0),
                current_time,
            );
        } else {
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
            self.last_detection.as_mut().unwrap().raw_position = measured_positions;
        }

        dies_core::debug_value(
            "ball_speed",
            self.last_detection.as_ref().unwrap().velocity.xy().norm(),
        );
    }

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
