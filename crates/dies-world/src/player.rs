use std::fmt::format;

use dies_core::{
    Angle, PlayerData, PlayerFeedbackMsg, PlayerId, TrackerSettings, Vector2, WorldInstant,
};
use dies_protos::ssl_vision_detection::SSL_DetectionRobot;
use nalgebra::{self as na, Vector4};

use crate::{
    coord_utils::{to_dies_coords2, to_dies_yaw},
    filter::{AngleLowPassFilter, MaybeKalman},
};

/// Stored data for a player from the last update.
///
/// This type contains **vision coordinates**, meaning the x axis is unchanged -- it may point towards our or the enemy goal.
///
/// For documentation on the fields, see the `PlayerData` struct in `dies-core`.
struct StoredData {
    timestamp: f64,
    raw_position: Vector2,
    position: Vector2,
    velocity: Vector2,
    yaw: Angle,
    raw_yaw: Angle,
    angular_speed: f64,
}

/// Tracker for a single player.
pub struct PlayerTracker {
    /// Player's unique id
    id: PlayerId,
    /// The sign of the enemy goal's x coordinate in ssl-vision coordinates. Used for
    /// converting coordinates.
    play_dir_x: f64,

    /// Kalman filter for the player's position and velocity
    filter: MaybeKalman<2, 4>,
    /// Low-pass filter for the player's yaw
    yaw_filter: AngleLowPassFilter,

    /// Last feedback received from the player
    last_feedback: Option<PlayerFeedbackMsg>,
    /// The result of the last vision update
    last_detection: Option<StoredData>,

    velocity_samples: Vec<Vector2>,

    breakbeam_triggered: Option<WorldInstant>,
}

impl PlayerTracker {
    /// Create a new PlayerTracker.
    pub fn new(id: PlayerId, settings: &TrackerSettings) -> PlayerTracker {
        PlayerTracker {
            id,
            play_dir_x: settings.initial_opp_goal_x,
            filter: MaybeKalman::new(
                0.1,
                settings.player_unit_transition_var,
                settings.player_measurement_var,
            ),
            yaw_filter: AngleLowPassFilter::new(settings.player_yaw_lpf_alpha),
            velocity_samples: Vec::new(),
            last_feedback: None,
            last_detection: None,
            breakbeam_triggered: None,
        }
    }

    pub fn is_init(&self) -> bool {
        self.last_detection.is_some()
    }

    /// Set the sign of the enemy goal's x coordinate in ssl-vision coordinates.
    pub fn set_play_dir_x(&mut self, play_dir_x: f64) {
        self.play_dir_x = play_dir_x;
    }

    /// Update the tracker with a new frame.
    pub fn update(&mut self, t_capture: f64, player: &SSL_DetectionRobot) {
        let raw_position = Vector2::new(player.x() as f64, player.y() as f64);
        let raw_yaw_f64 = player.orientation() as f64;
        let raw_yaw = Angle::from_radians(raw_yaw_f64);
        let yaw = Angle::from_radians(self.yaw_filter.update(raw_yaw_f64));

        match &mut self.filter {
            MaybeKalman::Init(filter) => {
                let z = na::convert(Vector2::new(raw_position.x, raw_position.y));
                if let Some(x) = filter.update(z, t_capture, false) {
                    let last_data = if let Some(last_data) = &mut self.last_detection {
                        last_data
                    } else {
                        self.last_detection = Some(StoredData {
                            timestamp: t_capture,
                            raw_position,
                            position: na::convert(Vector2::new(x[0], x[2])),
                            velocity: na::convert(Vector2::new(x[1], x[3])),
                            yaw,
                            raw_yaw,
                            angular_speed: 0.0,
                        });
                        self.last_detection.as_mut().unwrap()
                    };
                    last_data.raw_position = raw_position;
                    last_data.position = na::convert(Vector2::new(x[0], x[2]));
                    last_data.velocity = na::convert(Vector2::new(x[1], x[3]));
                    last_data.angular_speed = (yaw - last_data.yaw).radians()
                        / (t_capture - last_data.timestamp + std::f64::EPSILON);
                    last_data.yaw = yaw;
                    last_data.raw_yaw = raw_yaw;

                    if self.velocity_samples.len() < 10 {
                        self.velocity_samples.push(last_data.velocity);
                    } else {
                        self.velocity_samples.remove(0);
                        self.velocity_samples.push(last_data.velocity);

                        let acc = self.velocity_samples.windows(2).fold(0.0, |acc, w| {
                            acc + (w[1] - w[0]).norm() / (t_capture - last_data.timestamp)
                        }) / 9.0;
                        dies_core::debug_value(format!("p{}.acc", self.id), acc);
                    }

                    last_data.timestamp = t_capture;
                }
            }
            kalman => {
                kalman.init(
                    Vector4::new(raw_position.x, 0.0, raw_position.y, 0.0),
                    t_capture,
                );
            }
        }
    }

    /// Update the tracker with feedback from the player.
    pub fn update_from_feedback(&mut self, feedback: &PlayerFeedbackMsg, _time: WorldInstant) {
        if feedback.id == self.id {
            dies_core::debug_string(
                format!("p{}.breakbeam", self.id),
                feedback
                    .breakbeam_ball_detected
                    .unwrap_or(false)
                    .to_string(),
            );
            self.last_feedback = Some(*feedback);
        }
    }

    pub fn update_settings(&mut self, settings: &TrackerSettings) {
        if let Some(filter) = self.filter.as_mut() {
            filter.update_settings(
                settings.player_unit_transition_var,
                settings.player_measurement_var,
            );
        }
        self.yaw_filter
            .update_settings(settings.player_yaw_lpf_alpha);
    }

    pub fn get(&self) -> Option<PlayerData> {
        // If we have received feedback but not detection return some placeholder data
        // if let (None, Some(feedback)) = (self.last_detection.as_ref(), self.last_feedback) {
        //     return Some(PlayerData {
        //         id: self.id,
        //         timestamp: 0.0,
        //         position: Vector2::zeros(),
        //         velocity: Vector2::zeros(),
        //         yaw: Angle::default(),
        //         angular_speed: 0.0,
        //         raw_position: Vector2::zeros(),
        //         raw_yaw: Angle::default(),
        //         primary_status: feedback.primary_status,
        //         kicker_cap_voltage: feedback.kicker_cap_voltage,
        //         kicker_temp: feedback.kicker_temp,
        //         pack_voltages: feedback.pack_voltages,
        //         breakbeam_ball_detected: feedback.breakbeam_ball_detected.unwrap_or(false),
        //     });
        // }

        self.last_detection.as_ref().map(|data| PlayerData {
            id: self.id,
            timestamp: data.timestamp,
            position: to_dies_coords2(data.position, self.play_dir_x),
            velocity: to_dies_coords2(data.velocity, self.play_dir_x),
            yaw: to_dies_yaw(data.yaw, self.play_dir_x),
            // Flip the angular speed if the goal is on the left
            angular_speed: data.angular_speed * self.play_dir_x,
            raw_position: to_dies_coords2(data.raw_position, self.play_dir_x),
            raw_yaw: to_dies_yaw(data.raw_yaw, self.play_dir_x),
            primary_status: self.last_feedback.and_then(|f| f.primary_status),
            kicker_cap_voltage: self.last_feedback.and_then(|f| f.kicker_cap_voltage),
            kicker_temp: self.last_feedback.and_then(|f| f.kicker_temp),
            pack_voltages: self.last_feedback.and_then(|f| f.pack_voltages),
            breakbeam_ball_detected: self
                .last_feedback
                .and_then(|f| f.breakbeam_ball_detected)
                .unwrap_or(false),
            imu_status: self.last_feedback.and_then(|f| f.imu_status),
            kicker_status: self.last_feedback.and_then(|f| f.kicker_status),
        })
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_no_player() {
        let tracker = PlayerTracker::new(PlayerId::new(1), &TrackerSettings::default());

        assert!(!tracker.is_init());
        assert!(tracker.get().is_none());
    }

    #[test]
    fn test_no_data_after_one_update() {
        let mut tracker = PlayerTracker::new(PlayerId::new(1), &TrackerSettings::default());

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
        let id = PlayerId::new(1);
        let mut tracker = PlayerTracker::new(id, &TrackerSettings::default());

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
        assert_eq!(data.raw_position, Vector2::new(200.0, 300.0));
        assert!(data.velocity.norm() > 0.0);
        assert_eq!(data.raw_yaw, Angle::from_radians(1.0));
        assert!(data.angular_speed > 0.0);
    }

    #[test]
    fn test_x_flip() {
        let id = PlayerId::new(1);
        let settings = TrackerSettings {
            initial_opp_goal_x: -1.0,
            ..Default::default()
        };
        let mut tracker = PlayerTracker::new(id, &settings);

        let mut player = SSL_DetectionRobot::new();
        let dir = PI / 8.0;
        player.set_x(100.0);
        player.set_y(200.0);
        player.set_orientation(dir as f32);

        tracker.update(0.0, &player);
        assert!(!tracker.is_init());

        // Move player forward
        player.set_x(150.0);

        tracker.update(1.0, &player);
        assert!(tracker.is_init());

        let data = tracker.get().unwrap();
        assert_eq!(data.id.as_u32(), 1);
        assert_eq!(data.raw_position, Vector2::new(-150.0, 200.0));
        assert!(data.velocity.x < 0.0);
        assert_relative_eq!(
            data.yaw.radians(),
            (Angle::PI + Angle::from_radians(dir)).radians(),
            epsilon = 1e-6
        );
        assert_eq!(data.angular_speed, 0.0);

        tracker.set_play_dir_x(1.0);

        player.set_x(200.0);
        player.set_y(300.0);
        player.set_orientation(-dir as f32);

        tracker.update(2.0, &player);
        assert!(tracker.is_init());

        let data = tracker.get().unwrap();
        assert_eq!(data.id.as_u32(), 1);
        assert_eq!(data.raw_position, Vector2::new(200.0, 300.0));
        assert!(data.velocity.x > 0.0);
        assert_relative_eq!(data.raw_yaw.radians(), -dir, epsilon = 1e-6);
        assert!(data.angular_speed < 0.0);
    }
}
