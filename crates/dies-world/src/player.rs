use std::collections::{HashSet, VecDeque};

use dies_core::{
    Angle, Handicap, PlayerData, PlayerFeedbackMsg, PlayerId, TrackerSettings, Vector2,
    WorldInstant,
};
use dies_protos::ssl_vision_detection::SSL_DetectionRobot;
use nalgebra::{self as na, Vector4};

use crate::filter::{AngleLowPassFilter, MaybeKalman};

const BREAKBEAM_WINDOW: usize = 4;

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
    pub id: PlayerId,
    /// Whether this player's team is controlled (we receive feedback)
    is_controlled: bool,
    allow_no_vision: bool,

    /// Kalman filter for the player's position and velocity
    filter: MaybeKalman<2, 4>,
    /// Low-pass filter for the player's yaw
    yaw_filter: AngleLowPassFilter,

    /// Last feedback received from the player
    last_feedback: Option<PlayerFeedbackMsg>,
    last_feedback_time: Option<WorldInstant>,
    /// The result of the last vision update
    last_detection: Option<StoredData>,

    velocity_samples: Vec<Vector2>,

    /// How many positive breakbeam detections have been received in the past
    breakbeam_detections: VecDeque<usize>,

    pub is_gone: bool,
    rolling_control: f64,
    rolling_vision: f64,

    handicaps: HashSet<Handicap>,
}

impl PlayerTracker {
    /// Create a new PlayerTracker.
    pub fn new(
        id: PlayerId,
        handicaps: HashSet<Handicap>,
        settings: &TrackerSettings,
        allow_no_vision: bool,
    ) -> PlayerTracker {
        PlayerTracker {
            id,
            is_controlled: false, // Will be set to true when feedback is received
            allow_no_vision,
            filter: MaybeKalman::new(
                0.1,
                settings.player_unit_transition_var,
                settings.player_measurement_var,
            ),
            yaw_filter: AngleLowPassFilter::new(settings.player_yaw_lpf_alpha),
            velocity_samples: Vec::new(),
            last_feedback: None,
            last_detection: None,
            breakbeam_detections: VecDeque::with_capacity(BREAKBEAM_WINDOW),
            is_gone: false,
            last_feedback_time: None,
            rolling_vision: 1.0,
            rolling_control: 1.0,
            handicaps,
        }
    }

    pub fn is_init(&self) -> bool {
        self.last_detection.is_some()
    }

    pub fn check_is_gone(&mut self, time: f64, world_time: WorldInstant) {
        let vision_val = if let Some(last_detection) = &self.last_detection {
            if time - last_detection.timestamp < 0.2 {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        let control_val = if let Some(last_feedback) = &self.last_feedback_time {
            dies_core::debug_string(
                format!("player_{}_feedback_time", self.id),
                format!("{:.2}ms", world_time.duration_since(last_feedback) / 1000.0),
            );
            if world_time.duration_since(last_feedback) < 0.5 {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        let factor = 0.96;
        self.rolling_vision = self.rolling_vision * factor + vision_val * (1.0 - factor);
        self.rolling_control = self.rolling_control * factor + control_val * (1.0 - factor);

        // For controlled players, require both vision and control
        // For non-controlled players (opponent players), only require vision
        if self.is_controlled {
            if self.rolling_control < 0.2 || self.rolling_vision < 0.2 {
                self.is_gone = true;
            }
            if self.rolling_control > 0.8 && self.rolling_vision > 0.8 {
                self.is_gone = false;
            }
        } else {
            if self.rolling_vision < 0.2 {
                self.is_gone = true;
            }
            if self.rolling_vision > 0.8 {
                self.is_gone = false;
            }
        }
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
                    let dt = t_capture - last_data.timestamp;
                    last_data.raw_position = raw_position;
                    last_data.velocity = na::convert(Vector2::new(x[1], x[3]));
                    last_data.position = na::convert(Vector2::new(x[0], x[2]));
                    last_data.angular_speed = (yaw - last_data.yaw).radians()
                        / (t_capture - last_data.timestamp + std::f64::EPSILON);
                    last_data.yaw = yaw;
                    last_data.raw_yaw = raw_yaw;

                    if self.velocity_samples.len() < 10 {
                        self.velocity_samples.push(last_data.velocity);
                    } else {
                        self.velocity_samples.remove(0);
                        self.velocity_samples.push(last_data.velocity);

                        let acc = self
                            .velocity_samples
                            .windows(2)
                            .fold(0.0, |acc, w| acc + (w[1] - w[0]).norm() / dt)
                            / 9.0;
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
    pub fn update_from_feedback(&mut self, feedback: &PlayerFeedbackMsg, time: WorldInstant) {
        if feedback.id == self.id {
            if !self.is_controlled {
                self.rolling_control = 1.0;
            }
            self.is_controlled = true; // Mark as controlled when we receive feedback
            self.last_feedback_time = Some(time);
            if let Some(breakbeam) = feedback.breakbeam_ball_detected {
                self.breakbeam_detections.push_back(breakbeam as usize);
                if self.breakbeam_detections.len() > BREAKBEAM_WINDOW {
                    self.breakbeam_detections.pop_front();
                }
            }

            self.last_feedback = Some(*feedback);
        }
    }

    pub fn update_settings(&mut self, handicaps: HashSet<Handicap>, settings: &TrackerSettings) {
        if let Some(filter) = self.filter.as_mut() {
            filter.update_settings(
                settings.player_unit_transition_var,
                settings.player_measurement_var,
            );
        }
        self.handicaps = handicaps;
        self.yaw_filter
            .update_settings(settings.player_yaw_lpf_alpha);
    }

    /// Returns filtered breakbeam detection status.
    /// Returns true if there's at least one detection within the buffer.
    fn get_filtered_breakbeam_detection(&self) -> bool {
        self.breakbeam_detections.iter().any(|&detection| detection != 0)
    }

    pub fn get(&self) -> Option<PlayerData> {
        if self.last_detection.is_none() && self.allow_no_vision {
            return self.last_feedback.map(|f| PlayerData {
                id: self.id,
                timestamp: 0.0,
                position: Vector2::zeros(),
                velocity: Vector2::zeros(),
                yaw: Angle::from_radians(0.0),
                angular_speed: 0.0,
                raw_position: Vector2::zeros(),
                raw_yaw: Angle::from_radians(0.0),
                primary_status: f.primary_status,
                kicker_cap_voltage: f.kicker_cap_voltage,
                kicker_temp: f.kicker_temp,
                pack_voltages: f.pack_voltages,
                breakbeam_ball_detected: self.get_filtered_breakbeam_detection(),
                imu_status: f.imu_status,
                kicker_status: f.kicker_status,
                handicaps: self.handicaps.clone(),
            });
        }
        self.last_detection.as_ref().map(|data| PlayerData {
            id: self.id,
            timestamp: data.timestamp,
            position: data.position,
            velocity: data.velocity,
            yaw: data.yaw,
            angular_speed: data.angular_speed,
            raw_position: data.raw_position,
            raw_yaw: data.raw_yaw,
            primary_status: self.last_feedback.and_then(|f| f.primary_status),
            kicker_cap_voltage: self.last_feedback.and_then(|f| f.kicker_cap_voltage),
            kicker_temp: self.last_feedback.and_then(|f| f.kicker_temp),
            pack_voltages: self.last_feedback.and_then(|f| f.pack_voltages),
            breakbeam_ball_detected: self.get_filtered_breakbeam_detection(),
            imu_status: self.last_feedback.and_then(|f| f.imu_status),
            kicker_status: self.last_feedback.and_then(|f| f.kicker_status),
            handicaps: self.handicaps.clone(),
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
        let tracker = PlayerTracker::new(
            PlayerId::new(1),
            HashSet::new(),
            &TrackerSettings::default(),
            false,
        );

        assert!(!tracker.is_init());
        assert!(tracker.get().is_none());
    }

    #[test]
    fn test_no_data_after_one_update() {
        let mut tracker = PlayerTracker::new(
            PlayerId::new(1),
            HashSet::new(),
            &TrackerSettings::default(),
            false,
        );

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
        let mut tracker =
            PlayerTracker::new(id, HashSet::new(), &TrackerSettings::default(), false);

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
        let settings = TrackerSettings::default();
        let mut tracker = PlayerTracker::new(id, HashSet::new(), &settings, false);

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
        // Note: Now returns raw vision coordinates without transformation
        assert_eq!(data.raw_position, Vector2::new(150.0, 200.0));
        assert!(data.velocity.x > 0.0);
        assert_relative_eq!(data.yaw.radians(), dir, epsilon = 1e-6);
        assert_eq!(data.angular_speed, 0.0);

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
