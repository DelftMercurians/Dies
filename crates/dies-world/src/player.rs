use std::collections::VecDeque;

use dies_core::{Angle, PlayerFeedbackMsg, PlayerId, Vector2};
use dies_protos::ssl_vision_detection::SSL_DetectionRobot;
use nalgebra::{self as na, Vector4};

use crate::filter::{AngleLowPassFilter, MaybeKalman};
use crate::{tracker_settings::TrackerSettings, world::WorldInstant, PlayerFrame};

const BREAKBEAM_WINDOW: usize = 100;
const BREAKBEAM_DETECTION_THRESHOLD: usize = 5;

/// Stored data for a player from the last update.
struct StoredData {
    timestamp: f64,
    position: Vector2,
    velocity: Vector2,
    yaw: Angle,
    angular_speed: f64,
}

/// Tracker for a single player.
pub struct PlayerTracker {
    /// Player's unique id
    id: PlayerId,
    /// Whether this player receives feedback (is controlled)
    is_controlled: bool,

    /// Kalman filter for the player's position and velocity
    filter: MaybeKalman<2, 4>,
    /// Low-pass filter for the player's yaw
    yaw_filter: AngleLowPassFilter,

    /// Last feedback received from the player (if controlled)
    last_feedback: Option<PlayerFeedbackMsg>,
    last_feedback_time: Option<WorldInstant>,
    /// The result of the last vision update
    last_detection: Option<StoredData>,

    velocity_samples: Vec<Vector2>,

    /// How many positive breakbeam detections have been received recently
    breakbeam_detections: VecDeque<usize>,

    pub is_gone: bool,
    rolling_control: f64,
    rolling_vision: f64,
}

impl PlayerTracker {
    /// Create a new PlayerTracker.
    pub fn new(id: PlayerId, settings: &TrackerSettings, is_controlled: bool) -> PlayerTracker {
        PlayerTracker {
            id,
            is_controlled,
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
            rolling_vision: 0.0,
            rolling_control: 0.0,
        }
    }

    pub fn is_init(&self) -> bool {
        self.last_detection.is_some()
    }

    pub fn check_is_gone(&mut self, time: f64, world_time: WorldInstant) {
        if !self.is_controlled {
            let vision_val = if let Some(last_detection) = &self.last_detection {
                if time - last_detection.timestamp < 0.2 {
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            };
            self.rolling_vision = self.rolling_vision * 0.95 + vision_val * (1.0 - 0.95);
            if self.rolling_vision < 0.2 {
                self.is_gone = true;
            }
            if self.rolling_vision > 0.8 {
                self.is_gone = false;
            }

            return;
        }

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

        dies_core::debug_string(
            format!("p{}.rolling vision", self.id),
            format!("{}", self.rolling_vision),
        );
        dies_core::debug_string(
            format!("p{}.rolling control", self.id),
            format!("{}", self.rolling_control),
        );

        if self.rolling_control < 0.2 || self.rolling_vision < 0.2 {
            self.is_gone = true;
        }
        if self.rolling_control > 0.8 && self.rolling_vision > 0.8 {
            self.is_gone = false;
        }

        dies_core::debug_string(format!("p{}.is_gone", self.id), format!("{}", self.is_gone));
    }

    /// Update the tracker with a new frame.
    pub fn update(&mut self, t_capture: f64, player: &SSL_DetectionRobot) {
        let raw_position = Vector2::new(player.x() as f64, player.y() as f64);
        let raw_yaw_f64 = player.orientation() as f64;
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
                            position: na::convert(Vector2::new(x[0], x[2])),
                            velocity: na::convert(Vector2::new(x[1], x[3])),
                            yaw,
                            angular_speed: 0.0,
                        });
                        self.last_detection.as_mut().unwrap()
                    };
                    last_data.position = na::convert(Vector2::new(x[0], x[2]));
                    last_data.velocity = na::convert(Vector2::new(x[1], x[3]));
                    last_data.angular_speed = (yaw - last_data.yaw).radians()
                        / (t_capture - last_data.timestamp + f64::EPSILON);
                    last_data.yaw = yaw;
                    last_data.timestamp = t_capture;

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
        if !self.is_controlled || feedback.id != self.id {
            return;
        }

        self.last_feedback_time = Some(time);
        if let Some(breakbeam) = feedback.breakbeam_ball_detected {
            dies_core::debug_string(
                format!("p{}.breakbeam_value", self.id),
                breakbeam.to_string(),
            );
            self.breakbeam_detections.push_back(breakbeam as usize);
            if self.breakbeam_detections.len() > BREAKBEAM_WINDOW {
                self.breakbeam_detections.pop_front();
            }
        }

        self.last_feedback = Some(*feedback);
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

    pub fn get(&self) -> Option<PlayerFrame> {
        let breakbeam_count = self.breakbeam_detections.iter().sum::<usize>();
        if self.last_feedback.is_some() {
            dies_core::debug_value(format!("p{}.breakbeam", self.id), breakbeam_count as f64);
        }

        self.last_detection.as_ref().map(|data| PlayerFrame {
            id: self.id,
            timestamp: data.timestamp,
            position: data.position,
            velocity: data.velocity,
            yaw: data.yaw,
            angular_speed: data.angular_speed,
            is_controlled: self.is_controlled,
            primary_status: self.last_feedback.and_then(|f| f.primary_status),
            kicker_cap_voltage: self.last_feedback.and_then(|f| f.kicker_cap_voltage),
            kicker_temp: self.last_feedback.and_then(|f| f.kicker_temp),
            pack_voltages: self.last_feedback.and_then(|f| f.pack_voltages),
            breakbeam_ball_detected: breakbeam_count > BREAKBEAM_DETECTION_THRESHOLD,
            imu_status: self.last_feedback.and_then(|f| f.imu_status),
            kicker_status: self.last_feedback.and_then(|f| f.kicker_status),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_no_player() {
        let tracker = PlayerTracker::new(PlayerId::new(1), &TrackerSettings::default(), false);
        assert!(!tracker.is_init());
        assert!(tracker.get().is_none());
    }

    #[test]
    fn test_uncontrolled_player_update() {
        let mut tracker = PlayerTracker::new(PlayerId::new(1), &TrackerSettings::default(), false);

        // First update
        let mut player = SSL_DetectionRobot::new();
        player.set_x(100.0);
        player.set_y(200.0);
        player.set_orientation(0.0);

        tracker.update(0.0, &player);
        assert!(!tracker.is_init());

        // Second update
        tracker.update(1.0, &player);
        assert!(tracker.is_init());

        let data = tracker.get().unwrap();
        assert_eq!(data.id.as_u32(), 1);
        assert!(!data.is_controlled);
        assert_eq!(data.position, Vector2::new(100.0, 200.0));
    }

    #[test]
    fn test_controlled_player_update() {
        let mut tracker = PlayerTracker::new(PlayerId::new(1), &TrackerSettings::default(), true);

        // Vision update
        let mut player = SSL_DetectionRobot::new();
        player.set_x(100.0);
        player.set_y(200.0);
        player.set_orientation(0.0);

        tracker.update(0.0, &player);
        tracker.update(1.0, &player);

        // Feedback update
        let mut feedback = PlayerFeedbackMsg::empty(PlayerId::new(1));
        feedback.primary_status = Some(dies_core::SysStatus::Ready);
        tracker.update_from_feedback(&feedback, WorldInstant::simulated(1.0));

        let data = tracker.get().unwrap();
        assert_eq!(data.id.as_u32(), 1);
        assert!(data.is_controlled);
        assert_eq!(data.position, Vector2::new(100.0, 200.0));
    }

    #[test]
    fn test_velocity_calculation() {
        let mut tracker = PlayerTracker::new(PlayerId::new(1), &TrackerSettings::default(), false);

        // First position
        let mut player = SSL_DetectionRobot::new();
        player.set_x(0.0);
        player.set_y(0.0);
        player.set_orientation(0.0);
        tracker.update(0.0, &player);
        tracker.update(1.0, &player);

        // Move player
        player.set_x(100.0); // Move 100mm in 1 second
        player.set_y(0.0);
        tracker.update(2.0, &player);

        let data = tracker.get().unwrap();
        assert!(data.velocity.x > 0.0); // Should have positive x velocity
        assert_eq!(data.velocity.y, 0.0); // Should have no y velocity
    }
}
