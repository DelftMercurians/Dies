use std::collections::HashSet;

use dies_core::{
    Angle, Handicap, PlayerData, PlayerFeedbackMsg, PlayerId, TrackerSettings, Vector2,
    WorldInstant,
};
use dies_protos::ssl_vision_detection::SSL_DetectionRobot;
use nalgebra::{self as na, Vector4};

use crate::filter::{AngleLowPassFilter, MaybeKalman};

/// Upper bound on plausible robot speed (mm/s) used for teleport detection. Well
/// above the controller cap (~3 m/s) and the simulator's max (6 m/s), so real
/// motion is never mistaken for a discontinuity.
const MAX_PLAUSIBLE_SPEED: f64 = 8000.0;
/// Absolute slack (mm) added to the per-frame travel budget so vision noise on a
/// stationary robot never trips the teleport reset.
const TELEPORT_MARGIN: f64 = 150.0;

/// Player position filter, selectable between a constant-velocity model (state
/// `x, vx, y, vy`) and a constant-acceleration model (state `x, vx, ax, y, vy,
/// ay`). Both measure position only and expose a common `(position, velocity)`
/// interface, so the tracker is agnostic to the state layout. The CA model
/// anticipates sustained acceleration, reducing the velocity lag in hard
/// maneuvers at the cost of a little more velocity noise.
enum PlayerFilter {
    Cv(MaybeKalman<2, 4>),
    Ca(MaybeKalman<2, 6>),
}

impl PlayerFilter {
    fn new(use_accel: bool, init_var: f64, cv_var: f64, ca_var: f64, measurement_var: f64) -> Self {
        if use_accel {
            PlayerFilter::Ca(MaybeKalman::new(init_var, ca_var, measurement_var))
        } else {
            PlayerFilter::Cv(MaybeKalman::new(init_var, cv_var, measurement_var))
        }
    }

    /// Whether the underlying Kalman filter has been seeded.
    fn is_init(&self) -> bool {
        matches!(
            self,
            PlayerFilter::Cv(MaybeKalman::Init(_)) | PlayerFilter::Ca(MaybeKalman::Init(_))
        )
    }

    /// Seed the filter at a measured position with zero velocity (and acceleration).
    fn init(&mut self, pos: Vector2, t: f64) {
        match self {
            PlayerFilter::Cv(k) => k.init(Vector4::new(pos.x, 0.0, pos.y, 0.0), t),
            PlayerFilter::Ca(k) => k.init(
                na::SVector::<f64, 6>::new(pos.x, 0.0, 0.0, pos.y, 0.0, 0.0),
                t,
            ),
        }
    }

    /// Run the measurement update, returning `(position, velocity)` in the input
    /// (vision) frame, or `None` if the filter is uninitialised or the
    /// measurement is stale.
    ///
    /// `feedforward` is an optional `(commanded_velocity, tau)` in the **vision
    /// frame**. When set, the predict step nudges the state toward the command via
    /// a first-order lag with time constant `tau` (acceleration `(u - v)/tau`),
    /// reducing lag during commanded maneuvers.
    fn update(
        &mut self,
        z: Vector2,
        t: f64,
        use_gate: bool,
        feedforward: Option<(Vector2, f64)>,
    ) -> Option<(Vector2, Vector2)> {
        let zz = na::convert(z);
        match self {
            PlayerFilter::Cv(MaybeKalman::Init(f)) => {
                let ci = feedforward.map(|(u, tau)| {
                    let dt = t - f.last_t();
                    let s = f.state(); // [x, vx, y, vy]
                    let v = Vector2::new(s[1], s[3]);
                    let a = (u - v) / tau; // commanded acceleration (first-order lag)
                    na::SVector::<f64, 4>::new(
                        0.5 * a.x * dt * dt,
                        a.x * dt,
                        0.5 * a.y * dt * dt,
                        a.y * dt,
                    )
                });
                f.update_with_control(zz, t, use_gate, ci.as_ref())
                    .map(|x| (Vector2::new(x[0], x[2]), Vector2::new(x[1], x[3])))
            }
            PlayerFilter::Ca(MaybeKalman::Init(f)) => {
                let ci = feedforward.map(|(u, tau)| {
                    let dt = t - f.last_t();
                    let s = f.state(); // [x, vx, ax, y, vy, ay]
                    let v = Vector2::new(s[1], s[4]);
                    // Velocity feedforward only: nudge velocity (and position
                    // consistently) toward the command via a first-order lag. The
                    // acceleration state is left to the measurements — the command
                    // is a velocity setpoint, and forcing acc to (u-v)/tau would
                    // fight the CA model's own (correct) acceleration estimate.
                    let a = (u - v) / tau;
                    na::SVector::<f64, 6>::new(
                        0.5 * a.x * dt * dt,
                        a.x * dt,
                        0.0,
                        0.5 * a.y * dt * dt,
                        a.y * dt,
                        0.0,
                    )
                });
                f.update_with_control(zz, t, use_gate, ci.as_ref())
                    .map(|x| (Vector2::new(x[0], x[3]), Vector2::new(x[1], x[4])))
            }
            _ => None,
        }
    }

    /// Hard-reset to a measured position (zero velocity/acceleration).
    fn reset_to(&mut self, x: f64, y: f64, t: f64) {
        match self {
            PlayerFilter::Cv(MaybeKalman::Init(f)) => f.reset_to(x, y, t),
            PlayerFilter::Ca(MaybeKalman::Init(f)) => f.reset_to(x, y, t),
            _ => {}
        }
    }

    fn update_settings(&mut self, cv_var: f64, ca_var: f64, measurement_var: f64) {
        match self {
            PlayerFilter::Cv(k) => {
                if let Some(f) = k.as_mut() {
                    f.update_settings(cv_var, measurement_var);
                }
            }
            PlayerFilter::Ca(k) => {
                if let Some(f) = k.as_mut() {
                    f.update_settings(ca_var, measurement_var);
                }
            }
        }
    }
}

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

    /// Kalman filter for the player's position and velocity (CV or CA model)
    filter: PlayerFilter,
    /// Low-pass filter for the player's yaw
    yaw_filter: AngleLowPassFilter,

    /// Last feedback received from the player
    last_feedback: Option<PlayerFeedbackMsg>,
    last_feedback_time: Option<WorldInstant>,
    /// The result of the last vision update
    last_detection: Option<StoredData>,

    velocity_samples: Vec<Vector2>,

    pub is_gone: bool,
    rolling_control: f64,
    rolling_vision: f64,

    /// Command feedforward: latest commanded velocity in the **vision frame**,
    /// plus whether feedforward is enabled and its time constant. The executor
    /// pushes a command every control tick (zero when idle), so the value stays
    /// fresh; feedforward is simply skipped until the first command arrives.
    use_command_feedforward: bool,
    command_tau: f64,
    last_command: Option<Vector2>,

    /// EWMA of squared innovation magnitude (mm²). The Kalman innovation is
    /// `raw_measurement − model_prediction`; in steady state its RMS is the
    /// noise floor that downstream consumers (MPC, planners) actually see.
    /// Use `position_noise_rms()` for a human-readable mm value.
    position_noise_var: f64,

    handicaps: HashSet<Handicap>,
}

impl PlayerTracker {
    /// Create a new PlayerTracker.
    pub fn new(
        id: PlayerId,
        handicaps: HashSet<Handicap>,
        settings: &TrackerSettings,
        allow_no_vision: bool,
        is_controlled: bool,
    ) -> PlayerTracker {
        PlayerTracker {
            id,
            is_controlled,
            allow_no_vision,
            filter: PlayerFilter::new(
                settings.player_use_acceleration,
                0.1,
                settings.player_unit_transition_var,
                settings.player_ca_unit_transition_var,
                settings.player_measurement_var,
            ),
            use_command_feedforward: settings.player_use_command_feedforward,
            command_tau: settings.player_command_tau,
            last_command: None,
            yaw_filter: AngleLowPassFilter::new(settings.player_yaw_lpf_alpha),
            velocity_samples: Vec::new(),
            last_feedback: None,
            last_detection: None,
            is_gone: false,
            last_feedback_time: None,
            rolling_vision: 1.0,
            rolling_control: 1.0,
            position_noise_var: 0.0,
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
            if world_time.duration_since(last_feedback) < 0.5 {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        let factor = 0.99;
        self.rolling_vision = self.rolling_vision * factor + vision_val * (1.0 - factor);
        let factor = 0.99;
        self.rolling_control = self.rolling_control * factor + control_val * (1.0 - factor);

        // For controlled players, require both vision and control
        // For non-controlled players (opponent players), only require vision
        if self.is_controlled {
            if self.rolling_control < 0.1 || self.rolling_vision < 0.1 {
                if !self.is_gone && self.rolling_control < 0.1 {
                    log::warn!("Player {} is gone (control)", self.id);
                } else if !self.is_gone && self.rolling_vision < 0.1 {
                    log::warn!("Player {} is gone (vision)", self.id);
                }
                self.is_gone = true;
            }
            if self.rolling_control > 0.8 && self.rolling_vision > 0.8 {
                if self.is_gone {
                    log::warn!("Player {} is back", self.id);
                }
                self.is_gone = false;
            }
        } else {
            if self.rolling_vision < 0.1 {
                if !self.is_gone {
                    log::warn!("Player {} is gone (vision)", self.id);
                }
                self.is_gone = true;
            }
            if self.rolling_vision > 0.8 {
                if self.is_gone {
                    log::warn!("Player {} is back", self.id);
                }
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

        // Teleport / discontinuity detection. If the new measurement is too far
        // from the constant-velocity prediction to be real motion, the filter
        // would otherwise absorb the jump as a huge velocity and ring for ~1 s
        // (breaking skills and looking like the robot teleports). Hard-reset the
        // filter to the measurement with zero velocity instead.
        let teleported = if let Some(last) = &self.last_detection {
            let dt = t_capture - last.timestamp;
            if dt > 0.0 {
                let predicted = last.position + last.velocity * dt;
                let jump = (raw_position - predicted).norm();
                jump > MAX_PLAUSIBLE_SPEED * dt + TELEPORT_MARGIN
            } else {
                false
            }
        } else {
            false
        };

        // Not seeded yet → initialise at the measurement and wait for the next frame.
        if !self.filter.is_init() {
            self.filter.init(raw_position, t_capture);
            return;
        }

        if teleported {
            self.filter
                .reset_to(raw_position.x, raw_position.y, t_capture);
            if let Some(last) = &mut self.last_detection {
                last.timestamp = t_capture;
                last.raw_position = raw_position;
                last.position = raw_position;
                last.velocity = Vector2::zeros();
                last.yaw = yaw;
                last.raw_yaw = raw_yaw;
                last.angular_speed = 0.0;
            }
            self.velocity_samples.clear();
            return;
        }

        let feedforward = if self.use_command_feedforward {
            self.last_command.map(|u| (u, self.command_tau))
        } else {
            None
        };
        if let Some((position, velocity)) =
            self.filter
                .update(raw_position, t_capture, false, feedforward)
        {
            let last_data = if let Some(last_data) = &mut self.last_detection {
                last_data
            } else {
                self.last_detection = Some(StoredData {
                    timestamp: t_capture,
                    raw_position,
                    position,
                    velocity,
                    yaw,
                    raw_yaw,
                    angular_speed: 0.0,
                });
                self.last_detection.as_mut().unwrap()
            };
            let dt = t_capture - last_data.timestamp;
            // Innovation = raw measurement − constant-velocity prediction from the
            // previous filtered state. EWMA over ~20 frames (~0.3 s at 60 Hz
            // vision). Skipped on the bootstrap frame where dt = 0 and last_data
            // was just constructed from the filter output.
            if dt > 0.0 {
                let predicted = last_data.position + last_data.velocity * dt;
                let innovation = raw_position - predicted;
                const ALPHA: f64 = 0.05;
                self.position_noise_var =
                    (1.0 - ALPHA) * self.position_noise_var + ALPHA * innovation.norm_squared();
                // Note: the RMS is surfaced to the UI via `PlayerData.position_noise`;
                // no separate debug tag is emitted (avoids a loose `p{id}.*` key).
            }
            last_data.raw_position = raw_position;
            last_data.velocity = velocity;
            last_data.position = position;
            last_data.angular_speed = (yaw - last_data.yaw).radians()
                / (t_capture - last_data.timestamp + std::f64::EPSILON);
            last_data.yaw = yaw;
            last_data.raw_yaw = raw_yaw;

            if self.velocity_samples.len() < 10 {
                self.velocity_samples.push(last_data.velocity);
            } else {
                self.velocity_samples.remove(0);
                self.velocity_samples.push(last_data.velocity);

                let _acc = self
                    .velocity_samples
                    .windows(2)
                    .fold(0.0, |acc, w| acc + (w[1] - w[0]).norm() / dt)
                    / 9.0;
            }

            last_data.timestamp = t_capture;
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
            self.last_feedback = Some(*feedback);
        }
    }

    pub fn update_settings(&mut self, handicaps: HashSet<Handicap>, settings: &TrackerSettings) {
        self.filter.update_settings(
            settings.player_unit_transition_var,
            settings.player_ca_unit_transition_var,
            settings.player_measurement_var,
        );
        self.use_command_feedforward = settings.player_use_command_feedforward;
        self.command_tau = settings.player_command_tau;
        self.handicaps = handicaps;
        self.yaw_filter
            .update_settings(settings.player_yaw_lpf_alpha);
    }

    /// Set the latest commanded velocity for this player, in the **vision frame**
    /// (same frame as the raw detections). Consumed by command feedforward on the
    /// next vision update. No-op effect unless `player_use_command_feedforward`.
    pub fn set_command(&mut self, vel_vision: Vector2) {
        self.last_command = Some(vel_vision);
    }

    /// Latest raw breakbeam reading from feedback (false if none). Temporal
    /// conditioning now lives in the unified possession tracker, not here.
    fn raw_breakbeam(&self) -> bool {
        self.last_feedback
            .and_then(|f| f.breakbeam_ball_detected)
            .unwrap_or(false)
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
                position_noise: 0.0,
                raw_position: Vector2::zeros(),
                raw_yaw: Angle::from_radians(0.0),
                primary_status: f.primary_status,
                kicker_cap_voltage: f.kicker_cap_voltage,
                kicker_temp: f.kicker_temp,
                pack_voltages: f.pack_voltages,
                breakbeam_ball_detected: self.raw_breakbeam(),
                has_ball: false,
                imu_status: f.imu_status,
                imu_readings: f.imu_readings,
                kicker_status: f.kicker_status,
                skill: None,
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
            position_noise: self.position_noise_var.sqrt(),
            raw_position: data.raw_position,
            raw_yaw: data.raw_yaw,
            primary_status: self.last_feedback.and_then(|f| f.primary_status),
            kicker_cap_voltage: self.last_feedback.and_then(|f| f.kicker_cap_voltage),
            kicker_temp: self.last_feedback.and_then(|f| f.kicker_temp),
            pack_voltages: self.last_feedback.and_then(|f| f.pack_voltages),
            breakbeam_ball_detected: self.raw_breakbeam(),
            has_ball: false,
            imu_status: self.last_feedback.and_then(|f| f.imu_status),
            imu_readings: self.last_feedback.and_then(|f| f.imu_readings),
            kicker_status: self.last_feedback.and_then(|f| f.kicker_status),
            skill: None,
            handicaps: self.handicaps.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive a filter along a known accelerating trajectory and return the mean
    /// absolute velocity error (mm/s) vs the true velocity over the run.
    fn run_accel(use_ca: bool, feedforward: bool) -> f64 {
        let dt = 1.0 / 60.0; // 60 Hz vision
        let tau = 0.15;
        let accel = 3000.0_f64; // mm/s^2, a hard but realistic ramp along +x
        let cv_q = 3000.0;
        let ca_q = 1.0e6;
        let r = 0.4;
        let mut f = PlayerFilter::new(use_ca, 0.1, cv_q, ca_q, r);

        let mut t = 0.0;
        // seed
        f.init(Vector2::new(0.0, 0.0), t);
        let mut err_sum = 0.0;
        let mut n = 0;
        for step in 1..=120 {
            t = step as f64 * dt;
            let true_pos = 0.5 * accel * t * t;
            let true_vel = accel * t;
            let z = Vector2::new(true_pos, 0.0);
            let ff = if feedforward {
                Some((Vector2::new(true_vel, 0.0), tau))
            } else {
                None
            };
            if let Some((_pos, vel)) = f.update(z, t, false, ff) {
                // skip the first few frames while the filter converges
                if step > 20 {
                    err_sum += (vel.x - true_vel).abs();
                    n += 1;
                }
            }
        }
        err_sum / n as f64
    }

    #[test]
    fn command_feedforward_reduces_velocity_lag() {
        // Constant-velocity model: feedforward should markedly cut the velocity
        // error during sustained acceleration (the lag the filter can't model).
        let cv_off = run_accel(false, false);
        let cv_on = run_accel(false, true);
        assert!(
            cv_on < cv_off * 0.8,
            "CV feedforward should cut velocity error >20%: off={cv_off:.1} on={cv_on:.1}"
        );

        // Constant-acceleration model already tracks ramps well; feedforward should
        // not make it worse.
        let ca_off = run_accel(true, false);
        let ca_on = run_accel(true, true);
        assert!(
            ca_on <= ca_off * 1.05,
            "CA feedforward should not worsen tracking: off={ca_off:.1} on={ca_on:.1}"
        );
    }
}
