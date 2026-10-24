//! Simulator configuration and defaults.
use crate::{ball::BALL_RADIUS, realism::RealismConfig};
use dies_core::{FieldGeometry, SideAssignment};
use rapier3d_f64::prelude::*;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct SimulationConfig {
    // PHYSICAL CONSTANTS
    /// Gravity vector in mm/s^2
    pub gravity: Vector<f64>,
    /// Linear damping on the ball (rapier, proportional to speed). Kept at 0 —
    /// rolling friction is modelled by [`Self::ball_roll_decel`] instead, because
    /// proportional decay has the wrong shape (kills fast balls in the first
    /// metre, lets slow balls creep forever).
    pub ball_damping: f64,
    /// Constant rolling deceleration (mm/s²) applied to the ball's planar
    /// velocity each step — real carpet friction: speed falls linearly with
    /// time, v² linearly with distance. Calibrated against the 2026-07-02 real
    /// match logs (`.analysis/ball_speed_real.py`): median ≈ 1310 mm/s².
    pub ball_roll_decel: f64,
    /// Time between vision updates
    pub vision_update_step: f64,

    // ROBOT MODEL PARAMETERS
    /// Radius of the player in mm
    pub player_radius: f64,
    /// Height of the player in mm
    pub player_height: f64,
    /// Maximum reach of the dribbler from the edge of the robot in mm
    pub dribbler_radius: f64,
    /// Maximum angle from the front of the robot where the dribbler can pick up the ball
    /// in radians
    pub dribbler_angle: f64,
    /// Maximum force exerted by the kicker
    pub kicker_strength: f64,
    /// Time after which the player defaults to zero velocity if no command is received
    pub player_cmd_timeout: f64,
    /// Strength of the dribbler in picking up the ball
    pub dribbler_strength: f64,
    /// Delay for the execution of the command
    pub command_delay: f64,
    /// Maximum lateral acceleration in mm/s^2 (speeding up — sluggish).
    pub max_accel: f64,
    /// Maximum lateral deceleration in mm/s^2 (braking — much harder than
    /// acceleration on the real robots, which is why they stop without
    /// overshoot even when starting is slow).
    pub max_decel: f64,
    /// Maximum lateral speed in mm/s
    pub max_vel: f64,
    /// Maximum angular acceleration in rad/s^2
    pub max_ang_accel: f64,
    /// Maximum angular velocity in rad/s
    pub max_ang_vel: f64,
    /// Maximum difference between target and current velocity
    pub velocity_treshold: f64,
    /// Maximum difference between target and current angular velocity
    pub angular_velocity_treshold: f64,
    /// Interval for sending feedback packets in seconds
    pub feedback_interval: f64,
    /// Whether the robot has an IMU and is capable of absolute orientation
    pub has_imu: bool,

    // FIELD GEOMRTY PARAMETERS
    /// Field configuration
    pub field_geometry: FieldGeometry,
    /// Interval for sending geometry packets
    pub geometry_interval: f64,

    // TEAM CONFIGURATION
    /// Which teams are controlled/simulated by the AI
    pub blue_controlled: bool,
    pub yellow_controlled: bool,
    pub initial_side_assignment: SideAssignment,

    /// Number of robots to spawn for each team in the seeded/default builders
    /// (1..=6). Lets headless self-play run uneven or short-handed matches
    /// (e.g. 3v6) to exercise dynamic-robot-count behaviour.
    pub n_blue_robots: usize,
    pub n_yellow_robots: usize,

    /// Seed for the simulation RNG. Drives initial pose jitter (and any other
    /// seeded variation) so headless self-play matches are reproducible: same
    /// seed → identical match, different seed → different-but-reproducible match.
    pub seed: u64,

    /// Imperfect-hardware noise model (sim2real gap). Default = perfect actuator
    /// (off); self-play populates it from the environment.
    pub realism: RealismConfig,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            // PHYSICAL CONSTANTS
            gravity: Vector::z() * -9.81 * 1000.0,
            ball_damping: 0.0,
            ball_roll_decel: 1310.0,
            vision_update_step: 1.0 / 40.0,

            // ROBOT MODEL PARAMETERS
            player_radius: 90.0,
            player_height: 140.0,
            dribbler_radius: BALL_RADIUS + 120.0,
            dribbler_angle: PI / 6.0,
            // Calibrated against the 2026-07-02 real match logs: real launch
            // speed p50 ≈ 4310 mm/s; 300k launched at ≈ 3960.
            kicker_strength: 326000.0,
            player_cmd_timeout: 0.1,
            dribbler_strength: 0.6,
            command_delay: 30.0 / 1000.0,
            // Match real-hardware limits (measured): ~2.8 m/s top speed,
            // ~1.5 m/s^2 to accelerate but much harder braking (~6 m/s^2), so
            // the robot is sluggish to start yet stops without overshoot. The
            // controller may *command* more aggressively; the simulated robot,
            // like the real one, can't exceed these.
            max_accel: 1500.0,
            max_decel: 6000.0,
            max_vel: 2800.0,
            max_ang_accel: 50.0 * 720.0f64.to_radians(),
            max_ang_vel: 0.5 * 720.0f64.to_radians(),
            velocity_treshold: 1.0,
            angular_velocity_treshold: 0.0,
            feedback_interval: 0.05,
            has_imu: true,

            // FIELD GEOMRTY PARAMETERS
            field_geometry: FieldGeometry::default(),
            geometry_interval: 3.0,

            // TEAM CONFIGURATION
            blue_controlled: true,
            yellow_controlled: false,
            // Must match what the sim reports via `blue_team_on_positive_half`
            // (and therefore how the executor/strategies place and play the
            // teams), otherwise goal attribution and penalty checks are computed
            // against the opposite side from reality.
            initial_side_assignment: SideAssignment::YellowOnPositive,

            n_blue_robots: 6,
            n_yellow_robots: 6,

            seed: 0,
            realism: RealismConfig::default(),
        }
    }
}
