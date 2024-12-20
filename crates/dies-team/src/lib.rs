mod control;
mod gc_client;
mod player_override;
mod skills;
mod strategy_instance;
mod team_frame;
mod team_map;
mod utils;

pub use control::{KickerControlInput, PlayerControlInput, PlayerInputs};
use dies_core::{distance_to_line, Angle, Avoid, ControllerSettings, FieldGeometry, Vector2};
use serde::{Deserialize, Serialize};
use typeshare_annotation::typeshare;

// Enum to represent different obstacle types
#[derive(Debug, Clone, Serialize)]
pub enum Obstacle {
    Circle { center: Vector2, radius: f64 },
    Rectangle { min: Vector2, max: Vector2 },
}

pub fn nearest_safe_pos(
    avoding_point: Avoid,
    min_distance: f64,
    initial_pos: Vector2,
    target_pos: Vector2,
    max_radius: i32,
    field: &FieldGeometry,
) -> Vector2 {
    let mut best_pos = Vector2::new(f64::INFINITY, f64::INFINITY);
    let mut found_better = false;
    let min_theta = 0;
    let max_theta = 360;
    let mut i = 0;
    for theta in (min_theta..max_theta).step_by(10) {
        let theta = Angle::from_degrees(theta as f64);
        for radius in (0..max_radius).step_by(50) {
            let position = initial_pos + theta.to_vector() * (radius as f64);
            if is_pos_in_field(position, field)
                && avoding_point.distance_to(position) > min_distance
            {
                if (position - target_pos).norm() < (best_pos - target_pos).norm() {
                    // crate::debug_cross(format!("{i}"), position, crate::DebugColor::Green);
                    best_pos = position;
                    found_better = true;
                }
            } else {
                // crate::debug_cross(format!("{i}"), position, crate::DebugColor::Red);
            }
            i += 1;
        }
    }
    if !found_better {
        log::warn!("Could not find a safe position from {initial_pos}, avoiding {avoding_point:?}");
    }

    best_pos
}

pub fn is_pos_in_field(pos: Vector2, field: &FieldGeometry) -> bool {
    const MARGIN: f64 = 100.0;
    // check if pos outside field
    if pos.x.abs() > field.field_length / 2.0 - MARGIN
        || pos.y.abs() > field.field_width / 2.0 - MARGIN
    {
        return false;
    }

    true
}

#[derive(Debug, Clone)]
pub enum Avoid {
    Line { start: Vector2, end: Vector2 },
    Circle { center: Vector2 },
}

impl Avoid {
    fn distance_to(&self, pos: Vector2) -> f64 {
        match self {
            Avoid::Line { start, end } => distance_to_line(*start, *end, pos),
            Avoid::Circle { center } => (center - pos).norm(),
        }
    }
}

/// Settings for the low-level controller.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct ControllerSettings {
    /// Maximum acceleration of the robot in mm/s².
    pub max_acceleration: f64,
    /// Maximum velocity of the robot in mm/s.
    pub max_velocity: f64,
    /// Maximum deceleration of the robot in mm/s².
    pub max_deceleration: f64,
    /// Maximum angular velocity of the robot in rad/s.
    pub max_angular_velocity: f64,
    /// Maximum angular acceleration of the robot in rad/s².
    pub max_angular_acceleration: f64,
    /// Proportional gain for the close-range position controller.
    pub position_kp: f64,
    /// Time until destination in which the proportional controller is used, in seconds.
    pub position_proportional_time_window: f64,
    /// Distance used as threshold for the controller to prevent shaky behavior
    pub position_cutoff_distance: f64,
    /// Proportional gain for the close-range angle controller.
    pub angle_kp: f64,
    /// Distance used as threshold for the controller to prevent shaky behavior
    pub angle_cutoff_distance: f64,
}

impl Default for ControllerSettings {
    fn default() -> Self {
        Self {
            max_acceleration: 125000.0,
            max_velocity: 3120.0,
            max_deceleration: 6340.0,
            max_angular_velocity: 25.132741228718345,
            max_angular_acceleration: 349.0658503988659,
            position_kp: 50000.0,
            position_proportional_time_window: 1.1,
            position_cutoff_distance: 70.0,
            angle_kp: 2.8,
            angle_cutoff_distance: 0.03490658503988659,
        }
    }
}
