use crate::FieldGeometry;
use serde::{Deserialize, Serialize};

/// A field mask for the `WorldTracker`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldMask {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

impl FieldMask {
    pub fn contains(&self, x: f32, y: f32, field_geom: Option<&FieldGeometry>) -> bool {
        if let Some(field_geom) = field_geom {
            let x = x as f64;
            let y = y as f64;
            let x_min = (field_geom.field_length / 2.0 + field_geom.boundary_width) * self.x_min;
            let x_max = (field_geom.field_length / 2.0 + field_geom.boundary_width) * self.x_max;
            let y_min = (field_geom.field_width / 2.0 + field_geom.boundary_width) * self.y_min;
            let y_max = (field_geom.field_width / 2.0 + field_geom.boundary_width) * self.y_max;
            x >= x_min && x <= x_max && y >= y_min && y <= y_max
        } else {
            false
        }
    }
}

impl Default for FieldMask {
    fn default() -> Self {
        Self {
            x_min: -1.0,
            x_max: 1.0,
            y_min: -1.0,
            y_max: 1.0,
        }
    }
}

/// Settings for the `WorldTracker`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackerSettings {
    /// Whether our team color is blue
    pub is_blue: bool,
    /// The initial sign of the enemy goal's x coordinate in ssl-vision coordinates.
    pub initial_opp_goal_x: f64,

    pub field_mask: FieldMask,

    /// Transition variance for the player Kalman filter.
    pub player_unit_transition_var: f64,
    /// Measurement variance for the player Kalman filter.
    pub player_measurement_var: f64,
    /// Smoothinfg factor for the yaw LPF
    pub player_yaw_lpf_alpha: f64,

    /// Transition variance for the ball Kalman filter.
    pub ball_unit_transition_var: f64,
    /// Measurement variance for the ball Kalman filter.
    pub ball_measurement_var: f64,
}

impl Default for TrackerSettings {
    fn default() -> Self {
        Self {
            is_blue: true,
            initial_opp_goal_x: 1.0,
            player_unit_transition_var: 95.75,
            player_measurement_var: 0.01,
            player_yaw_lpf_alpha: 0.15,
            ball_unit_transition_var: 20.48,
            ball_measurement_var: 0.01,
            field_mask: FieldMask::default(),
        }
    }
}
