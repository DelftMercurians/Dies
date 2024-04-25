use std::f32::consts::PI;
use nalgebra::{Vector2, Vector3};

/// Convert ssl-vision coordinates to dies coordinates.
///
/// The x coordinate is flipped if the goal is on the left.
pub fn to_dies_coords2(x: f32, y: f32, opp_goal_sign: f32) -> Vector2<f32> {
    Vector2::new(x * opp_goal_sign, y)
}

/// Convert ssl-vision coordinates to dies coordinates.
///
/// The x coordinate is flipped if the goal is on the left.
pub fn to_dies_coords3(x: f32, y: f32, z: f32, opp_goal_sign: f32) -> Vector3<f32> {
    Vector3::new(x * opp_goal_sign, y, z)
}


pub fn to_dies_coords_angle(x: f32, opp_goal_sign: f32) -> f32 {
    if opp_goal_sign == 1.0 {
        x
    } else if (x > 0.0){
        PI - x
    }
    else {
        -PI - x
    }
}