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
