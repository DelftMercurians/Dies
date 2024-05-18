use dies_core::{Vector2, Vector3};

/// Convert ssl-vision coordinates to dies coordinates.
///
/// The x coordinate is flipped if the goal is on the left.
pub fn to_dies_coords2(x: f32, y: f32, opp_goal_sign: f64) -> Vector2 {
    let x = x as f64;
    let y = y as f64;
    Vector2::new(x * opp_goal_sign, y)
}

/// Convert ssl-vision coordinates to dies coordinates.
///
/// The x coordinate is flipped if the goal is on the left.
pub fn to_dies_coords3(x: f32, y: f32, z: f32, opp_goal_sign: f64) -> Vector3 {
    let x = x as f64;
    let y = y as f64;
    let z = z as f64;
    Vector3::new(x * opp_goal_sign, y, z)
}
