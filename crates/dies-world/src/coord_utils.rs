use dies_core::{Angle, Vector2, Vector3};

/// Convert ssl-vision coordinates to dies coordinates.
///
/// The x coordinate is flipped if the goal is on the left.
pub fn to_dies_coords2(pos: Vector2, opp_goal_sign: f64) -> Vector2 {
    Vector2::new(pos.x * opp_goal_sign, pos.y)
}

/// Convert ssl-vision yaw to dies yaw.
///
/// The yaw is rotated by 180 degrees if the goal is on the left.
pub fn to_dies_yaw(yaw: Angle, opp_goal_sign: f64) -> Angle {
    if opp_goal_sign < 0.0 {
        Angle::PI - yaw
    } else {
        yaw
    }
}

/// Convert ssl-vision coordinates to dies coordinates.
///
/// The x coordinate is flipped if the goal is on the left.
pub fn to_dies_coords3(pos: Vector3, opp_goal_sign: f64) -> Vector3 {
    Vector3::new(pos.x * opp_goal_sign, pos.y, pos.z)
}
