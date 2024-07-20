use crate::{Angle, Vector2, Vector3, WorldData};

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
    //     if (angle >= 0)
    //       return pi - angle
    //    else
    //       return -pi - angle
    if opp_goal_sign > 0.0 {
        yaw
    } else {
        // Invert around y-axis
        if yaw.radians() >= 0.0 {
            Angle::from_radians(std::f64::consts::PI - yaw.radians())
        } else {
            Angle::from_radians(-std::f64::consts::PI - yaw.radians())
        }
    }
}

/// Convert ssl-vision coordinates to dies coordinates.
///
/// The x coordinate is flipped if the goal is on the left.
pub fn to_dies_coords3(pos: Vector3, opp_goal_sign: f64) -> Vector3 {
    Vector3::new(pos.x * opp_goal_sign, pos.y, pos.z)
}

/// Finds the intersection point of two lines.
///
/// Each line is defined by a point and a direction vector. Returns None if the lines
/// are parallel.
pub fn find_intersection(
    point1: Vector2,
    direction1: Vector2,
    point2: Vector2,
    direction2: Vector2,
) -> Option<Vector2> {
    let det = direction1.x * direction2.y - direction1.y * direction2.x;
    if det.abs() < 1e-10 {
        return None;
    }

    let dp = point2 - point1;
    let t = (dp.x * direction2.y - dp.y * direction2.x) / det;
    let intersection = point1 + t * direction1;

    Some(intersection)
}

/// Returns a vector perpendicular to the given vector.
pub fn perp(v: Vector2) -> Vector2 {
    Vector2::new(-v.y, v.x)
}

/// calculate if the robot is on the left or right side of the target point
/// return true iff the robot can directly move to the target point without interference the ball
pub fn which_side_of_robot(heading: Angle, target_pos: Vector2, robot_pos: Vector2) -> bool {
    let heading_v = Angle::to_vector(&heading);
    let heading_v_tangent = perp(heading_v);
    cross_product(heading_v_tangent, robot_pos - target_pos) >= -10.0
}

/// given a circle and a point, return the two tangent line directions
pub fn get_tangent_line_direction(
    circle_center: Vector2,
    circle_radius: f64,
    point: Vector2,
) -> (Angle, Angle) {
    let direction: Vector2 = circle_center - point;
    let distance = direction.norm();
    let angle = Angle::from_radians(direction.y.atan2(direction.x));
    let v = Angle::from_radians((circle_radius / distance).asin());
    let angle1 = angle + v;
    let angle2 = angle - v;
    (angle1, angle2)
}

pub fn cross_product(v1: Vector2, v2: Vector2) -> f64 {
    v1.x * v2.y - v1.y * v2.x
}

pub fn distance_to_line(a: Vector2, b: Vector2, p: Vector2) -> f64 {
    let n = (b - a).normalize();
    let ap = p - a;
    let proj = ap.dot(&n);
    let proj = proj.max(0.0).min((b - a).norm());
    (ap - proj * n).norm()
}

/// Compute a "badness" score for a line of sight between two points based on the minumum
/// distance to the line of sight from the closest enemy player.
///
/// The score is higher if the line of sight is further from the enemy players.
pub fn score_line_of_sight(world: &WorldData, from: Vector2, to: Vector2) -> f64 {
    let mut min_distance = f64::MAX;
    for player in world.opp_players.iter() {
        let distance = distance_to_line(from, to, player.position);
        if distance < min_distance {
            min_distance = distance;
        }
    }
    min_distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn test_which_side_of_robot() {
        let heading = Angle::from_degrees(90.0);
        let target_pos = Vector2::new(1.0, 1.0);
        let robot_pos = Vector2::new(-225.0, -215.0);
        let result = which_side_of_robot(heading, target_pos, robot_pos);
        assert!(result);
    }
    #[test]
    fn test_get_tengent_line_direction() {
        let circle_center = Vector2::new(1.0, 1.0);
        let circle_radius = 1.0;
        let point = Vector2::new(2.0, 2.0);
        let (angle1, angle2) = get_tangent_line_direction(circle_center, circle_radius, point);
        assert_relative_eq!(angle1.degrees(), -90.0, epsilon = 1e-10);
        assert_relative_eq!(angle2.degrees(), 180.0, epsilon = 1e-10);
    }
    #[test]
    fn test_intersecting_lines() {
        let point1 = Vector2::new(0.0, 0.0);
        let direction1 = Vector2::new(1.0, 1.0);
        let point2 = Vector2::new(2.0, 0.0);
        let direction2 = Vector2::new(-1.0, 1.0);

        let intersection = find_intersection(point1, direction1, point2, direction2).unwrap();
        assert_relative_eq!(intersection.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(intersection.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parallel_lines() {
        let point1 = Vector2::new(0.0, 0.0);
        let direction1 = Vector2::new(1.0, 1.0);
        let point2 = Vector2::new(1.0, 1.0);
        let direction2 = Vector2::new(1.0, 1.0);

        let intersection = find_intersection(point1, direction1, point2, direction2);
        assert!(intersection.is_none());
    }

    #[test]
    fn test_coincident_lines() {
        let point1 = Vector2::new(0.0, 0.0);
        let direction1 = Vector2::new(1.0, 1.0);
        let point2 = Vector2::new(1.0, 1.0);
        let direction2 = Vector2::new(2.0, 2.0);

        let intersection = find_intersection(point1, direction1, point2, direction2);
        assert!(intersection.is_none());
    }

    #[test]
    fn test_perpendicular_lines() {
        let point1 = Vector2::new(0.0, 0.0);
        let direction1 = Vector2::new(1.0, 0.0);
        let point2 = Vector2::new(1.0, 1.0);
        let direction2 = Vector2::new(0.0, -1.0);

        let intersection = find_intersection(point1, direction1, point2, direction2).unwrap();
        assert_relative_eq!(intersection.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(intersection.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_almost_parallel_lines() {
        let point1 = Vector2::new(0.0, 0.0);
        let direction1 = Vector2::new(1.0, 1.0);
        let point2 = Vector2::new(1.0, 0.0);
        let direction2 = Vector2::new(1.0, 1.000001);

        let intersection = find_intersection(point1, direction1, point2, direction2);
        assert!(intersection.is_some());
    }

    #[test]
    fn test_invert_angle() {
        let tests = vec![
            (0.0, 180.0),
            (90.0, 90.0),
            (-90.0, -90.0),
            (180.0, 0.0),
            (-180.0, 0.0),
            (45.0, 135.0),
            (-45.0, -135.0),
            (135.0, 45.0),
            (-135.0, -45.0),
        ];

        for (input, expected) in tests {
            let angle = Angle::from_degrees(input);
            let inverted = to_dies_yaw(angle, -1.0);
            assert_relative_eq!(inverted.degrees(), expected, epsilon = 1e-10);
        }
    }
}
