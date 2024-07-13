use crate::Vector2;

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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
}
