use dies_core::Vector2;

pub(crate) fn rotate_vector(v: Vector2, angle: f64) -> Vector2 {
    let rot = nalgebra::Rotation2::new(angle);
    rot * v
}
