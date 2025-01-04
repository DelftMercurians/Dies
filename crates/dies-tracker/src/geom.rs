use dies_core::Vector2;
use dies_core::{FieldCircularArc, FieldGeometry, FieldLineSegment};
use dies_protos::ssl_vision_geometry::SSL_GeometryFieldSize;

pub trait FromProtobuf<T> {
    fn from_protobuf(geometry: &T) -> Self;
}

impl FromProtobuf<SSL_GeometryFieldSize> for FieldGeometry {
    fn from_protobuf(geometry: &SSL_GeometryFieldSize) -> Self {
        // Get field lines
        let mut field_line_segments = Vec::new();
        for line in geometry.field_lines.iter() {
            let p1 = if let Some(p1) = line.p1.as_ref() {
                Vector2::new(p1.x() as f64, p1.y() as f64)
            } else {
                log::error!("Field line segment has no p1");
                continue;
            };

            let p2 = if let Some(p2) = line.p2.as_ref() {
                Vector2::new(p2.x() as f64, p2.y() as f64)
            } else {
                log::error!("Field line segment has no p2");
                continue;
            };
            field_line_segments.push(FieldLineSegment {
                name: line.name().to_owned(),
                p1,
                p2,
                thickness: line.thickness() as f64,
            });
        }

        // Get field arcs
        let mut field_circular_arcs = Vec::new();
        for arc in geometry.field_arcs.iter() {
            let center = if let Some(center) = arc.center.as_ref() {
                Vector2::new(center.x() as f64, center.y() as f64)
            } else {
                log::error!("Field circular arc has no center");
                continue;
            };

            field_circular_arcs.push(FieldCircularArc {
                name: arc.name().to_owned(),
                center,
                radius: arc.radius() as f64,
                a1: arc.a1() as f64,
                a2: arc.a2() as f64,
                thickness: arc.thickness() as f64,
            });
        }

        FieldGeometry {
            field_length: geometry.field_length() as f64,
            field_width: geometry.field_width() as f64,
            goal_width: geometry.goal_width() as f64,
            goal_depth: geometry.goal_depth() as f64,
            boundary_width: geometry.boundary_width() as f64,
            line_segments: field_line_segments,
            circular_arcs: field_circular_arcs,
            penalty_area_depth: geometry.penalty_area_depth() as f64,
            penalty_area_width: geometry.penalty_area_width() as f64,
            center_circle_radius: geometry.center_circle_radius() as f64,
            goal_line_to_penalty_mark: geometry.goal_center_to_penalty_mark() as f64,
            ball_radius: geometry.ball_radius() as f64,
        }
    }
}
