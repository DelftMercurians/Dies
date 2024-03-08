use dies_protos::ssl_vision_geometry::SSL_GeometryFieldSize;
use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

/// A single field arc -- eg. the center circle
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FieldCircularArc {
    /// Readable name of the arc
    pub name: String,
    /// Center of the arc, in dies coordinates
    pub center: Vector2<f32>,
    // Radius of the arc, in mm
    pub radius: f32,
    // Start angle in counter-clockwise order, in radians
    pub a1: f32,
    // End angle in counter-clockwise order, in radians
    pub a2: f32,
    // Thickness of the arc stroke, in mm
    pub thickness: f32,
}

/// A single field line segment
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FieldLineSegment {
    /// Readable name of the line segment
    pub name: String,
    /// Start point of the line segment, in dies coordinates
    pub p1: Vector2<f32>,
    /// End point of the line segment, in dies coordinates
    pub p2: Vector2<f32>,
    /// Thickness of the line segment, in mm
    pub thickness: f32,
}

/// The field geometry.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FieldGeometry {
    /// Field length (distance between goal lines) in mm
    pub field_length: i32,
    /// Field width (distance between touch lines) in mm
    pub field_width: i32,
    /// Goal width (distance inner edges of goal posts) in mm
    pub goal_width: i32,
    /// Goal depth (distance from outer goal line edge to inner goal back) in mm
    pub goal_depth: i32,
    /// Boundary width (distance from touch/goal line centers to boundary walls) in mm
    pub boundary_width: i32,
    /// Generated line segments based on the other parameters
    pub line_segments: Vec<FieldLineSegment>,
    /// Generated circular arcs based on the other parameters
    pub circular_arcs: Vec<FieldCircularArc>,
}

impl FieldGeometry {
    /// Create a new FieldGeometry from a protobuf message.
    pub fn from_protobuf(geometry: &SSL_GeometryFieldSize) -> Self {
        // Get field lines
        let mut field_line_segments = Vec::new();
        for line in geometry.field_lines.iter() {
            let p1 = if let Some(p1) = line.p1.as_ref() {
                Vector2::new(p1.x(), p1.y())
            } else {
                tracing::error!("Field line segment has no p1");
                continue;
            };

            let p2 = if let Some(p2) = line.p2.as_ref() {
                Vector2::new(p2.x(), p2.y())
            } else {
                tracing::error!("Field line segment has no p2");
                continue;
            };
            field_line_segments.push(FieldLineSegment {
                name: line.name().to_owned(),
                p1,
                p2,
                thickness: line.thickness(),
            });
        }

        // Get field arcs
        let mut field_circular_arcs = Vec::new();
        for arc in geometry.field_arcs.iter() {
            let center = if let Some(center) = arc.center.as_ref() {
                Vector2::new(center.x(), center.y())
            } else {
                tracing::error!("Field circular arc has no center");
                continue;
            };

            field_circular_arcs.push(FieldCircularArc {
                name: arc.name().to_owned(),
                center,
                radius: arc.radius(),
                a1: arc.a1(),
                a2: arc.a2(),
                thickness: arc.thickness(),
            });
        }

        FieldGeometry {
            field_length: geometry.field_length(),
            field_width: geometry.field_width(),
            goal_width: geometry.goal_width(),
            goal_depth: geometry.goal_depth(),
            boundary_width: geometry.boundary_width(),
            line_segments: field_line_segments,
            circular_arcs: field_circular_arcs,
        }
    }
}
