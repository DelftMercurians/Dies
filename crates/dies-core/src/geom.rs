use dies_protos::ssl_vision_geometry::SSL_GeometryFieldSize;
use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::Vector2;

/// A single field arc -- eg. the center circle
#[derive(Serialize, Deserialize, Clone, Debug)]
#[typeshare]
pub struct FieldCircularArc {
    /// Readable name of the arc
    pub name: String,
    /// Center of the arc, in dies coordinates
    pub center: Vector2,
    // Radius of the arc, in mm
    pub radius: f64,
    // Start angle in counter-clockwise order, in radians
    pub a1: f64,
    // End angle in counter-clockwise order, in radians
    pub a2: f64,
    // Thickness of the arc stroke, in mm
    pub thickness: f64,
}

impl FieldCircularArc {
    pub fn new(name: &str, center: Vector2, radius: f64, a1: f64, a2: f64, thickness: f64) -> Self {
        Self {
            name: name.to_owned(),
            center,
            radius,
            a1,
            a2,
            thickness,
        }
    }
}

/// A single field line segment
#[derive(Serialize, Deserialize, Clone, Debug)]
#[typeshare]
pub struct FieldLineSegment {
    /// Readable name of the line segment
    pub name: String,
    /// Start Vector2 of the line segment, in dies coordinates
    pub p1: Vector2,
    /// End Vector2 of the line segment, in dies coordinates
    pub p2: Vector2,
    /// Thickness of the line segment, in mm
    pub thickness: f64,
}

impl FieldLineSegment {
    pub fn new(name: &str, p1: Vector2, p2: Vector2, thickness: f64) -> Self {
        Self {
            name: name.to_owned(),
            p1,
            p2,
            thickness,
        }
    }
}

/// The field geometry.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[typeshare]
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

impl Default for FieldGeometry {
    fn default() -> Self {
        let lines = vec![
            FieldLineSegment::new(
                "TopTouchLine",
                Vector2::new(-6020.0, -4510.0),
                Vector2::new(6020.0, -4510.0),
                10.0,
            ),
            FieldLineSegment::new(
                "BottomTouchLine",
                Vector2::new(-6020.0, 4510.0),
                Vector2::new(6020.0, 4510.0),
                10.0,
            ),
            FieldLineSegment::new(
                "LeftGoalLine",
                Vector2::new(-6020.0, -4510.0),
                Vector2::new(-6020.0, 4510.0),
                10.0,
            ),
            FieldLineSegment::new(
                "RightGoalLine",
                Vector2::new(6020.0, -4510.0),
                Vector2::new(6020.0, 4510.0),
                10.0,
            ),
            FieldLineSegment::new(
                "HalfwayLine",
                Vector2::new(0.0, -4510.0),
                Vector2::new(0.0, 4510.0),
                10.0,
            ),
            FieldLineSegment::new(
                "CenterLine",
                Vector2::new(-6020.0, 0.0),
                Vector2::new(6020.0, 0.0),
                10.0,
            ),
            FieldLineSegment::new(
                "LeftPenaltyStretch",
                Vector2::new(-4800.0, -1205.0),
                Vector2::new(-4800.0, 1205.0),
                10.0,
            ),
            FieldLineSegment::new(
                "RightPenaltyStretch",
                Vector2::new(4800.0, -1205.0),
                Vector2::new(4800.0, 1205.0),
                10.0,
            ),
            FieldLineSegment::new(
                "LeftFieldLeftPenaltyStretch",
                Vector2::new(-6020.0, -1205.0),
                Vector2::new(-4800.0, -1205.0),
                10.0,
            ),
            FieldLineSegment::new(
                "LeftFieldRightPenaltyStretch",
                Vector2::new(-6020.0, 1205.0),
                Vector2::new(-4800.0, 1205.0),
                10.0,
            ),
            FieldLineSegment::new(
                "RightFieldRightPenaltyStretch",
                Vector2::new(6020.0, -1205.0),
                Vector2::new(4800.0, -1205.0),
                10.0,
            ),
            FieldLineSegment::new(
                "RightFieldLeftPenaltyStretch",
                Vector2::new(6020.0, 1205.0),
                Vector2::new(4800.0, 1205.0),
                10.0,
            ),
        ];
        let arcs = vec![FieldCircularArc::new(
            "CenterCircle",
            Vector2::new(0.0, 0.0),
            500.0,
            0.0,
            6.283185,
            10.0,
        )];

        Self {
            field_width: 9020,
            field_length: 12040,
            boundary_width: 300,
            goal_depth: 180,
            goal_width: 1200,
            line_segments: lines,
            circular_arcs: arcs,
        }
    }
}
