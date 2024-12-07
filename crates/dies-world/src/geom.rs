use std::f64::consts::PI;

use dies_protos::ssl_vision_geometry::SSL_GeometryFieldSize;
use serde::{Deserialize, Serialize};

use crate::Vector2;

/// A single field arc -- eg. the center circle
#[derive(Serialize, Deserialize, Clone, Debug)]
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
pub struct FieldGeometry {
    /// Field length (distance between goal lines) in mm
    pub field_length: f64,
    /// Field width (distance between touch lines) in mm
    pub field_width: f64,
    /// Goal width (distance inner edges of goal posts) in mm
    pub goal_width: f64,
    /// Goal depth (distance from outer goal line edge to inner goal back) in mm
    pub goal_depth: f64,
    /// Boundary width (distance from touch/goal line centers to boundary walls) in mm
    pub boundary_width: f64,
    /// Generated line segments based on the other parameters
    pub line_segments: Vec<FieldLineSegment>,
    /// Generated circular arcs based on the other parameters
    pub circular_arcs: Vec<FieldCircularArc>,
    /// Penalty area depth (distance from goal line to penalty mark) in mm
    pub penalty_area_depth: f64,
    /// Penalty area width (distance from penalty mark to penalty area edge) in mm
    pub penalty_area_width: f64,
    /// Center circle radius in mm
    pub center_circle_radius: f64,
    /// Distance from goal line to penalty mark in mm
    pub goal_line_to_penalty_mark: f64,
    /// Ball radius in mm
    pub ball_radius: f64,
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

impl Default for FieldGeometry {
    fn default() -> Self {
        // Division B constants
        // Sourced from https://github.com/DelftMercurians/ssl-vision/blob/fd7b426df1b19a183ed2a6bdad7d7448ad1a9832/src/shared/util/field_default_constants.h#L50-L66
        // and https://github.com/DelftMercurians/ssl-vision/blob/fd7b426df1b19a183ed2a6bdad7d7448ad1a9832/src/shared/util/field.cpp#L681
        let field_length = 9000.0f64;
        let field_width = 6000.0f64;
        let goal_width = 1000.0f64;
        let goal_depth = 180.0f64;
        let boundary_width = 300.0f64;
        let line_thickness = 10.0f64;
        let penalty_area_depth = 1000.0f64;
        let penalty_area_width = 2000.0f64;
        let center_circle_radius = 500.0f64;
        let goal_line_to_penalty_mark = 6000.0f64;
        let ball_radius = 21.5;

        let field_length_half = field_length / 2.0;
        let field_width_half = field_width / 2.0;
        let pen_area_x = field_length_half - penalty_area_depth;
        let pen_area_y = penalty_area_width / 2.0;

        let line_segments = vec![
            FieldLineSegment {
                name: "TopTouchLine".to_string(),
                p1: Vector2::new(-field_length_half, field_width_half),
                p2: Vector2::new(field_length_half, field_width_half),
                thickness: line_thickness,
            },
            FieldLineSegment {
                name: "BottomTouchLine".to_string(),
                p1: Vector2::new(-field_length_half, -field_width_half),
                p2: Vector2::new(field_length_half, -field_width_half),
                thickness: line_thickness,
            },
            FieldLineSegment {
                name: "LeftGoalLine".to_string(),
                p1: Vector2::new(-field_length_half, -field_width_half),
                p2: Vector2::new(-field_length_half, field_width_half),
                thickness: line_thickness,
            },
            FieldLineSegment {
                name: "RightGoalLine".to_string(),
                p1: Vector2::new(field_length_half, -field_width_half),
                p2: Vector2::new(field_length_half, field_width_half),
                thickness: line_thickness,
            },
            FieldLineSegment {
                name: "HalfwayLine".to_string(),
                p1: Vector2::new(0.0, -field_width_half),
                p2: Vector2::new(0.0, field_width_half),
                thickness: line_thickness,
            },
            FieldLineSegment {
                name: "CenterLine".to_string(),
                p1: Vector2::new(-field_length_half, 0.0),
                p2: Vector2::new(field_length_half, 0.0),
                thickness: line_thickness,
            },
            FieldLineSegment {
                name: "LeftPenaltyStretch".to_string(),
                p1: Vector2::new(-pen_area_x, -pen_area_y),
                p2: Vector2::new(-pen_area_x, pen_area_y),
                thickness: line_thickness,
            },
            FieldLineSegment {
                name: "RightPenaltyStretch".to_string(),
                p1: Vector2::new(pen_area_x, -pen_area_y),
                p2: Vector2::new(pen_area_x, pen_area_y),
                thickness: line_thickness,
            },
            FieldLineSegment {
                name: "LeftFieldLeftPenaltyStretch".to_string(),
                p1: Vector2::new(-field_length_half, -pen_area_y),
                p2: Vector2::new(-pen_area_x, -pen_area_y),
                thickness: line_thickness,
            },
            FieldLineSegment {
                name: "LeftFieldRightPenaltyStretch".to_string(),
                p1: Vector2::new(-field_length_half, pen_area_y),
                p2: Vector2::new(-pen_area_x, pen_area_y),
                thickness: line_thickness,
            },
            FieldLineSegment {
                name: "RightFieldRightPenaltyStretch".to_string(),
                p1: Vector2::new(field_length_half, -pen_area_y),
                p2: Vector2::new(pen_area_x, -pen_area_y),
                thickness: line_thickness,
            },
            FieldLineSegment {
                name: "RightFieldLeftPenaltyStretch".to_string(),
                p1: Vector2::new(field_length_half, pen_area_y),
                p2: Vector2::new(pen_area_x, pen_area_y),
                thickness: line_thickness,
            },
        ];

        let circular_arcs = vec![FieldCircularArc {
            name: "CenterCircle".to_string(),
            center: Vector2::zeros(),
            radius: center_circle_radius,
            a1: 0.0,
            a2: 2.0 * PI,
            thickness: line_thickness,
        }];

        Self {
            field_length,
            field_width,
            goal_width,
            goal_depth,
            boundary_width,
            line_segments,
            circular_arcs,
            penalty_area_depth,
            penalty_area_width,
            center_circle_radius,
            goal_line_to_penalty_mark,
            ball_radius,
        }
    }
}
