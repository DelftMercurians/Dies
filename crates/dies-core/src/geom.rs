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

/// A single field line segment
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FieldLineSegment {
    /// Readable name of the line segment
    pub name: String,
    /// Start point of the line segment, in dies coordinates
    pub p1: Vector2,
    /// End point of the line segment, in dies coordinates
    pub p2: Vector2,
    /// Thickness of the line segment, in mm
    pub thickness: f64,
}

impl FieldLineSegment {
    pub fn new(name: String, p1: Vector2, p2: Vector2, thickness: f64) -> Self {
        Self {
            name,
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
    // <Var name="Field Length" type="double" minval="" maxval="">
    //     12040.000000
    // </Var>
    // <Var name="Field Width" type="double" minval="" maxval="">
    //     9020.000000
    // </Var>
    // <Var name="Goal Width" type="double" minval="" maxval="">
    //     1200.000000
    // </Var>
    // <Var name="Goal Depth" type="double" minval="" maxval="">
    //     180.000000
    // </Var>
    // <Var name="Goal Height" type="double" minval="" maxval="">
    //     155.000000
    // </Var>
    // <Var name="Boundary Width" type="double" minval="" maxval="">
    //     300.000000
    // </Var>
    // <Var name="Line Thickness" type="double" minval="" maxval="">
    //     10.000000
    // </Var>
    // <Var name="Penalty Area Depth" type="double" minval="" maxval="">
    //     1220.000000
    // </Var>
    // <Var name="Penalty Area Width" type="double" minval="" maxval="">
    //     2410.000000
    // </Var>
    // <Var name="Goal Line to Penalty Mark" type="double" minval="" maxval="">
    //     8000.000000
    // </Var>
    // <Var name="Center Circle Radius" type="double" minval="" maxval="">
    //     500.000000
    // </Var>
    // <Var name="Field Lines" type="list">
    // 			<Var name="TopTouchLine" type="list">
    // 				<Var name="Name" type="string">
    // 					TopTouchLine
    // 				</Var>
    // 				<Var name="Type" type="stringenum">
    // 					TopTouchLine
    // 					<Var name="0" type="string">
    // 						Undefined
    // 					</Var>
    // 					<Var name="1" type="string">
    // 						TopTouchLine
    // 					</Var>
    // 					<Var name="2" type="string">
    // 						BottomTouchLine
    // 					</Var>
    // 					<Var name="3" type="string">
    // 						LeftGoalLine
    // 					</Var>
    // 					<Var name="4" type="string">
    // 						RightGoalLine
    // 					</Var>
    // 					<Var name="5" type="string">
    // 						HalfwayLine
    // 					</Var>
    // 					<Var name="6" type="string">
    // 						CenterLine
    // 					</Var>
    // 					<Var name="7" type="string">
    // 						LeftPenaltyStretch
    // 					</Var>
    // 					<Var name="8" type="string">
    // 						RightPenaltyStretch
    // 					</Var>
    // 					<Var name="9" type="string">
    // 						LeftFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="10" type="string">
    // 						LeftFieldRightPenaltyStretch
    // 					</Var>
    // 					<Var name="11" type="string">
    // 						RightFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="12" type="string">
    // 						RightFieldRightPenaltyStretch
    // 					</Var>
    // 				</Var>
    // 				<Var name="P1.x" type="double" minval="" maxval="">
    // 					-6020.000000
    // 				</Var>
    // 				<Var name="P1.y" type="double" minval="" maxval="">
    // 					4510.000000
    // 				</Var>
    // 				<Var name="P2.x" type="double" minval="" maxval="">
    // 					6020.000000
    // 				</Var>
    // 				<Var name="P2.y" type="double" minval="" maxval="">
    // 					4510.000000
    // 				</Var>
    // 				<Var name="Line thickness" type="double" minval="" maxval="">
    // 					10.000000
    // 				</Var>
    // 			</Var>
    // 			<Var name="BottomTouchLine" type="list">
    // 				<Var name="Name" type="string">
    // 					BottomTouchLine
    // 				</Var>
    // 				<Var name="Type" type="stringenum">
    // 					BottomTouchLine
    // 					<Var name="0" type="string">
    // 						Undefined
    // 					</Var>
    // 					<Var name="1" type="string">
    // 						TopTouchLine
    // 					</Var>
    // 					<Var name="2" type="string">
    // 						BottomTouchLine
    // 					</Var>
    // 					<Var name="3" type="string">
    // 						LeftGoalLine
    // 					</Var>
    // 					<Var name="4" type="string">
    // 						RightGoalLine
    // 					</Var>
    // 					<Var name="5" type="string">
    // 						HalfwayLine
    // 					</Var>
    // 					<Var name="6" type="string">
    // 						CenterLine
    // 					</Var>
    // 					<Var name="7" type="string">
    // 						LeftPenaltyStretch
    // 					</Var>
    // 					<Var name="8" type="string">
    // 						RightPenaltyStretch
    // 					</Var>
    // 					<Var name="9" type="string">
    // 						LeftFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="10" type="string">
    // 						LeftFieldRightPenaltyStretch
    // 					</Var>
    // 					<Var name="11" type="string">
    // 						RightFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="12" type="string">
    // 						RightFieldRightPenaltyStretch
    // 					</Var>
    // 				</Var>
    // 				<Var name="P1.x" type="double" minval="" maxval="">
    // 					-6020.000000
    // 				</Var>
    // 				<Var name="P1.y" type="double" minval="" maxval="">
    // 					-4510.000000
    // 				</Var>
    // 				<Var name="P2.x" type="double" minval="" maxval="">
    // 					6020.000000
    // 				</Var>
    // 				<Var name="P2.y" type="double" minval="" maxval="">
    // 					-4510.000000
    // 				</Var>
    // 				<Var name="Line thickness" type="double" minval="" maxval="">
    // 					10.000000
    // 				</Var>
    // 			</Var>
    // 			<Var name="LeftGoalLine" type="list">
    // 				<Var name="Name" type="string">
    // 					LeftGoalLine
    // 				</Var>
    // 				<Var name="Type" type="stringenum">
    // 					LeftGoalLine
    // 					<Var name="0" type="string">
    // 						Undefined
    // 					</Var>
    // 					<Var name="1" type="string">
    // 						TopTouchLine
    // 					</Var>
    // 					<Var name="2" type="string">
    // 						BottomTouchLine
    // 					</Var>
    // 					<Var name="3" type="string">
    // 						LeftGoalLine
    // 					</Var>
    // 					<Var name="4" type="string">
    // 						RightGoalLine
    // 					</Var>
    // 					<Var name="5" type="string">
    // 						HalfwayLine
    // 					</Var>
    // 					<Var name="6" type="string">
    // 						CenterLine
    // 					</Var>
    // 					<Var name="7" type="string">
    // 						LeftPenaltyStretch
    // 					</Var>
    // 					<Var name="8" type="string">
    // 						RightPenaltyStretch
    // 					</Var>
    // 					<Var name="9" type="string">
    // 						LeftFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="10" type="string">
    // 						LeftFieldRightPenaltyStretch
    // 					</Var>
    // 					<Var name="11" type="string">
    // 						RightFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="12" type="string">
    // 						RightFieldRightPenaltyStretch
    // 					</Var>
    // 				</Var>
    // 				<Var name="P1.x" type="double" minval="" maxval="">
    // 					-6020.000000
    // 				</Var>
    // 				<Var name="P1.y" type="double" minval="" maxval="">
    // 					-4510.000000
    // 				</Var>
    // 				<Var name="P2.x" type="double" minval="" maxval="">
    // 					-6020.000000
    // 				</Var>
    // 				<Var name="P2.y" type="double" minval="" maxval="">
    // 					4510.000000
    // 				</Var>
    // 				<Var name="Line thickness" type="double" minval="" maxval="">
    // 					10.000000
    // 				</Var>
    // 			</Var>
    // 			<Var name="RightGoalLine" type="list">
    // 				<Var name="Name" type="string">
    // 					RightGoalLine
    // 				</Var>
    // 				<Var name="Type" type="stringenum">
    // 					RightGoalLine
    // 					<Var name="0" type="string">
    // 						Undefined
    // 					</Var>
    // 					<Var name="1" type="string">
    // 						TopTouchLine
    // 					</Var>
    // 					<Var name="2" type="string">
    // 						BottomTouchLine
    // 					</Var>
    // 					<Var name="3" type="string">
    // 						LeftGoalLine
    // 					</Var>
    // 					<Var name="4" type="string">
    // 						RightGoalLine
    // 					</Var>
    // 					<Var name="5" type="string">
    // 						HalfwayLine
    // 					</Var>
    // 					<Var name="6" type="string">
    // 						CenterLine
    // 					</Var>
    // 					<Var name="7" type="string">
    // 						LeftPenaltyStretch
    // 					</Var>
    // 					<Var name="8" type="string">
    // 						RightPenaltyStretch
    // 					</Var>
    // 					<Var name="9" type="string">
    // 						LeftFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="10" type="string">
    // 						LeftFieldRightPenaltyStretch
    // 					</Var>
    // 					<Var name="11" type="string">
    // 						RightFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="12" type="string">
    // 						RightFieldRightPenaltyStretch
    // 					</Var>
    // 				</Var>
    // 				<Var name="P1.x" type="double" minval="" maxval="">
    // 					6020.000000
    // 				</Var>
    // 				<Var name="P1.y" type="double" minval="" maxval="">
    // 					-4510.000000
    // 				</Var>
    // 				<Var name="P2.x" type="double" minval="" maxval="">
    // 					6020.000000
    // 				</Var>
    // 				<Var name="P2.y" type="double" minval="" maxval="">
    // 					4510.000000
    // 				</Var>
    // 				<Var name="Line thickness" type="double" minval="" maxval="">
    // 					10.000000
    // 				</Var>
    // 			</Var>
    // 			<Var name="HalfwayLine" type="list">
    // 				<Var name="Name" type="string">
    // 					HalfwayLine
    // 				</Var>
    // 				<Var name="Type" type="stringenum">
    // 					HalfwayLine
    // 					<Var name="0" type="string">
    // 						Undefined
    // 					</Var>
    // 					<Var name="1" type="string">
    // 						TopTouchLine
    // 					</Var>
    // 					<Var name="2" type="string">
    // 						BottomTouchLine
    // 					</Var>
    // 					<Var name="3" type="string">
    // 						LeftGoalLine
    // 					</Var>
    // 					<Var name="4" type="string">
    // 						RightGoalLine
    // 					</Var>
    // 					<Var name="5" type="string">
    // 						HalfwayLine
    // 					</Var>
    // 					<Var name="6" type="string">
    // 						CenterLine
    // 					</Var>
    // 					<Var name="7" type="string">
    // 						LeftPenaltyStretch
    // 					</Var>
    // 					<Var name="8" type="string">
    // 						RightPenaltyStretch
    // 					</Var>
    // 					<Var name="9" type="string">
    // 						LeftFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="10" type="string">
    // 						LeftFieldRightPenaltyStretch
    // 					</Var>
    // 					<Var name="11" type="string">
    // 						RightFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="12" type="string">
    // 						RightFieldRightPenaltyStretch
    // 					</Var>
    // 				</Var>
    // 				<Var name="P1.x" type="double" minval="" maxval="">
    // 					0.000000
    // 				</Var>
    // 				<Var name="P1.y" type="double" minval="" maxval="">
    // 					-4510.000000
    // 				</Var>
    // 				<Var name="P2.x" type="double" minval="" maxval="">
    // 					0.000000
    // 				</Var>
    // 				<Var name="P2.y" type="double" minval="" maxval="">
    // 					4510.000000
    // 				</Var>
    // 				<Var name="Line thickness" type="double" minval="" maxval="">
    // 					10.000000
    // 				</Var>
    // 			</Var>
    // 			<Var name="CenterLine" type="list">
    // 				<Var name="Name" type="string">
    // 					CenterLine
    // 				</Var>
    // 				<Var name="Type" type="stringenum">
    // 					CenterLine
    // 					<Var name="0" type="string">
    // 						Undefined
    // 					</Var>
    // 					<Var name="1" type="string">
    // 						TopTouchLine
    // 					</Var>
    // 					<Var name="2" type="string">
    // 						BottomTouchLine
    // 					</Var>
    // 					<Var name="3" type="string">
    // 						LeftGoalLine
    // 					</Var>
    // 					<Var name="4" type="string">
    // 						RightGoalLine
    // 					</Var>
    // 					<Var name="5" type="string">
    // 						HalfwayLine
    // 					</Var>
    // 					<Var name="6" type="string">
    // 						CenterLine
    // 					</Var>
    // 					<Var name="7" type="string">
    // 						LeftPenaltyStretch
    // 					</Var>
    // 					<Var name="8" type="string">
    // 						RightPenaltyStretch
    // 					</Var>
    // 					<Var name="9" type="string">
    // 						LeftFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="10" type="string">
    // 						LeftFieldRightPenaltyStretch
    // 					</Var>
    // 					<Var name="11" type="string">
    // 						RightFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="12" type="string">
    // 						RightFieldRightPenaltyStretch
    // 					</Var>
    // 				</Var>
    // 				<Var name="P1.x" type="double" minval="" maxval="">
    // 					-6020.000000
    // 				</Var>
    // 				<Var name="P1.y" type="double" minval="" maxval="">
    // 					0.000000
    // 				</Var>
    // 				<Var name="P2.x" type="double" minval="" maxval="">
    // 					6020.000000
    // 				</Var>
    // 				<Var name="P2.y" type="double" minval="" maxval="">
    // 					0.000000
    // 				</Var>
    // 				<Var name="Line thickness" type="double" minval="" maxval="">
    // 					10.000000
    // 				</Var>
    // 			</Var>
    // 			<Var name="LeftPenaltyStretch" type="list">
    // 				<Var name="Name" type="string">
    // 					LeftPenaltyStretch
    // 				</Var>
    // 				<Var name="Type" type="stringenum">
    // 					LeftPenaltyStretch
    // 					<Var name="0" type="string">
    // 						Undefined
    // 					</Var>
    // 					<Var name="1" type="string">
    // 						TopTouchLine
    // 					</Var>
    // 					<Var name="2" type="string">
    // 						BottomTouchLine
    // 					</Var>
    // 					<Var name="3" type="string">
    // 						LeftGoalLine
    // 					</Var>
    // 					<Var name="4" type="string">
    // 						RightGoalLine
    // 					</Var>
    // 					<Var name="5" type="string">
    // 						HalfwayLine
    // 					</Var>
    // 					<Var name="6" type="string">
    // 						CenterLine
    // 					</Var>
    // 					<Var name="7" type="string">
    // 						LeftPenaltyStretch
    // 					</Var>
    // 					<Var name="8" type="string">
    // 						RightPenaltyStretch
    // 					</Var>
    // 					<Var name="9" type="string">
    // 						LeftFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="10" type="string">
    // 						LeftFieldRightPenaltyStretch
    // 					</Var>
    // 					<Var name="11" type="string">
    // 						RightFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="12" type="string">
    // 						RightFieldRightPenaltyStretch
    // 					</Var>
    // 				</Var>
    // 				<Var name="P1.x" type="double" minval="" maxval="">
    // 					-4800.000000
    // 				</Var>
    // 				<Var name="P1.y" type="double" minval="" maxval="">
    // 					-1205.000000
    // 				</Var>
    // 				<Var name="P2.x" type="double" minval="" maxval="">
    // 					-4800.000000
    // 				</Var>
    // 				<Var name="P2.y" type="double" minval="" maxval="">
    // 					1205.000000
    // 				</Var>
    // 				<Var name="Line thickness" type="double" minval="" maxval="">
    // 					10.000000
    // 				</Var>
    // 			</Var>
    // 			<Var name="RightPenaltyStretch" type="list">
    // 				<Var name="Name" type="string">
    // 					RightPenaltyStretch
    // 				</Var>
    // 				<Var name="Type" type="stringenum">
    // 					RightPenaltyStretch
    // 					<Var name="0" type="string">
    // 						Undefined
    // 					</Var>
    // 					<Var name="1" type="string">
    // 						TopTouchLine
    // 					</Var>
    // 					<Var name="2" type="string">
    // 						BottomTouchLine
    // 					</Var>
    // 					<Var name="3" type="string">
    // 						LeftGoalLine
    // 					</Var>
    // 					<Var name="4" type="string">
    // 						RightGoalLine
    // 					</Var>
    // 					<Var name="5" type="string">
    // 						HalfwayLine
    // 					</Var>
    // 					<Var name="6" type="string">
    // 						CenterLine
    // 					</Var>
    // 					<Var name="7" type="string">
    // 						LeftPenaltyStretch
    // 					</Var>
    // 					<Var name="8" type="string">
    // 						RightPenaltyStretch
    // 					</Var>
    // 					<Var name="9" type="string">
    // 						LeftFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="10" type="string">
    // 						LeftFieldRightPenaltyStretch
    // 					</Var>
    // 					<Var name="11" type="string">
    // 						RightFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="12" type="string">
    // 						RightFieldRightPenaltyStretch
    // 					</Var>
    // 				</Var>
    // 				<Var name="P1.x" type="double" minval="" maxval="">
    // 					4800.000000
    // 				</Var>
    // 				<Var name="P1.y" type="double" minval="" maxval="">
    // 					-1205.000000
    // 				</Var>
    // 				<Var name="P2.x" type="double" minval="" maxval="">
    // 					4800.000000
    // 				</Var>
    // 				<Var name="P2.y" type="double" minval="" maxval="">
    // 					1205.000000
    // 				</Var>
    // 				<Var name="Line thickness" type="double" minval="" maxval="">
    // 					10.000000
    // 				</Var>
    // 			</Var>
    // 			<Var name="LeftFieldLeftPenaltyStretch" type="list">
    // 				<Var name="Name" type="string">
    // 					LeftFieldLeftPenaltyStretch
    // 				</Var>
    // 				<Var name="Type" type="stringenum">
    // 					LeftFieldLeftPenaltyStretch
    // 					<Var name="0" type="string">
    // 						Undefined
    // 					</Var>
    // 					<Var name="1" type="string">
    // 						TopTouchLine
    // 					</Var>
    // 					<Var name="2" type="string">
    // 						BottomTouchLine
    // 					</Var>
    // 					<Var name="3" type="string">
    // 						LeftGoalLine
    // 					</Var>
    // 					<Var name="4" type="string">
    // 						RightGoalLine
    // 					</Var>
    // 					<Var name="5" type="string">
    // 						HalfwayLine
    // 					</Var>
    // 					<Var name="6" type="string">
    // 						CenterLine
    // 					</Var>
    // 					<Var name="7" type="string">
    // 						LeftPenaltyStretch
    // 					</Var>
    // 					<Var name="8" type="string">
    // 						RightPenaltyStretch
    // 					</Var>
    // 					<Var name="9" type="string">
    // 						LeftFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="10" type="string">
    // 						LeftFieldRightPenaltyStretch
    // 					</Var>
    // 					<Var name="11" type="string">
    // 						RightFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="12" type="string">
    // 						RightFieldRightPenaltyStretch
    // 					</Var>
    // 				</Var>
    // 				<Var name="P1.x" type="double" minval="" maxval="">
    // 					-6020.000000
    // 				</Var>
    // 				<Var name="P1.y" type="double" minval="" maxval="">
    // 					-1205.000000
    // 				</Var>
    // 				<Var name="P2.x" type="double" minval="" maxval="">
    // 					-4800.000000
    // 				</Var>
    // 				<Var name="P2.y" type="double" minval="" maxval="">
    // 					-1205.000000
    // 				</Var>
    // 				<Var name="Line thickness" type="double" minval="" maxval="">
    // 					10.000000
    // 				</Var>
    // 			</Var>
    // 			<Var name="LeftFieldRightPenaltyStretch" type="list">
    // 				<Var name="Name" type="string">
    // 					LeftFieldRightPenaltyStretch
    // 				</Var>
    // 				<Var name="Type" type="stringenum">
    // 					LeftFieldRightPenaltyStretch
    // 					<Var name="0" type="string">
    // 						Undefined
    // 					</Var>
    // 					<Var name="1" type="string">
    // 						TopTouchLine
    // 					</Var>
    // 					<Var name="2" type="string">
    // 						BottomTouchLine
    // 					</Var>
    // 					<Var name="3" type="string">
    // 						LeftGoalLine
    // 					</Var>
    // 					<Var name="4" type="string">
    // 						RightGoalLine
    // 					</Var>
    // 					<Var name="5" type="string">
    // 						HalfwayLine
    // 					</Var>
    // 					<Var name="6" type="string">
    // 						CenterLine
    // 					</Var>
    // 					<Var name="7" type="string">
    // 						LeftPenaltyStretch
    // 					</Var>
    // 					<Var name="8" type="string">
    // 						RightPenaltyStretch
    // 					</Var>
    // 					<Var name="9" type="string">
    // 						LeftFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="10" type="string">
    // 						LeftFieldRightPenaltyStretch
    // 					</Var>
    // 					<Var name="11" type="string">
    // 						RightFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="12" type="string">
    // 						RightFieldRightPenaltyStretch
    // 					</Var>
    // 				</Var>
    // 				<Var name="P1.x" type="double" minval="" maxval="">
    // 					-6020.000000
    // 				</Var>
    // 				<Var name="P1.y" type="double" minval="" maxval="">
    // 					1205.000000
    // 				</Var>
    // 				<Var name="P2.x" type="double" minval="" maxval="">
    // 					-4800.000000
    // 				</Var>
    // 				<Var name="P2.y" type="double" minval="" maxval="">
    // 					1205.000000
    // 				</Var>
    // 				<Var name="Line thickness" type="double" minval="" maxval="">
    // 					10.000000
    // 				</Var>
    // 			</Var>
    // 			<Var name="RightFieldRightPenaltyStretch" type="list">
    // 				<Var name="Name" type="string">
    // 					RightFieldRightPenaltyStretch
    // 				</Var>
    // 				<Var name="Type" type="stringenum">
    // 					RightFieldRightPenaltyStretch
    // 					<Var name="0" type="string">
    // 						Undefined
    // 					</Var>
    // 					<Var name="1" type="string">
    // 						TopTouchLine
    // 					</Var>
    // 					<Var name="2" type="string">
    // 						BottomTouchLine
    // 					</Var>
    // 					<Var name="3" type="string">
    // 						LeftGoalLine
    // 					</Var>
    // 					<Var name="4" type="string">
    // 						RightGoalLine
    // 					</Var>
    // 					<Var name="5" type="string">
    // 						HalfwayLine
    // 					</Var>
    // 					<Var name="6" type="string">
    // 						CenterLine
    // 					</Var>
    // 					<Var name="7" type="string">
    // 						LeftPenaltyStretch
    // 					</Var>
    // 					<Var name="8" type="string">
    // 						RightPenaltyStretch
    // 					</Var>
    // 					<Var name="9" type="string">
    // 						LeftFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="10" type="string">
    // 						LeftFieldRightPenaltyStretch
    // 					</Var>
    // 					<Var name="11" type="string">
    // 						RightFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="12" type="string">
    // 						RightFieldRightPenaltyStretch
    // 					</Var>
    // 				</Var>
    // 				<Var name="P1.x" type="double" minval="" maxval="">
    // 					6020.000000
    // 				</Var>
    // 				<Var name="P1.y" type="double" minval="" maxval="">
    // 					-1205.000000
    // 				</Var>
    // 				<Var name="P2.x" type="double" minval="" maxval="">
    // 					4800.000000
    // 				</Var>
    // 				<Var name="P2.y" type="double" minval="" maxval="">
    // 					-1205.000000
    // 				</Var>
    // 				<Var name="Line thickness" type="double" minval="" maxval="">
    // 					10.000000
    // 				</Var>
    // 			</Var>
    // 			<Var name="RightFieldLeftPenaltyStretch" type="list">
    // 				<Var name="Name" type="string">
    // 					RightFieldLeftPenaltyStretch
    // 				</Var>
    // 				<Var name="Type" type="stringenum">
    // 					RightFieldLeftPenaltyStretch
    // 					<Var name="0" type="string">
    // 						Undefined
    // 					</Var>
    // 					<Var name="1" type="string">
    // 						TopTouchLine
    // 					</Var>
    // 					<Var name="2" type="string">
    // 						BottomTouchLine
    // 					</Var>
    // 					<Var name="3" type="string">
    // 						LeftGoalLine
    // 					</Var>
    // 					<Var name="4" type="string">
    // 						RightGoalLine
    // 					</Var>
    // 					<Var name="5" type="string">
    // 						HalfwayLine
    // 					</Var>
    // 					<Var name="6" type="string">
    // 						CenterLine
    // 					</Var>
    // 					<Var name="7" type="string">
    // 						LeftPenaltyStretch
    // 					</Var>
    // 					<Var name="8" type="string">
    // 						RightPenaltyStretch
    // 					</Var>
    // 					<Var name="9" type="string">
    // 						LeftFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="10" type="string">
    // 						LeftFieldRightPenaltyStretch
    // 					</Var>
    // 					<Var name="11" type="string">
    // 						RightFieldLeftPenaltyStretch
    // 					</Var>
    // 					<Var name="12" type="string">
    // 						RightFieldRightPenaltyStretch
    // 					</Var>
    // 				</Var>
    // 				<Var name="P1.x" type="double" minval="" maxval="">
    // 					6020.000000
    // 				</Var>
    // 				<Var name="P1.y" type="double" minval="" maxval="">
    // 					1205.000000
    // 				</Var>
    // 				<Var name="P2.x" type="double" minval="" maxval="">
    // 					4800.000000
    // 				</Var>
    // 				<Var name="P2.y" type="double" minval="" maxval="">
    // 					1205.000000
    // 				</Var>
    // 				<Var name="Line thickness" type="double" minval="" maxval="">
    // 					10.000000
    // 				</Var>
    // 			</Var>
    // 		</Var>
    // 		<Var name="Field Arcs" type="list">
    // 			<Var name="CenterCircle" type="list">
    // 				<Var name="Name" type="string">
    // 					CenterCircle
    // 				</Var>
    // 				<Var name="Type" type="stringenum">
    // 					CenterCircle
    // 					<Var name="0" type="string">
    // 						Undefined
    // 					</Var>
    // 					<Var name="1" type="string">
    // 						CenterCircle
    // 					</Var>
    // 				</Var>
    // 				<Var name="Center.x" type="double" minval="" maxval="">
    // 					0.000000
    // 				</Var>
    // 				<Var name="Center.y" type="double" minval="" maxval="">
    // 					0.000000
    // 				</Var>
    // 				<Var name="Radius" type="double" minval="" maxval="">
    // 					500.000000
    // 				</Var>
    // 				<Var name="Start angle" type="double" minval="" maxval="">
    // 					0.000000
    // 				</Var>
    // 				<Var name="End angle" type="double" minval="" maxval="">
    // 					6.283185
    // 				</Var>
    // 				<Var name="Line thickness" type="double" minval="" maxval="">
    // 					10.000000
    // 				</Var>
    // 			</Var>
    // 		</Var>
    // 	</Var>

    fn default() -> Self {
        // let lines = vec![
        //     FieldLineSegment::new("TopTouchLine",
        // ]
        Self {
            field_length: todo!(),
            field_width: todo!(),
            goal_width: todo!(),
            goal_depth: todo!(),
            boundary_width: todo!(),
            line_segments: todo!(),
            circular_arcs: todo!(),
        }
    }
}
