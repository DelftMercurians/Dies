use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FieldGeometry {
    pub field_length: f32,
    pub field_width: f32,
    pub goal_width: f32,
    pub goal_depth: f32,
    pub boundary_width: f32,
    pub penalty_area_depth: f32,
    pub penalty_area_width: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FieldCircularArc {
    // A structure to store a single field arc
    // Params:
    // a1: Start angle in counter-clockwise order.
    // a2: End angle in counter-clockwise order.
    pub index: i64,
    pub name: String,
    pub center: Vector2<f32>,
    pub radius: f32,
    pub a1: f32,
    pub a2: f32,
    pub thickness: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FieldLineSegment {
    // A single field line segment
    // Params:
    // p1: Start point of segment
    // p2: End point of segment
    pub index: i64,
    pub name: String,
    pub p1: Vector2<f32>,
    pub p2: Vector2<f32>,
    pub thickness: f32,
}
