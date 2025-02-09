use serde::{Deserialize, Serialize};

use crate::Vector2;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DebugColor {
    #[default]
    Red,
    Green,
    Orange,
    Purple,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum DebugShape {
    Cross {
        center: Vector2,
        color: DebugColor,
    },
    Circle {
        center: Vector2,
        radius: f64,
        fill: Option<DebugColor>,
        stroke: Option<DebugColor>,
    },
    Line {
        start: Vector2,
        end: Vector2,
        color: DebugColor,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum DebugValue {
    Shape(DebugShape),
    Number(f64),
    String(String),
}
