//! Debug visualization types for strategies.
//!
//! These types allow strategies to send debug information back to the executor
//! for visualization in the UI.

use serde::{Deserialize, Serialize};

use crate::Vector2;

/// A color for debug visualization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebugColor {
    #[default]
    Red,
    Green,
    Blue,
    Orange,
    Purple,
    Gray,
    Yellow,
    Cyan,
    Magenta,
    White,
    Black,
}

impl From<dies_core::DebugColor> for DebugColor {
    fn from(c: dies_core::DebugColor) -> Self {
        match c {
            dies_core::DebugColor::Red => DebugColor::Red,
            dies_core::DebugColor::Green => DebugColor::Green,
            dies_core::DebugColor::Blue => DebugColor::Blue,
            dies_core::DebugColor::Orange => DebugColor::Orange,
            dies_core::DebugColor::Purple => DebugColor::Purple,
            dies_core::DebugColor::Gray => DebugColor::Gray,
        }
    }
}

impl From<DebugColor> for dies_core::DebugColor {
    fn from(c: DebugColor) -> Self {
        match c {
            DebugColor::Red => dies_core::DebugColor::Red,
            DebugColor::Green => dies_core::DebugColor::Green,
            DebugColor::Blue => dies_core::DebugColor::Blue,
            DebugColor::Orange => dies_core::DebugColor::Orange,
            DebugColor::Purple => dies_core::DebugColor::Purple,
            DebugColor::Gray => dies_core::DebugColor::Gray,
            // Map additional colors to closest dies_core equivalents
            DebugColor::Yellow => dies_core::DebugColor::Orange,
            DebugColor::Cyan => dies_core::DebugColor::Blue,
            DebugColor::Magenta => dies_core::DebugColor::Purple,
            DebugColor::White => dies_core::DebugColor::Gray,
            DebugColor::Black => dies_core::DebugColor::Gray,
        }
    }
}

/// A debug shape to draw on the field.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DebugShape {
    /// A cross marker at a position.
    Cross { center: Vector2, color: DebugColor },

    /// A line between two points.
    Line {
        start: Vector2,
        end: Vector2,
        color: DebugColor,
    },

    /// A circle (filled or stroked).
    Circle {
        center: Vector2,
        radius: f64,
        fill: Option<DebugColor>,
        stroke: Option<DebugColor>,
    },
}

/// A debug value that can be displayed in the UI.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DebugValue {
    /// A shape to draw on the field.
    Shape(DebugShape),
    /// A numeric value (for plotting/display).
    Number(f64),
    /// A string value (for display).
    String(String),
}

impl From<DebugShape> for DebugValue {
    fn from(shape: DebugShape) -> Self {
        DebugValue::Shape(shape)
    }
}

/// A debug entry with a key and value.
///
/// Keys should be `snake_case` and use `.` as a separator for hierarchical keys.
/// Common patterns:
/// - `p{player_id}.{value}`: A value associated with a specific player
/// - `team.{value}`: A team-level value
/// - `strategy.{value}`: A strategy-specific value
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DebugEntry {
    /// The key for this debug entry.
    pub key: String,
    /// The debug value.
    pub value: DebugValue,
}

impl DebugEntry {
    /// Create a new debug entry.
    pub fn new(key: impl Into<String>, value: DebugValue) -> Self {
        Self {
            key: key.into(),
            value,
        }
    }

    /// Create a debug entry with a cross shape.
    pub fn cross(key: impl Into<String>, center: Vector2, color: DebugColor) -> Self {
        Self::new(key, DebugValue::Shape(DebugShape::Cross { center, color }))
    }

    /// Create a debug entry with a line shape.
    pub fn line(
        key: impl Into<String>,
        start: Vector2,
        end: Vector2,
        color: DebugColor,
    ) -> Self {
        Self::new(
            key,
            DebugValue::Shape(DebugShape::Line { start, end, color }),
        )
    }

    /// Create a debug entry with a filled circle.
    pub fn circle_fill(
        key: impl Into<String>,
        center: Vector2,
        radius: f64,
        fill: DebugColor,
    ) -> Self {
        Self::new(
            key,
            DebugValue::Shape(DebugShape::Circle {
                center,
                radius,
                fill: Some(fill),
                stroke: None,
            }),
        )
    }

    /// Create a debug entry with a stroked circle.
    pub fn circle_stroke(
        key: impl Into<String>,
        center: Vector2,
        radius: f64,
        stroke: DebugColor,
    ) -> Self {
        Self::new(
            key,
            DebugValue::Shape(DebugShape::Circle {
                center,
                radius,
                fill: None,
                stroke: Some(stroke),
            }),
        )
    }

    /// Create a debug entry with a numeric value.
    pub fn number(key: impl Into<String>, value: f64) -> Self {
        Self::new(key, DebugValue::Number(value))
    }

    /// Create a debug entry with a string value.
    pub fn string(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self::new(key, DebugValue::String(value.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_entry_cross_serialization() {
        let entry = DebugEntry::cross("test.marker", Vector2::new(100.0, 200.0), DebugColor::Red);

        let encoded = bincode::serialize(&entry).unwrap();
        let decoded: DebugEntry = bincode::deserialize(&encoded).unwrap();

        assert_eq!(decoded.key, "test.marker");
        match decoded.value {
            DebugValue::Shape(DebugShape::Cross { center, color }) => {
                assert!((center.x - 100.0).abs() < 1e-6);
                assert!((center.y - 200.0).abs() < 1e-6);
                assert_eq!(color, DebugColor::Red);
            }
            _ => panic!("Expected Cross shape"),
        }
    }

    #[test]
    fn test_debug_entry_line_serialization() {
        let entry = DebugEntry::line(
            "test.line",
            Vector2::new(0.0, 0.0),
            Vector2::new(100.0, 100.0),
            DebugColor::Green,
        );

        let encoded = bincode::serialize(&entry).unwrap();
        let decoded: DebugEntry = bincode::deserialize(&encoded).unwrap();

        assert_eq!(decoded.key, "test.line");
        match decoded.value {
            DebugValue::Shape(DebugShape::Line { start, end, color }) => {
                assert!((start.x - 0.0).abs() < 1e-6);
                assert!((end.x - 100.0).abs() < 1e-6);
                assert_eq!(color, DebugColor::Green);
            }
            _ => panic!("Expected Line shape"),
        }
    }

    #[test]
    fn test_debug_entry_circle_fill_serialization() {
        let entry =
            DebugEntry::circle_fill("test.circle", Vector2::new(50.0, 50.0), 25.0, DebugColor::Blue);

        let encoded = bincode::serialize(&entry).unwrap();
        let decoded: DebugEntry = bincode::deserialize(&encoded).unwrap();

        match decoded.value {
            DebugValue::Shape(DebugShape::Circle {
                center,
                radius,
                fill,
                stroke,
            }) => {
                assert!((center.x - 50.0).abs() < 1e-6);
                assert!((radius - 25.0).abs() < 1e-6);
                assert_eq!(fill, Some(DebugColor::Blue));
                assert_eq!(stroke, None);
            }
            _ => panic!("Expected Circle shape"),
        }
    }

    #[test]
    fn test_debug_entry_number_serialization() {
        let entry = DebugEntry::number("test.value", 42.5);

        let encoded = bincode::serialize(&entry).unwrap();
        let decoded: DebugEntry = bincode::deserialize(&encoded).unwrap();

        assert_eq!(decoded.key, "test.value");
        match decoded.value {
            DebugValue::Number(n) => assert!((n - 42.5).abs() < 1e-6),
            _ => panic!("Expected Number value"),
        }
    }

    #[test]
    fn test_debug_entry_string_serialization() {
        let entry = DebugEntry::string("test.message", "Hello, World!");

        let encoded = bincode::serialize(&entry).unwrap();
        let decoded: DebugEntry = bincode::deserialize(&encoded).unwrap();

        assert_eq!(decoded.key, "test.message");
        match decoded.value {
            DebugValue::String(s) => assert_eq!(s, "Hello, World!"),
            _ => panic!("Expected String value"),
        }
    }

    #[test]
    fn test_debug_color_conversion() {
        // Test round-trip for colors that exist in both enums
        for core_color in [
            dies_core::DebugColor::Red,
            dies_core::DebugColor::Green,
            dies_core::DebugColor::Blue,
            dies_core::DebugColor::Orange,
            dies_core::DebugColor::Purple,
            dies_core::DebugColor::Gray,
        ] {
            let protocol_color: DebugColor = core_color.clone().into();
            let back_to_core: dies_core::DebugColor = protocol_color.into();
            assert_eq!(format!("{:?}", core_color), format!("{:?}", back_to_core));
        }
    }
}

