//! Debug visualization API for strategies.
//!
//! This module provides functions for adding debug visualizations that appear
//! in the UI. All coordinates are in the team-relative frame and are automatically
//! transformed for display.
//!
//! # Usage
//!
//! ```ignore
//! use dies_strategy_api::debug;
//!
//! fn update(&mut self, ctx: &mut TeamContext) {
//!     // Draw a cross at a position
//!     debug::cross("target", target_pos);
//!     
//!     // Draw a line
//!     debug::line("path", start, end);
//!     
//!     // Draw a circle
//!     debug::circle("zone", center, 500.0);
//!     
//!     // Record a value for plotting
//!     debug::value("ball_speed", ball_velocity.norm());
//! }
//! ```

use std::cell::RefCell;

use dies_strategy_protocol::{DebugColor, DebugEntry, DebugShape, DebugValue, Vector2};

// Thread-local storage for debug entries collected during a frame.
thread_local! {
    static DEBUG_ENTRIES: RefCell<Vec<DebugEntry>> = RefCell::new(Vec::new());
}

/// Draw a cross marker at a position (default color: red).
pub fn cross(key: &str, position: Vector2) {
    cross_colored(key, position, DebugColor::Red);
}

/// Draw a cross marker at a position with a specific color.
pub fn cross_colored(key: &str, position: Vector2, color: DebugColor) {
    add_entry(DebugEntry::new(
        key,
        DebugValue::Shape(DebugShape::Cross {
            center: position,
            color,
        }),
    ));
}

/// Draw a line between two points (default color: green).
pub fn line(key: &str, start: Vector2, end: Vector2) {
    line_colored(key, start, end, DebugColor::Green);
}

/// Draw a line between two points with a specific color.
pub fn line_colored(key: &str, start: Vector2, end: Vector2, color: DebugColor) {
    add_entry(DebugEntry::new(
        key,
        DebugValue::Shape(DebugShape::Line { start, end, color }),
    ));
}

/// Draw a circle outline (default color: blue).
pub fn circle(key: &str, center: Vector2, radius: f64) {
    circle_stroke(key, center, radius, DebugColor::Blue);
}

/// Draw a circle outline with a specific color.
pub fn circle_stroke(key: &str, center: Vector2, radius: f64, color: DebugColor) {
    add_entry(DebugEntry::new(
        key,
        DebugValue::Shape(DebugShape::Circle {
            center,
            radius,
            fill: None,
            stroke: Some(color),
        }),
    ));
}

/// Draw a filled circle.
pub fn circle_filled(key: &str, center: Vector2, radius: f64, color: DebugColor) {
    add_entry(DebugEntry::new(
        key,
        DebugValue::Shape(DebugShape::Circle {
            center,
            radius,
            fill: Some(color),
            stroke: None,
        }),
    ));
}

/// Draw a circle with both fill and stroke.
pub fn circle_filled_stroke(
    key: &str,
    center: Vector2,
    radius: f64,
    fill: DebugColor,
    stroke: DebugColor,
) {
    add_entry(DebugEntry::new(
        key,
        DebugValue::Shape(DebugShape::Circle {
            center,
            radius,
            fill: Some(fill),
            stroke: Some(stroke),
        }),
    ));
}

/// Record a numeric value (for plotting/display).
///
/// Values with the same key are tracked over time and can be plotted in the UI.
pub fn value(key: &str, v: f64) {
    add_entry(DebugEntry::new(key, DebugValue::Number(v)));
}

/// Record a string value (for display).
pub fn string(key: &str, v: &str) {
    add_entry(DebugEntry::new(key, DebugValue::String(v.to_string())));
}

/// Remove a debug entry by key.
///
/// Note: This removes the entry from the current frame's output but doesn't
/// affect entries from previous frames that may still be displayed.
pub fn remove(key: &str) {
    DEBUG_ENTRIES.with(|entries| {
        entries.borrow_mut().retain(|e| e.key != key);
    });
}

/// Clear all debug entries for this frame.
pub fn clear() {
    DEBUG_ENTRIES.with(|entries| {
        entries.borrow_mut().clear();
    });
}

/// Add a debug entry (internal).
fn add_entry(entry: DebugEntry) {
    DEBUG_ENTRIES.with(|entries| {
        entries.borrow_mut().push(entry);
    });
}

/// Collect and drain all debug entries (called by runner after update).
pub fn collect_entries() -> Vec<DebugEntry> {
    DEBUG_ENTRIES.with(|entries| entries.borrow_mut().drain(..).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cleanup() {
        clear();
    }

    #[test]
    fn test_cross() {
        cleanup();

        cross("test.cross", Vector2::new(100.0, 200.0));

        let entries = collect_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, "test.cross");

        match &entries[0].value {
            DebugValue::Shape(DebugShape::Cross { center, color }) => {
                assert_eq!(center.x, 100.0);
                assert_eq!(center.y, 200.0);
                assert_eq!(*color, DebugColor::Red);
            }
            _ => panic!("Expected Cross shape"),
        }
    }

    #[test]
    fn test_line() {
        cleanup();

        line_colored(
            "test.line",
            Vector2::new(0.0, 0.0),
            Vector2::new(100.0, 100.0),
            DebugColor::Green,
        );

        let entries = collect_entries();
        assert_eq!(entries.len(), 1);

        match &entries[0].value {
            DebugValue::Shape(DebugShape::Line { start, end, color }) => {
                assert_eq!(start.x, 0.0);
                assert_eq!(end.x, 100.0);
                assert_eq!(*color, DebugColor::Green);
            }
            _ => panic!("Expected Line shape"),
        }
    }

    #[test]
    fn test_circle() {
        cleanup();

        circle_filled(
            "test.circle",
            Vector2::new(50.0, 50.0),
            25.0,
            DebugColor::Blue,
        );

        let entries = collect_entries();
        assert_eq!(entries.len(), 1);

        match &entries[0].value {
            DebugValue::Shape(DebugShape::Circle {
                center,
                radius,
                fill,
                stroke,
            }) => {
                assert_eq!(center.x, 50.0);
                assert_eq!(*radius, 25.0);
                assert_eq!(*fill, Some(DebugColor::Blue));
                assert_eq!(*stroke, None);
            }
            _ => panic!("Expected Circle shape"),
        }
    }

    #[test]
    fn test_value() {
        cleanup();

        value("test.speed", 42.5);

        let entries = collect_entries();
        assert_eq!(entries.len(), 1);

        match &entries[0].value {
            DebugValue::Number(v) => assert!((*v - 42.5).abs() < 1e-6),
            _ => panic!("Expected Number value"),
        }
    }

    #[test]
    fn test_string() {
        cleanup();

        string("test.message", "Hello");

        let entries = collect_entries();
        assert_eq!(entries.len(), 1);

        match &entries[0].value {
            DebugValue::String(s) => assert_eq!(s, "Hello"),
            _ => panic!("Expected String value"),
        }
    }

    #[test]
    fn test_remove() {
        cleanup();

        cross("keep", Vector2::new(0.0, 0.0));
        cross("remove", Vector2::new(1.0, 1.0));

        remove("remove");

        let entries = collect_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, "keep");
    }

    #[test]
    fn test_clear() {
        cleanup();

        cross("a", Vector2::new(0.0, 0.0));
        cross("b", Vector2::new(1.0, 1.0));

        clear();

        let entries = collect_entries();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_multiple_entries() {
        cleanup();

        cross("marker", Vector2::new(0.0, 0.0));
        line("path", Vector2::new(0.0, 0.0), Vector2::new(100.0, 100.0));
        value("speed", 10.0);

        let entries = collect_entries();
        assert_eq!(entries.len(), 3);
    }
}
