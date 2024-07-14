use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, RwLock},
};

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Notify};
use typeshare::typeshare;

use crate::Vector2;

static DEBUG_MESSAGES: OnceLock<mpsc::UnboundedSender<UpdateMsg>> = OnceLock::new();

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[typeshare]
pub enum DebugColor {
    #[default]
    Red,
    Green,
    Orange,
    Purple,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
#[typeshare]
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
#[typeshare]
pub enum DebugValue {
    Shape(DebugShape),
    Number(f64),
    String(String),
}

/// A map of debug messages.
///
/// # Key format
///
/// The keys should always be `snake_case` and should only contain alphanumerical
/// characters. The `.` character is reserved for separating different parts of the key.
///
/// Keys can be arbitrary, but there are some patterns that are recognized by the
/// UI:
/// - `p{player_id}.{value}`: A value associated with a player.
#[typeshare]
pub type DebugMap = HashMap<String, DebugValue>;

#[derive(Clone)]
pub struct DebugSubscriber {
    map: Arc<RwLock<DebugMap>>,
    notify: Arc<Notify>,
}

#[derive(Debug)]
enum UpdateMsg {
    InsertRecord { key: String, value: DebugValue },
    RemoveRecord { key: String },
}

impl DebugSubscriber {
    /// Spawns a new debug subscriber.
    ///
    /// # Panics
    ///
    /// Panics if another debug subscriber has already been spawned.
    pub fn spawn() -> Self {
        let (record_tx, mut record_rx) = mpsc::unbounded_channel();
        DEBUG_MESSAGES
            .set(record_tx)
            .expect("Only one debug subscriber can be created");
        let map = Arc::new(RwLock::new(HashMap::new()));
        let notify = Arc::new(Notify::new());
        {
            let map = map.clone();
            let notify = notify.clone();
            tokio::spawn(async move {
                while let Some(record) = record_rx.recv().await {
                    match record {
                        UpdateMsg::RemoveRecord { key } => {
                            map.write().unwrap().remove(&key);
                        }
                        UpdateMsg::InsertRecord { key, value } => {
                            map.write().unwrap().insert(key, value);
                        }
                    }
                    notify.notify_waiters();
                }
            });
        }
        Self { map, notify }
    }

    /// Get a copy of the current debug map.
    pub fn get_copy(&self) -> DebugMap {
        self.map.read().unwrap().clone()
    }

    /// Wait for a new debug message and get a copy of the current debug map.
    pub async fn wait_and_get_copy(&self) -> DebugMap {
        self.notify.notified().await;
        self.get_copy()
    }
}

/// A debug record that will be removed when dropped.
#[derive(Debug)]
pub struct DebugRecord {
    key: String,
}

impl Default for DebugRecord {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugRecord {
    /// Create a new debug record with a random key.
    pub fn new() -> Self {
        Self {
            key: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// Set the value of the debug record.
    pub fn set(&self, value: DebugValue) {
        debug_record(&self.key, value);
    }

    /// Set a cross shape for the debug record.
    pub fn set_cross(&self, center: Vector2, color: DebugColor) {
        self.set(DebugValue::Shape(DebugShape::Cross { center, color }));
    }

    /// Set a filled circle shape for the debug record.
    pub fn set_circle_fill(&self, center: Vector2, radius: f64, fill: DebugColor) {
        self.set(DebugValue::Shape(DebugShape::Circle {
            center,
            radius,
            fill: Some(fill),
            stroke: None,
        }));
    }

    /// Set a hollow circle shape for the debug record.
    pub fn set_circle_stroke(&self, center: Vector2, radius: f64, stroke: DebugColor) {
        self.set(DebugValue::Shape(DebugShape::Circle {
            center,
            radius,
            fill: None,
            stroke: Some(stroke),
        }));
    }

    /// Set a line shape for the debug record.
    pub fn set_line(&self, start: Vector2, end: Vector2, color: DebugColor) {
        self.set(DebugValue::Shape(DebugShape::Line { start, end, color }));
    }

    /// Set a numeric value for the debug record.
    pub fn set_value(&self, value: f64) {
        self.set(DebugValue::Number(value));
    }

    /// Set a string value for the debug record.
    pub fn set_string(&self, value: impl Into<String>) {
        self.set(DebugValue::String(value.into()));
    }
}

impl Drop for DebugRecord {
    fn drop(&mut self) {
        debug_remove(self.key.clone());
    }
}

/// Record a debug message.
pub fn debug_record(key: impl Into<String>, value: DebugValue) {
    if let Some(sender) = DEBUG_MESSAGES.get() {
        let _ = sender.send(UpdateMsg::InsertRecord {
            key: key.into(),
            value,
        });
    }
}

/// Remove a debug message.
pub fn debug_remove(key: impl Into<String>) {
    if let Some(sender) = DEBUG_MESSAGES.get() {
        let _ = sender.send(UpdateMsg::RemoveRecord { key: key.into() });
    }
}

/// Record a debug message with a cross.
pub fn debug_cross(key: impl Into<String>, center: Vector2, color: DebugColor) {
    debug_record(
        key.into(),
        DebugValue::Shape(DebugShape::Cross { center, color }),
    );
}

/// Record a debug message with a filled circle.
pub fn debug_circle_fill(key: impl Into<String>, center: Vector2, radius: f64, fill: DebugColor) {
    debug_record(
        key.into(),
        DebugValue::Shape(DebugShape::Circle {
            center,
            radius,
            fill: Some(fill),
            stroke: None,
        }),
    );
}

/// Record a debug message with a hollow circle.
pub fn debug_circle_stroke(
    key: impl Into<String>,
    center: Vector2,
    radius: f64,
    stroke: DebugColor,
) {
    debug_record(
        key.into(),
        DebugValue::Shape(DebugShape::Circle {
            center,
            radius,
            fill: None,
            stroke: Some(stroke),
        }),
    );
}

/// Record a debug message with a line.
pub fn debug_line(key: impl Into<String>, start: Vector2, end: Vector2, color: DebugColor) {
    debug_record(
        key.into(),
        DebugValue::Shape(DebugShape::Line { start, end, color }),
    );
}

/// Record a debug message with a numeric value.
pub fn debug_value(key: impl Into<String>, value: f64) {
    debug_record(key, DebugValue::Number(value));
}

/// Record a debug message with a string.
pub fn debug_string(key: impl Into<String>, value: impl Into<String>) {
    debug_record(key, DebugValue::String(value.into()));
}

/// Draw a debug shape on the field in the UI. A string key is the first argument,
/// which is used only to update and remove shapes.
///
/// # Examples
///
/// ```rust
/// # use dies_core::{dbg_draw, Vector2, DebugColor};
/// dbg_draw!("test", cross, Vector2::new(0.0, 0.0));
///
/// let id = 1;
/// // Cross shape with specified color
/// dbg_draw!(("p{}.test", id), cross, Vector2::new(0.0, 0.0), DebugColor::Red);
///
/// // Line shape with default color
/// dbg_draw!(
///     ("p{}.test", id),
///     line,
///     Vector2::new(0.0, 0.0),
///     Vector2::new(1.0, 1.0)
/// );
///
/// // Filled circle shape with specified fill color
/// dbg_draw!(
///     ("p{}.test", id),
///     circle_fill,
///     Vector2::new(0.0, 0.0),
///     1.0,
///     DebugColor::Green
/// );
/// ```
#[macro_export]
macro_rules! dbg_draw {
    // Pattern for cross with default color
    (($($key:tt)+), cross, $pos:expr) => {
        dies_core::debug_cross(format!($($key)+), $pos, dies_core::DebugColor::default())
    };
    ($key:tt, cross, $pos:expr) => {
        dies_core::debug_cross($key, $pos, dies_core::DebugColor::default())
    };

    // Pattern for cross with specified color
    (($($key:tt)+), cross, $pos:expr, $color:expr) => {
        dies_core::debug_cross(format!($($key)+), $pos, $color)
    };
    ($key:tt, cross, $pos:expr, $color:expr) => {
        dies_core::debug_cross($key, $pos, $color)
    };

    // Pattern for line with default color
    (($($key:tt)+), line, $v1:expr, $v2:expr) => {
        dies_core::debug_line(format!($($key)+), $v1, $v2, dies_core::DebugColor::default())
    };
    ($key:tt, line, $v1:expr, $v2:expr) => {
        dies_core::debug_line($key, $v1, $v2, dies_core::DebugColor::default())
    };

    // Pattern for line with specified color
    (($($key:tt)+), line, $v1:expr, $v2:expr, $color:expr) => {
        dies_core::debug_line(format!($($key)+), $v1, $v2, $color)
    };
    ($key:tt, line, $v1:expr, $v2:expr, $color:expr) => {
        dies_core::debug_line($key, $v1, $v2, $color)
    };

    // Pattern for filled circle
    (($($key:tt)+), circle_fill, $center:expr, $radius:expr, $fill:expr) => {
        dies_core::debug_circle_fill(format!($($key)+), $center, $radius, $fill)
    };
    ($key:tt, circle_fill, $center:expr, $radius:expr, $fill:expr) => {
        dies_core::debug_circle_fill($key, $center, $radius, $fill)
    };
    (($($key:tt)+), circle_fill, $center:expr, $radius:expr) => {
        dies_core::debug_circle_fill(format!($($key)+), $center, $radius, dies_core::DebugColor::default())
    };
    ($key:tt, circle_fill, $center:expr, $radius:expr) => {
        dies_core::debug_circle_fill($key, $center, $radius, dies_core::DebugColor::default())
    };

    // Pattern for hollow circle
    (($($key:tt)+), circle_stroke, $center:expr, $radius:expr, $stroke:expr) => {
        dies_core::debug_circle_stroke(format!($($key)+), $center, $radius, $stroke)
    };
    ($key:tt, circle_stroke, $center:expr, $radius:expr, $stroke:expr) => {
        dies_core::debug_circle_stroke($key, $center, $radius, $stroke)
    };
    (($($key:tt)+), circle_stroke, $center:expr, $radius:expr) => {
        dies_core::debug_circle_stroke(format!($($key)+), $center, $radius, dies_core::DebugColor::default())
    };
    ($key:tt, circle_stroke, $center:expr, $radius:expr) => {
        dies_core::debug_circle_stroke($key, $center, $radius, dies_core::DebugColor::default())
    };
}
