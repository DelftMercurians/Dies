use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, OnceLock, RwLock},
};

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Notify};
use typeshare::typeshare;

use crate::{PlayerId, Vector2};

static DEBUG_MESSAGES: OnceLock<mpsc::UnboundedSender<UpdateMsg>> = OnceLock::new();
static DEBUG_SUBSCRIBER: OnceLock<DebugSubscriber> = OnceLock::new();

/// Number of frames a debug key may go un-refreshed before it is evicted as
/// stale. Every producer is expected to re-emit its keys each frame; one-shot
/// keys therefore disappear shortly after they stop being recorded.
const DEBUG_TTL_FRAMES: u64 = 5;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[typeshare]
pub enum DebugColor {
    #[default]
    Red,
    Green,
    Orange,
    Purple,
    Blue,
    Gray,
}

/// Semantic kind of a [`DebugShape::Marker`], controlling which glyph the UI draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[typeshare]
pub enum MarkerKind {
    /// A robot's final destination.
    Target,
    /// An intermediate point along a path.
    Waypoint,
    /// A kick / shot aim point.
    KickTarget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
#[typeshare]
pub enum DebugShape {
    Cross {
        center: Vector2,
        color: DebugColor,
    },
    /// A semantic marker drawn with a distinct glyph, optionally tethered to its
    /// owning robot in the UI.
    Marker {
        kind: MarkerKind,
        center: Vector2,
        color: DebugColor,
        owner: Option<PlayerId>,
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
    TreeNode {
        name: String,
        id: String,
        children_ids: Vec<String>,
        is_active: bool,
        node_type: String,
        internal_state: Option<String>,
        additional_info: Option<String>,
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
    Clear,
    /// Advance the frame clock and evict keys not refreshed within the TTL.
    Tick,
}

impl DebugSubscriber {
    /// Get a handle to the process-wide debug subscriber, spawning the backing
    /// task on the first call. Multiple consumers (webui, test driver, ...)
    /// share one map.
    ///
    /// Must be called inside a tokio runtime — the first call spawns an async
    /// task that drains the channel.
    pub fn instance() -> Self {
        DEBUG_SUBSCRIBER.get_or_init(Self::spawn_inner).clone()
    }

    fn spawn_inner() -> Self {
        let (record_tx, mut record_rx) = mpsc::unbounded_channel();
        DEBUG_MESSAGES
            .set(record_tx)
            .expect("DEBUG_MESSAGES already initialized");
        let map = Arc::new(RwLock::new(HashMap::new()));
        let notify = Arc::new(Notify::new());
        {
            let map = map.clone();
            let notify = notify.clone();
            tokio::spawn(async move {
                // Frame clock + per-key last-seen frame, owned solely by this
                // task. Used to evict keys that stop being refreshed.
                let mut generation: u64 = 0;
                let mut last_seen: HashMap<String, u64> = HashMap::new();
                while let Some(record) = record_rx.recv().await {
                    match record {
                        UpdateMsg::RemoveRecord { key } => {
                            map.write().unwrap().remove(&key);
                            last_seen.remove(&key);
                        }
                        UpdateMsg::InsertRecord { key, value } => {
                            last_seen.insert(key.clone(), generation);
                            map.write().unwrap().insert(key, value);
                        }
                        UpdateMsg::Clear => {
                            map.write().unwrap().clear();
                            last_seen.clear();
                        }
                        UpdateMsg::Tick => {
                            generation += 1;
                            let gen = generation;
                            let mut map_w = map.write().unwrap();
                            last_seen.retain(|key, seen| {
                                if gen - *seen > DEBUG_TTL_FRAMES {
                                    map_w.remove(key);
                                    false
                                } else {
                                    true
                                }
                            });
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

/// Clear all debug messages.
pub fn debug_clear() {
    if let Some(sender) = DEBUG_MESSAGES.get() {
        let _ = sender.send(UpdateMsg::Clear);
    }
}

/// Advance the debug frame clock by one tick, evicting any key that has not been
/// re-recorded within the last [`DEBUG_TTL_FRAMES`] frames.
///
/// Call exactly once per frame, after all of the frame's debug has been
/// recorded. Replay does not call this (it clears and republishes each logged
/// snapshot wholesale), so eviction only runs on the live/sim executor loop.
pub fn debug_tick() {
    if let Some(sender) = DEBUG_MESSAGES.get() {
        let _ = sender.send(UpdateMsg::Tick);
    }
}

/// Tracks loose player keys we've already warned about, so the log isn't spammed.
static WARNED_LOOSE_KEYS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

/// Returns true if `key` looks like a loose, un-grouped player tag, i.e. it
/// starts with `p<digits>` followed by end-of-key or a `.` separator
/// (e.g. `p0`, `p3.control.target`). Such keys must instead be grouped under a
/// team via `PlayerContext` so they become `team_{color}.p{id}.*`.
fn is_loose_player_key(key: &str) -> bool {
    let rest = match key.strip_prefix('p') {
        Some(r) => r,
        None => return false,
    };
    let digits_end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if digits_end == 0 {
        return false; // no digits after 'p'
    }
    matches!(rest.as_bytes().get(digits_end), None | Some(b'.'))
}

/// Record a debug message.
///
/// Loose player keys (`p{id}.*`) are rejected: every player-associated tag must
/// be grouped under its team via `PlayerContext` (`team_{color}.p{id}.*`).
pub fn debug_record(key: impl Into<String>, value: DebugValue) {
    let key = key.into();
    if is_loose_player_key(&key) {
        let warned = WARNED_LOOSE_KEYS.get_or_init(|| Mutex::new(HashSet::new()));
        let is_new = warned.lock().unwrap().insert(key.clone());
        if is_new {
            log::warn!(
                "rejected loose player debug tag {:?}: player tags must be grouped under a team \
                 (use PlayerContext so the key becomes `team_{{color}}.p{{id}}.*`)",
                key
            );
        }
        return;
    }
    if let Some(sender) = DEBUG_MESSAGES.get() {
        let _ = sender.send(UpdateMsg::InsertRecord { key, value });
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

/// Record a debug message with a tree node.
pub fn debug_tree_node(
    key: impl Into<String>,
    name: impl Into<String>,
    id: impl Into<String>,
    children_ids: impl Into<Vec<String>>,
    is_active: bool,
    node_type: impl Into<String>,
    internal_state: Option<String>,
    additional_info: Option<String>,
) {
    debug_record(
        key.into(),
        DebugValue::Shape(DebugShape::TreeNode {
            name: name.into(),
            id: id.into(),
            children_ids: children_ids.into(),
            is_active,
            node_type: node_type.into(),
            internal_state,
            additional_info,
        }),
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

#[cfg(test)]
mod tests {
    use super::is_loose_player_key;

    #[test]
    fn loose_player_keys_are_detected() {
        assert!(is_loose_player_key("p0"));
        assert!(is_loose_player_key("p3.control.target"));
        assert!(is_loose_player_key("p12.ilqr.cost"));
    }

    #[test]
    fn grouped_and_global_keys_are_allowed() {
        assert!(!is_loose_player_key("team_Blue.p0.foo"));
        assert!(!is_loose_player_key("team_Yellow.p3.control.target"));
        assert!(!is_loose_player_key("game_state"));
        assert!(!is_loose_player_key("dt"));
        // `p` not followed by digits is not a player key.
        assert!(!is_loose_player_key("path.length"));
        assert!(!is_loose_player_key("ball_placement_target_0"));
    }
}
