//! Static "logo" formation — robots pose into a triangle (the Mercurians
//! play-button shape) pointing toward the opponent goal. Used in two places:
//! - the **warmup** param, to form up while robots are placed before a match;
//! - a real-match **Timeout**, as a cosmetic ready-formation.
//!
//! Slots are **ID-based**: a robot fills the slot for its player ID, so the shape
//! is deterministic and missing IDs simply leave their spot empty.
//!
//!   - id 0      → tip (front, nearest the opponent goal)
//!   - id 1, 2   → middle layer, top then bottom
//!   - id 3, 4, 5 → back layer (nearest our goal), top / middle / bottom
//!
//! Coordinates are team-relative (the executor transforms to absolute), centered
//! on our own half with the tip pointing toward the opponent goal (+x).

use dies_strategy_api::{PlayerId, Vector2};

/// Center of the triangle, team-relative (own half, x toward opponent goal).
const CENTER_X: f64 = -2000.0;
const CENTER_Y: f64 = 0.0;
/// Layer separation along x and row separation along y (mm).
const SPACING: f64 = 800.0;

/// Team-relative target for the given player ID, or `None` for IDs outside the
/// 6-slot triangle (those robots are left where they are).
pub fn slot_for(id: PlayerId) -> Option<Vector2> {
    let (cx, cy, s) = (CENTER_X, CENTER_Y, SPACING);
    let p = |x: f64, y: f64| Vector2::new(x, y);
    Some(match id.as_u32() {
        0 => p(cx + s, cy),           // tip — front, toward opponent goal
        1 => p(cx, cy + 0.5 * s),     // middle, top
        2 => p(cx, cy - 0.5 * s),     // middle, bottom
        3 => p(cx - s, cy + s),       // back, top
        4 => p(cx - s, cy),           // back, middle
        5 => p(cx - s, cy - s),       // back, bottom
        _ => return None,
    })
}
