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

pub trait GeneralDebugContext {
    fn line(&self, key: &str, start: Vector2, end: Vector2, color: DebugColor);

    fn circle(&self, key: &str, center: Vector2, radius: f64, color: DebugColor);

    fn cross(&self, key: &str, center: Vector2, color: DebugColor);

    /// Record a numeric value
    fn value(&self, key: &str, value: f64);

    /// Record a string value
    fn string(&self, key: &str, value: impl Into<String>);

    /// Create a new scoped context
    fn scoped(&self, subkey: &str) -> ScopedDebugCtx<Self>
    where
        Self: Clone,
    {
        ScopedDebugCtx {
            parent: self.clone(),
            subkey: subkey.to_string(),
        }
    }

    fn clear(&self, key: &str);

    fn clear_all(&self);

    fn red_line(&self, key: &str, start: Vector2, end: Vector2) {
        self.line(key, start, end, DebugColor::Red);
    }

    fn green_line(&self, key: &str, start: Vector2, end: Vector2) {
        self.line(key, start, end, DebugColor::Green);
    }

    fn orange_line(&self, key: &str, start: Vector2, end: Vector2) {
        self.line(key, start, end, DebugColor::Orange);
    }

    fn purple_line(&self, key: &str, start: Vector2, end: Vector2) {
        self.line(key, start, end, DebugColor::Purple);
    }

    fn red_circle(&self, key: &str, center: Vector2, radius: f64) {
        self.circle(key, center, radius, DebugColor::Red);
    }

    fn green_circle(&self, key: &str, center: Vector2, radius: f64) {
        self.circle(key, center, radius, DebugColor::Green);
    }

    fn orange_circle(&self, key: &str, center: Vector2, radius: f64) {
        self.circle(key, center, radius, DebugColor::Orange);
    }

    fn purple_circle(&self, key: &str, center: Vector2, radius: f64) {
        self.circle(key, center, radius, DebugColor::Purple);
    }

    fn red_cross(&self, key: &str, center: Vector2) {
        self.cross(key, center, DebugColor::Red);
    }

    fn green_cross(&self, key: &str, center: Vector2) {
        self.cross(key, center, DebugColor::Green);
    }

    fn orange_cross(&self, key: &str, center: Vector2) {
        self.cross(key, center, DebugColor::Orange);
    }

    fn purple_cross(&self, key: &str, center: Vector2) {
        self.cross(key, center, DebugColor::Purple);
    }
}

pub struct ScopedDebugCtx<T> {
    parent: T,
    subkey: String,
}

impl<T: GeneralDebugContext> ScopedDebugCtx<T> {
    pub fn scoped(&self, subkey: &str) -> Self
    where
        Self: Sized,
        T: Clone,
    {
        let subkey = format!("{}.{}", self.subkey, subkey);
        self.parent.scoped(&subkey)
    }

    fn key(&self, key: &str) -> String {
        format!("{}.{}", self.subkey, key)
    }
}

impl<T: GeneralDebugContext> GeneralDebugContext for ScopedDebugCtx<T> {
    fn clear(&self, key: &str) {
        self.parent.clear(&self.key(key));
    }

    fn clear_all(&self) {
        self.parent.clear_all();
    }

    fn line(&self, key: &str, start: Vector2, end: Vector2, color: DebugColor) {
        self.parent.line(&self.key(key), start, end, color);
    }

    fn circle(&self, key: &str, center: Vector2, radius: f64, color: DebugColor) {
        self.parent.circle(&self.key(key), center, radius, color);
    }

    fn cross(&self, key: &str, center: Vector2, color: DebugColor) {
        self.parent.cross(&self.key(key), center, color);
    }

    fn value(&self, key: &str, value: f64) {
        self.parent.value(&self.key(key), value);
    }

    fn string(&self, key: &str, value: impl Into<String>) {
        self.parent.string(&self.key(key), value);
    }
}

impl<T: Clone> Clone for ScopedDebugCtx<T> {
    fn clone(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            subkey: self.subkey.clone(),
        }
    }
}
