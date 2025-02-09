use std::sync::Arc;

use crate::{
    debug::{DebugColor, DebugService, DebugShape, DebugValue},
    Settings, SettingsHandle, Vector2, WorldFrame,
};

use super::WorldView;

#[derive(Clone)]
pub struct BaseCtx {
    world: Arc<WorldFrame>,
    settings: SettingsHandle,
    debug: DebugService,
    debug_subkey: String,
}

impl BaseCtx {
    fn debug_send(&self, key: &str, value: Option<DebugValue>) {
        let key = format!("{}.{}", self.debug_subkey, key);
        self.debug.send(&key, value);
    }
}

impl WorldView for BaseCtx {
    fn world_frame(&self) -> &WorldFrame {
        &self.world
    }
}

pub trait BaseContext {
    fn settings_manager(&self) -> &SettingsHandle;

    fn dbg_send(&self, key: &str, value: Option<DebugValue>);

    /// Create a new scoped context
    fn dbg_scoped(&self, subkey: &str) -> Self;

    fn settings<T: Settings + 'static>(&self) -> Arc<T> {
        self.settings_manager().get::<T>()
    }

    fn dbg_line(&self, key: &str, start: Vector2, end: Vector2, color: DebugColor) {
        self.dbg_send(
            key,
            Some(DebugValue::Shape(DebugShape::Line { start, end, color })),
        );
    }

    fn dbg_circle(&self, key: &str, center: Vector2, radius: f64, color: DebugColor) {
        self.dbg_send(
            key,
            Some(DebugValue::Shape(DebugShape::Circle {
                center,
                radius,
                fill: Some(color),
                stroke: None,
            })),
        );
    }

    fn dbg_cross(&self, key: &str, center: Vector2, color: DebugColor) {
        self.dbg_send(
            key,
            Some(DebugValue::Shape(DebugShape::Cross { center, color })),
        );
    }

    /// Record a numeric value
    fn dbg_value(&self, key: &str, value: f64) {
        self.dbg_send(key, Some(DebugValue::Number(value)));
    }

    /// Record a string value
    fn dbg_string(&self, key: &str, value: impl Into<String>) {
        self.dbg_send(key, Some(DebugValue::String(value.into())));
    }

    fn dbg_clear(&self, key: &str) {
        self.dbg_send(key, None);
    }

    fn dbg_red_line(&self, key: &str, start: Vector2, end: Vector2) {
        self.dbg_line(key, start, end, DebugColor::Red);
    }

    fn dbg_green_line(&self, key: &str, start: Vector2, end: Vector2) {
        self.dbg_line(key, start, end, DebugColor::Green);
    }

    fn dbg_orange_line(&self, key: &str, start: Vector2, end: Vector2) {
        self.dbg_line(key, start, end, DebugColor::Orange);
    }

    fn dbg_purple_line(&self, key: &str, start: Vector2, end: Vector2) {
        self.dbg_line(key, start, end, DebugColor::Purple);
    }

    fn dbg_red_circle(&self, key: &str, center: Vector2, radius: f64) {
        self.dbg_circle(key, center, radius, DebugColor::Red);
    }

    fn dbg_green_circle(&self, key: &str, center: Vector2, radius: f64) {
        self.dbg_circle(key, center, radius, DebugColor::Green);
    }

    fn dbg_orange_circle(&self, key: &str, center: Vector2, radius: f64) {
        self.dbg_circle(key, center, radius, DebugColor::Orange);
    }

    fn dbg_purple_circle(&self, key: &str, center: Vector2, radius: f64) {
        self.dbg_circle(key, center, radius, DebugColor::Purple);
    }

    fn dbg_red_cross(&self, key: &str, center: Vector2) {
        self.dbg_cross(key, center, DebugColor::Red);
    }

    fn dbg_green_cross(&self, key: &str, center: Vector2) {
        self.dbg_cross(key, center, DebugColor::Green);
    }

    fn dbg_orange_cross(&self, key: &str, center: Vector2) {
        self.dbg_cross(key, center, DebugColor::Orange);
    }

    fn dbg_purple_cross(&self, key: &str, center: Vector2) {
        self.dbg_cross(key, center, DebugColor::Purple);
    }
}

impl BaseContext for BaseCtx {
    fn settings_manager(&self) -> &SettingsHandle {
        &self.settings
    }

    fn dbg_send(&self, key: &str, value: Option<DebugValue>) {
        self.debug.send(key, value);
    }

    fn dbg_scoped(&self, subkey: &str) -> Self {
        Self {
            world: self.world.clone(),
            settings: self.settings.clone(),
            debug: self.debug.clone(),
            debug_subkey: format!("{}.{}", self.debug_subkey, subkey),
        }
    }

    fn dbg_clear(&self, key: &str) {
        self.debug_send(key, None);
    }
}
pub struct BaseCtxBuilder {
    settings: SettingsHandle,
    debug: DebugService,
}

impl BaseCtxBuilder {
    pub fn new(settings: SettingsHandle, debug: DebugService) -> Self {
        Self { settings, debug }
    }

    pub fn build(self, world: Arc<WorldFrame>) -> BaseCtx {
        BaseCtx {
            world,
            settings: self.settings,
            debug: self.debug,
            debug_subkey: "".to_string(),
        }
    }
}
