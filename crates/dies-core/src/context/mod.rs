mod base;
mod debug;
mod settings;
mod team_ctx;
mod world_ctx;

pub use debug::*;
pub use settings::*;
pub use team_ctx::*;
pub use world_ctx::*;

/// The base context trait that all contexts must implement.
pub trait BaseContext: Clone + GeneralDebugContext + SettingsContext + WorldView {}

impl<T: Clone + GeneralDebugContext + SettingsContext + WorldView> BaseContext for T {}
