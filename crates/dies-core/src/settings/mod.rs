pub mod descriptor;
mod manager;

use descriptor::TypeDesc;
pub use manager::SettingsHandle;

use serde::{Deserialize, Serialize};

/// Core trait for settings types
pub trait Settings: Send + Sync + Clone + Default + Serialize + for<'de> Deserialize<'de> {
    /// Get the **unique** name of the settings type
    ///
    /// # Safety
    ///
    /// Two setting objects sharing the same name is undefined behavior and can cause
    /// data corruption and panics.
    fn name() -> &'static str;

    /// Get the descriptor for this settings type
    fn descriptor() -> TypeDesc;
}
