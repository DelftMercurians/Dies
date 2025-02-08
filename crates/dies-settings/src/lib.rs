mod descriptor;

pub use descriptor::*;

pub use arc_swap;

use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::OnceLock;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SettingsError {
    #[error("Failed to serialize settings: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Failed to load settings: {0}")]
    LoadError(String),
    #[error("Failed to save settings: {0}")]
    SaveError(String),
    #[error("Settings persistence is not enabled")]
    PersistenceDisabled,
}


static SETTINGS_MANAGER: OnceLock<SettingsManager> = OnceLock::new();

pub struct SettingsManager {
    // Map of TypeId -> ArcSwap<dyn Any> to store settings instances
    settings: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    // Track persistence configuration
    persistence_enabled: bool,
    settings_dir: Option<PathBuf>,
}

impl SettingsManager {
    fn global() -> &'static SettingsManager {
        SETTINGS_MANAGER.get_or_init(|| SettingsManager {
            settings: HashMap::new(),
            persistence_enabled: false,
            settings_dir: None,
        })
    }

    // Register a new settings type
    pub fn register<T: Settings>(&mut self) {
        let type_id = TypeId::of::<T>();
        if !self.settings.contains_key(&type_id) {
            let settings = Arc::new(T::default());
            let swap = ArcSwap::new(settings);
            self.settings.insert(type_id, Box::new(swap));
        }
    }

    // Get settings instance
    pub fn get<T: Settings>() -> Arc<T> {
        let manager = Self::global();
        let type_id = TypeId::of::<T>();

        let swap = manager
            .settings
            .get(&type_id)
            .expect("Settings type not registered")
            .downcast_ref::<ArcSwap<T>>()
            .expect("Invalid settings type");

        swap.load_full()
    }

    // Update settings
    pub fn update<T: Settings>(new_value: T) -> Result<(), SettingsError> {
        let manager = Self::global();
        let type_id = TypeId::of::<T>();

        let swap = manager
            .settings
            .get(&type_id)
            .expect("Settings type not registered")
            .downcast_ref::<ArcSwap<T>>()
            .expect("Invalid settings type");

        // Store in memory
        swap.store(Arc::new(new_value.clone()));

        // Persist if enabled
        if manager.persistence_enabled {
            if let Some(dir) = &manager.settings_dir {
                // Save to file
                todo!("Implement persistence");
            } else {
                return Err(SettingsError::PersistenceDisabled);
            }
        }

        Ok(())
    }

    // Configure persistence
    pub fn set_persistence(enabled: bool) {
        let manager = Self::global();
        manager.persistence_enabled = enabled;
    }

    pub fn set_settings_dir(path: &Path) -> Result<(), SettingsError> {
        let manager = Self::global();
        manager.settings_dir = Some(path.to_path_buf());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_macros::Settings;
    use serde::{Deserialize, Serialize};
    use std::sync::OnceLock;

    #[derive(Debug, Clone, Serialize, Deserialize, Settings)]
    struct TestSettings {
        value: f64,
    }

    impl Default for TestSettings {
        fn default() -> Self {
            Self { value: 42.0 }
        }
    }

    #[test]
    fn test_basic_settings() {
        let settings = TestSettings::load();
        assert_eq!(settings.value, 42.0);

        // Create new settings value
        let new_settings = TestSettings { value: 43.0 };

        // Store the new value
        TestSettings::store(new_settings).unwrap();

        // Load the updated value
        let updated_settings = TestSettings::load();
        assert_eq!(updated_settings.value, 43.0);
    }
}
