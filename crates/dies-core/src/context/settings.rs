use std::sync::Arc;

use crate::settings::{Settings, SettingsHandle};

pub trait SettingsContext {
    fn settings_manager(&self) -> &SettingsHandle;

    fn settings<T: Settings + 'static>(&self) -> Arc<T> {
        self.settings_manager().get::<T>()
    }
}

pub trait WriteSettingsContext: SettingsContext {
    fn update_settings<T: Settings + 'static>(&mut self, settings: T) {
        self.settings_manager().save::<T>(settings);
    }
}
