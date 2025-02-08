use std::sync::Arc;

use crate::{SettingsHandle, WorldFrame};

pub struct BaseCtx {
    world: Arc<WorldFrame>,
    settings: SettingsHandle,
    debug: DebugHandle,
}
