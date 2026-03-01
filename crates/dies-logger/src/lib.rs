mod log_codec;
mod logger;
mod playback;

use dies_core::{DebugMap, WorldData};
pub use log_codec::*;
pub use logger::*;
pub use playback::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataLog {
    World(WorldData),
    Debug(DebugMap),
}
