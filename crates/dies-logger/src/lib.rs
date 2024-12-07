mod log_codec;
mod logger;
mod playback;

use dies_core::{DebugColor, DebugMap, DebugShape, WorldFrame};
pub use log_codec::*;
pub use logger::*;
pub use playback::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub(crate) enum TestData {
    A { color: DebugColor },
    B { shape: DebugShape },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataLog {
    World(WorldFrame),
    Debug(DebugMap),
}
