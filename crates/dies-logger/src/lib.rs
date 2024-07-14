mod log_codec;
mod playback;
mod logger;

use std::borrow::Cow;

use dies_core::{DebugMap, WorldData};
pub use log_codec::*;
pub use logger::*;
pub use playback::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum DataLogRef<'a> {
    World(Cow<'a, WorldData>),
    Debug(Cow<'a, DebugMap>),
}

#[derive(Debug, Clone)]
pub enum DataLog {
    World(WorldData),
    Debug(DebugMap),
}

impl<'a> From<&'a DataLog> for DataLogRef<'a> {
    fn from(data: &'a DataLog) -> Self {
        match data {
            DataLog::World(data) => DataLogRef::World(Cow::Borrowed(data)),
            DataLog::Debug(data) => DataLogRef::Debug(Cow::Borrowed(data)),
        }
    }
}

impl From<DataLogRef<'_>> for DataLog {
    fn from(data: DataLogRef) -> Self {
        match data {
            DataLogRef::World(data) => DataLog::World(data.into_owned()),
            DataLogRef::Debug(data) => DataLog::Debug(data.into_owned()),
        }
    }
}
