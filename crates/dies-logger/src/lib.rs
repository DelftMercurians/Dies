mod log_codec;
mod logger;
mod playback;

use std::borrow::Cow;

use dies_core::{DebugColor, DebugMap, DebugShape, PlayerData, Vector2, WorldData};
pub use log_codec::*;
pub use logger::*;
pub use playback::*;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum DataLogRef<'a> {
    World(Cow<'a, WorldData>),
    Debug(Cow<'a, DebugMap>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub(crate) enum TestData {
    A { color: DebugColor },
    B { shape: DebugShape },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataLog {
    World(WorldData),
    Debug { map: DebugMap },
    Test(TestData),
}

// impl DataLog {
//     pub fn as_ref(&self) -> DataLogRef<'_> {
//         self.into()
//     }
// }

// impl<'a> From<&'a DataLog> for DataLogRef<'a> {
//     fn from(data: &'a DataLog) -> Self {
//         match data {
//             DataLog::World(data) => DataLogRef::World(Cow::Borrowed(data)),
//             DataLog::Debug { map: data } => DataLogRef::Debug(Cow::Borrowed(data)),
//         }
//     }
// }

// impl From<DataLogRef<'_>> for DataLog {
//     fn from(data: DataLogRef) -> Self {
//         match data {
//             DataLogRef::World(data) => DataLog::World(data.into_owned()),
//             DataLogRef::Debug(data) => DataLog::Debug { map: data.into_owned() },
//         }
//     }
// }
