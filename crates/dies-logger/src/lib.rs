//! Columnar (Apache Arrow / Parquet) match logger.
//!
//! A log is a directory of per-table Arrow IPC streams keyed on a monotonic
//! `frame_id`, compacted to Parquet + a STORED `.dieslog` zip on close. The
//! public write API lives in [`worker`]; the read/replay side in [`replay`].

mod builders;
mod compact;
mod flatten;
mod frame;
mod meta;
mod schema;
mod writer;
pub mod replay;
pub mod worker;

pub use frame::side_assignment_str;
pub use meta::{MetaJson, FORMAT_VERSION};
