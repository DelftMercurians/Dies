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
pub mod replay;
mod schema;
pub mod worker;
mod writer;

pub use frame::side_assignment_str;
pub use meta::{MetaJson, FORMAT_VERSION};
pub use worker::set_console_observer;
