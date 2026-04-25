//! JS-scripted test harness for dies.
//!
//! Embeds a Boa JavaScript engine and executes a single `.js` scenario file
//! per run. The scenario drives robots through either the skill-command path
//! (same channel strategies use) or the direct `PlayerControlInput` path (same
//! channel manual override uses). See `/Users/mablin/.claude/plans/1-single-js-compiled-tulip.md`
//! for the design.

mod bridge;
mod capture;
mod log_bus;
mod primitives;
mod runtime;

pub use bridge::{SlotStore, TestFrameOutput};
pub use log_bus::{LogBus, TestLogEntry, TestLogLevel};
pub use primitives::{ExcitationAxis, ExcitationProfile};
pub use runtime::{
    PlayerControlSlot, ScenarioArtifact, ScenarioEnv, ScenarioMeta, TestDriver, TestEnv, TestStatus,
};
