//! Executable skills for the strategy-controlled path.
//!
//! This module contains streamlined skill implementations that work with
//! the [`SkillExecutor`](crate::control::skill_executor::SkillExecutor).
//!
//! These skills implement the [`ExecutableSkill`](crate::control::skill_executor::ExecutableSkill)
//! trait and support:
//! - Parameter updates while running
//! - Status reporting
//! - Clean completion semantics

mod dribble;
mod go_to_bounded;
mod go_to_pos;
mod handle_ball;
mod receive;
mod reflex_receive;
mod shoot;
mod snatch;

use std::collections::HashMap;

use dies_core::TunableSpec;

pub use dribble::DribbleSkill;
pub use go_to_bounded::GoToBoundedSkill;
pub use go_to_pos::GoToPosSkill;
pub use handle_ball::HandleBallSkill;
pub use receive::ReceiveSkill;
pub use reflex_receive::ReflexReceiveSkill;
pub use shoot::ShootSkill;
pub use snatch::SnatchSkill;

/// Every skill module that declares a `tunables!` block. Adding a skill with
/// tunables = add it here once.
macro_rules! for_each_tunable_module {
    ($mac:ident) => {
        $mac!(dribble);
        $mac!(go_to_bounded);
        $mac!(go_to_pos);
        $mac!(handle_ball);
        $mac!(receive);
        $mac!(reflex_receive);
        $mac!(shoot);
        $mac!(snatch);
    };
}

/// Collect the code-generated UI metadata for every skill tunable.
pub fn all_skill_tunable_specs() -> Vec<TunableSpec> {
    let mut specs = Vec::new();
    macro_rules! collect {
        ($m:ident) => {
            specs.extend($m::__tunable_specs());
        };
    }
    for_each_tunable_module!(collect);
    specs
}

/// Apply the persisted overrides into the global tunable cells. Every cell is
/// reset to its compile-time default first, then overrides are applied, so this
/// is idempotent and a cleared key reverts to the default.
pub fn apply_skill_tunables(overrides: &HashMap<String, f64>) {
    macro_rules! reset {
        ($m:ident) => {
            $m::__tunable_reset();
        };
    }
    for_each_tunable_module!(reset);

    for (key, value) in overrides {
        macro_rules! try_set {
            ($m:ident) => {
                if $m::__tunable_set(key, *value) {
                    continue;
                }
            };
        }
        for_each_tunable_module!(try_set);
    }
}
