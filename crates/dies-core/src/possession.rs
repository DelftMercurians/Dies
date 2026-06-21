//! Unified ball-possession types.
//!
//! Possession is computed in exactly one place (`dies-world`) by fusing
//! ball-position proximity with the (position-gated) breakbeam sensor into a
//! single latched signal. These are the plain data types that result rides on;
//! the stateful tracker lives in `dies-world`. Strategies see a team-relative
//! view (see `dies-strategy-protocol`); the framework consumes this absolute one.

use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::TeamPlayerId;

/// Who controls the ball, in absolute (team-tagged) terms.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Default)]
#[serde(tag = "type", content = "data")]
#[typeshare]
pub enum PossessionState {
    /// Nobody clearly controls the ball.
    #[default]
    Loose,
    /// A single robot confidently controls the ball.
    Owned { owner: TeamPlayerId },
    /// Multiple robots are equally plausible owners and can't be disambiguated.
    Contested { candidates: Vec<TeamPlayerId> },
}

impl PossessionState {
    /// The confident single owner, if any (`None` for `Loose`/`Contested`).
    pub fn owner(&self) -> Option<TeamPlayerId> {
        match self {
            PossessionState::Owned { owner } => Some(*owner),
            _ => None,
        }
    }
}

/// Stable, debounced possession with a staleness flag.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Default)]
#[typeshare]
pub struct Possession {
    pub state: PossessionState,
    /// True while the belief is being coasted through a vision/contact dropout
    /// (held, but not currently backed by a fresh observation).
    pub stale: bool,
}

/// Tunable parameters for the unified possession metric. One central home for
/// every threshold that used to be scattered across the strategy and the pass
/// coordinator. Distances in mm, times in seconds, speeds in mm/s.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct PossessionConfig {
    /// Ball-to-robot distance at which a robot *gains* possession (near band).
    pub acquire_dist: f64,
    /// Looser distance at which the current owner *retains* possession. Must be
    /// `>= acquire_dist`; the gap is the hysteresis that prevents oscillation.
    pub release_dist: f64,
    /// Breakbeam is ignored unless the ball is within this distance of the robot
    /// (rejects spurious sensor triggers with the ball nowhere near).
    pub breakbeam_gate_dist: f64,
    /// Frames a proximity-only possession change must persist before it commits.
    /// Breakbeam-confirmed gains bypass this and commit in one frame.
    pub acquire_frames: u32,
    /// If the second-closest candidate is within this margin of the closest, the
    /// state is `Contested` instead of `Owned`.
    pub ambiguity_margin: f64,
    /// A ball faster than this can't be claimed by proximity (breakbeam still can,
    /// for a robot genuinely carrying it).
    pub max_ball_speed: f64,
    /// How long to hold the last stable possession while the ball is undetected
    /// before decaying to `Loose`. The belief is marked `stale` while holding.
    pub hold_secs: f64,
    /// After a commanded kick, suppress the kicker's *proximity* re-acquisition for
    /// this long (breakbeam re-acquisition is never suppressed).
    pub release_suppress_secs: f64,
}

impl Default for PossessionConfig {
    fn default() -> Self {
        Self {
            acquire_dist: 120.0,
            release_dist: 180.0,
            breakbeam_gate_dist: 200.0,
            acquire_frames: 4,
            ambiguity_margin: 60.0,
            max_ball_speed: 1000.0,
            hold_secs: 0.25,
            release_suppress_secs: 0.2,
        }
    }
}
