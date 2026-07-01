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

/// A physical contest over the ball, *orthogonal* to ownership: at least one
/// robot from each team is crowding a near-stationary ball. Fires even while the
/// ball reads `Owned` (our breakbeam latches ownership the instant we touch it,
/// hiding an opponent pressing from the far side — the deadlock the strategy
/// otherwise can't see).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Default)]
#[typeshare]
pub struct BallContest {
    /// Robots from *both* teams within `contest_dist` of the ball this frame.
    pub near: Vec<TeamPlayerId>,
}

/// Stable, debounced possession with a staleness flag.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Default)]
#[typeshare]
pub struct Possession {
    pub state: PossessionState,
    /// True while the belief is being coasted through a vision/contact dropout
    /// (held, but not currently backed by a fresh observation).
    pub stale: bool,
    /// Set when the ball is being physically contested (see [`BallContest`]).
    /// `None` in the common, uncontested case. Independent of `state`.
    pub contest: Option<BallContest>,
}

/// Tunable parameters for the unified possession metric. One central home for
/// every threshold that used to be scattered across the strategy and the pass
/// coordinator. Distances in mm, times in seconds, speeds in mm/s.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
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
    /// An *opposing* robot within this distance of the ball counts as pressing it.
    /// Deliberately looser than `acquire_dist` (a presser sits ~one robot-radius
    /// out, so it never registers as a proximity owner — which is why contests
    /// were invisible). Used only by the contest signal, not ownership.
    pub contest_dist: f64,
    /// The ball must be slower than this for a contest to register — a genuine pin
    /// is ~stationary. Far tighter than `max_ball_speed`; a moving ball means
    /// someone is already winning, so there's nothing to resolve.
    pub contest_speed_max: f64,
    /// Frames a contest must persist before it commits (react fast — urgent).
    pub contest_enter_frames: u32,
    /// Frames an absent contest must persist before it clears (release slow — don't
    /// drop on a single jittery opponent-position frame).
    pub contest_exit_frames: u32,

    // --- ToF-backup breakbeam substitute (per-robot toggle in TeamSpecificSettings) ---
    // These thresholds are GLOBAL (same sensor geometry across robots); only the
    // per-robot enable lives on the team settings. The raw condition is
    //   `confidence > conf_th  &&  |x| < x_th  &&  y < y_th`
    // (x/y/confidence in raw ToF sensor units), latched through an asymmetric
    // enter/exit frame debounce (the Schmitt trigger, time domain).
    // TODO(valentin): replace the PLACEHOLDER defaults below with tuned numbers.
    /// ToF confidence must exceed this for the raw detection to hold (`c_th`).
    pub tof_backup_conf_th: f64,
    /// Raw detection requires `|tof_x| < x_th` (`x_th`, sensor units).
    pub tof_backup_x_th: f64,
    /// Raw detection requires `tof_y < y_th` (`y_th`, sensor units).
    pub tof_backup_y_th: f64,
    /// Frames the raw condition must hold before the latch turns ON (`N`).
    pub tof_backup_enter_frames: u32,
    /// Frames the raw condition must fail before the latch turns OFF (`M`).
    pub tof_backup_exit_frames: u32,
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
            contest_dist: 220.0,
            contest_speed_max: 200.0,
            contest_enter_frames: 2,
            contest_exit_frames: 5,
            tof_backup_conf_th: 50.0,
            tof_backup_x_th: 10.0,
            tof_backup_y_th: 15.0,
            tof_backup_enter_frames: 2,
            tof_backup_exit_frames: 10,
        }
    }
}
