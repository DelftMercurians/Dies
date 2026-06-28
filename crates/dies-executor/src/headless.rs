//! Headless, deterministic, faster-than-realtime self-play.
//!
//! Types for running one A-vs-B match in simulation with no UI, no wall-clock
//! pacing, and blocking strategy IPC, so the whole match is reproducible from
//! `(seed, blue_strategy, yellow_strategy)` on the same binary. The loop itself
//! lives on [`crate::Executor::run_headless`].

use dies_core::{FieldSnapshot, TeamColor};
use serde::Serialize;

/// Inputs for a single headless self-play match.
#[derive(Debug, Clone)]
pub struct HeadlessConfig {
    /// Strategy binary name for the blue team.
    pub blue_strategy: String,
    /// Strategy binary name for the yellow team.
    pub yellow_strategy: String,
    /// RNG seed driving initial pose jitter (and any other seeded variation).
    pub seed: u64,
    /// Match length in simulated seconds.
    pub duration_secs: f64,
    /// Optional early stop once the combined score reaches this many goals.
    pub max_goals: Option<u32>,
    /// If set, write a full Dies binary log of the match under this base
    /// directory (one session dir per match), for offline analytics.
    pub log_dir: Option<std::path::PathBuf>,
    /// If set, seed the field from this snapshot before the match starts: robot
    /// poses + ball are teleported into place, and the game state is forced to
    /// the snapshot's `game_state` (or to `Run`/free play if the snapshot has
    /// none) instead of running the normal kickoff sequence.
    pub initial_snapshot: Option<FieldSnapshot>,
}

/// Why a headless match ended.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EndReason {
    /// Reached the configured sim-time limit.
    TimeLimit,
    /// Reached the configured combined-goal cap.
    GoalCap,
    /// A strategy died or stopped responding (match aborted, result partial).
    StrategyError(String),
    /// A `Stop` control message was received.
    Stopped,
}

/// A goal scored during the match, with the simulated time it occurred.
#[derive(Debug, Clone, Serialize)]
pub struct GoalEvent {
    pub t_secs: f64,
    pub team: TeamColor,
}

/// The outcome of a single headless self-play match. Serializable to JSON; for
/// a fixed binary, same `(seed, blue_strategy, yellow_strategy)` produces a
/// byte-identical result.
#[derive(Debug, Clone, Serialize)]
pub struct MatchResult {
    pub blue_strategy: String,
    pub yellow_strategy: String,
    pub seed: u64,
    pub blue_score: u32,
    pub yellow_score: u32,
    /// Final simulated time in seconds.
    pub duration_secs: f64,
    pub goals: Vec<GoalEvent>,
    pub end_reason: EndReason,
    /// Path to the match log session directory, if logging was enabled.
    pub log_path: Option<String>,
    /// Rolling fingerprint of the whole simulated trajectory (ball + player
    /// poses, every frame). Identical for identical `(seed, strategies)` runs;
    /// differs as soon as the match diverges. Lets two runs be compared for
    /// determinism even when the score is the same (e.g. a 0-0 draw).
    pub trace_hash: u64,
}
