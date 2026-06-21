//! Tunable parameters for the Concerto strategy.
//!
//! All physical constants live here so behaviour can be tuned in one place.
//! Distances are in mm, times in seconds, speeds in mm/s, accelerations in mm/s².

// ── Robot motion model (used by momentum-aware redirect cost) ───────────────
/// Assumed top speed for time-to-target estimates.
pub const V_MAX: f64 = 3000.0;
/// Assumed acceleration for time-to-target estimates.
pub const A_MAX: f64 = 3000.0;

// ── Possession classification ───────────────────────────────────────────────
/// A teammate without breakbeam is treated as possessing the ball within this range.
pub const WE_POSSESSION_DIST: f64 = 120.0;
/// An opponent is treated as possessing the ball within this range.
pub const OPP_POSSESSION_DIST: f64 = 150.0;
/// Frames a non-breakbeam possession change must persist before it commits.
pub const DEBOUNCE_FRAMES: u32 = 4;
/// How long to hold the last stable possession while the ball is undetected
/// before forcing `Loose`.
pub const POSSESSION_HOLD_SECS: f64 = 0.25;
/// A ball faster than this can't be "possessed" by proximity (breakbeam still
/// authoritative). Keeps a kicked/whizzing ball from registering as controlled.
/// Load-bearing for normal possession too — watch in sim.
pub const POSSESSION_MAX_BALL_SPEED: f64 = 1000.0;
/// After we command a kick, suppress proximity re-acquisition of `We(kicker)` for
/// this long (breakbeam re-acquisition is never suppressed).
pub const RELEASE_SUPPRESS_SECS: f64 = 0.2;

// ── Planner ─────────────────────────────────────────────────────────────────
/// Corridor width for the "clear shot to goal" check.
pub const CLEAR_SHOT_CORRIDOR: f64 = 700.0;
/// Maximum distance to goal from which we attempt a direct shot.
pub const SHOOT_RANGE: f64 = 4000.0;
/// How far toward the opponent goal a dribble waypoint advances the ball.
pub const DRIBBLE_ADVANCE: f64 = 1500.0;
/// Maximum distance to the ball for a steal to be worth attempting (M1 crude gate).
pub const STEAL_MAX_DIST: f64 = 2500.0;
/// How long a robot is excluded from re-selection after a NoProgress failure.
pub const NOPROGRESS_TTL: f64 = 1.0;

// ── Driver ──────────────────────────────────────────────────────────────────
/// Distance to the ball at which the capture switches from driving to picking up.
pub const CAPTURE_PICKUP_DIST: f64 = 500.0;
/// Distance to the target area at which a dribble is considered arrived.
pub const DRIBBLE_ARRIVE_DIST: f64 = 500.0;
/// Ball displacement from the engagement point that aborts any waypoint.
pub const BALL_MOVED_DIST: f64 = 2000.0;
/// Per-phase timeouts.
pub const APPROACH_TIMEOUT: f64 = 3.0;
pub const PICKUP_TIMEOUT: f64 = 2.0;
pub const DRIBBLE_TIMEOUT: f64 = 4.0;
pub const SHOOT_TIMEOUT: f64 = 2.0;

// ── Formation ─────────────────────────────────────────────────────────────
/// Threat ramp: distance to our goal at which threat is maximal / negligible.
pub const THREAT_GOAL_NEAR: f64 = 1500.0;
pub const THREAT_GOAL_FAR: f64 = 6000.0;
/// Shadow (goal-coverage) role count scales with threat between these bounds.
pub const SHADOW_MIN: usize = 1;
pub const SHADOW_MAX: usize = 3;
/// How far in front of our goal the shadow line sits.
pub const SHADOW_STANDOFF: f64 = 1500.0;
/// Marking standoff in front of the marked opponent (toward our goal).
pub const MARK_STANDOFF: f64 = 400.0;
/// Corridor width for the marking lane-openness term.
pub const MARK_LANE_CORRIDOR: f64 = 500.0;
/// Number of offensive support roles (split across flanks).
pub const SUPPORT_COUNT: usize = 2;
/// Push support points away from a nearby opponent within this range.
pub const SUPPORT_AVOID_RANGE: f64 = 800.0;
/// Importance ladder (points; converted to seconds via SEC_PER_IMPORTANCE).
pub const IMP_SHADOW_BASE: f64 = 8.0;
pub const IMP_MARK_BASE: f64 = 6.0;
pub const IMP_SUPPORT: f64 = 3.0;
pub const IMP_RECEIVER: f64 = 12.0;
pub const IMP_SPREAD: f64 = 0.5;
/// Primary tuning knob: seconds of redirect time one importance point is worth.
pub const SEC_PER_IMPORTANCE: f64 = 0.4;
/// Over-generate roles to this multiple of the assignable robot count.
pub const OVERGEN_FACTOR: f64 = 1.5;
/// Assignment recalculation cadence.
pub const RECALC_COOLDOWN: f64 = 0.18;
pub const RECALC_BG_PERIOD: f64 = 0.4;
/// A plan-context area is "moved" (a trigger) when it shifts more than this.
pub const PLAN_CTX_MOVE_EPS: f64 = 300.0;

// ── Goalkeeper ──────────────────────────────────────────────────────────────
/// How far in front of the goal line the keeper sits.
pub const KEEPER_DEPTH: f64 = 200.0;
