//! Tunable parameters for the Concerto strategy.
//!
//! All physical constants live here so behaviour can be tuned in one place.
//! Distances are in mm, times in seconds, speeds in mm/s, accelerations in mm/s².

// ── Robot motion model (used by momentum-aware redirect cost) ───────────────
/// Assumed top speed for time-to-target estimates.
pub const V_MAX: f64 = 3000.0;
/// Assumed acceleration for time-to-target estimates.
pub const A_MAX: f64 = 3000.0;

// Possession classification now lives in the framework (`dies-world`), tuned via
// `PossessionConfig` in the executor settings — not here.

// ── Planner ─────────────────────────────────────────────────────────────────
// Offense policy: dribbling is unreliable and bounded by the 1m excessive-dribbling
// rule, so the PRIMARY advancement action is a kick-ahead (Shoot at a supporter or
// open forward space). Dribbling is only a small, rare correction.
/// Corridor width for the "clear shot to goal" check.
pub const CLEAR_SHOT_CORRIDOR: f64 = 700.0;
/// Maximum distance to goal from which we attempt a direct shot.
pub const SHOOT_RANGE: f64 = 4000.0;
/// A kick-ahead target teammate must be at least this far forward of the carrier.
pub const SUPPORTER_FWD_MARGIN: f64 = 400.0;
/// Corridor width for the carrier→supporter lane-openness check.
pub const KICK_LANE_CORRIDOR: f64 = 400.0;
/// Minimum lane openness (0..1) for a supporter to be a viable kick-ahead target.
pub const SUPPORTER_MIN_OPENNESS: f64 = 0.5;
/// Lead the kick-ahead this far past the supporter toward goal (kick into space).
pub const SUPPORTER_LEAD: f64 = 350.0;
/// Per-step distance of a corrective dribble.
pub const DRIBBLE_CORRECTION_STEP: f64 = 250.0;
/// Hard cap on how far the ball may be carried from the contact point before we
/// must kick (well under the 1m excessive-dribbling limit).
pub const DRIBBLE_CORRECTION_LIMIT: f64 = 350.0;
/// Maximum distance to the ball for a steal to be worth attempting (M1 crude gate).
pub const STEAL_MAX_DIST: f64 = 2500.0;
/// Lateral step a contested carrier strafes off the squeeze axis to break a pin
/// while keeping the ball (the keep-possession escape). Short — it only needs to
/// move the opponent out from between us and goal, after which we replan normally.
pub const ESCAPE_STEP: f64 = 300.0;
/// How long a robot is excluded from re-selection after a NoProgress failure.
pub const NOPROGRESS_TTL: f64 = 1.0;

// ── Driver ──────────────────────────────────────────────────────────────────
/// Distance to the ball at which the capture switches from driving to picking up.
pub const CAPTURE_PICKUP_DIST: f64 = 500.0;
/// Distance to the target area at which a dribble is considered arrived. Kept
/// below the correction step so a corrective dribble is a real move, not an
/// instant "already there" that would spin the replan loop.
pub const DRIBBLE_ARRIVE_DIST: f64 = 150.0;
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
/// Corridor width for the ball→support lane-openness used to place supporters in
/// open outlets (rather than stranded behind opponents).
pub const SUPPORT_LANE_CORRIDOR: f64 = 500.0;
/// A ball contest with at least this much threat to our goal relieves one shadow
/// (the contesting plan robot stands in for it). Keeps shadows off the snatcher.
pub const SHADOW_RELIEF_THREAT: f64 = 0.4;
/// Importance ladder (points; converted to seconds via SEC_PER_IMPORTANCE).
pub const IMP_SHADOW_BASE: f64 = 8.0;
pub const IMP_MARK_BASE: f64 = 6.0;
pub const IMP_SUPPORT: f64 = 3.0;
pub const IMP_SPREAD: f64 = 0.5;
/// Coverage accounting: radius around a plan robot's ball contest within which
/// formation roles are de-prioritised (soft suppression, avoids clustering).
pub const SUPPRESS_RADIUS: f64 = 1400.0;
/// Max importance penalty applied at the centre of a suppression zone. Set above
/// the shadow/mark bases so a redundant role on the contested ball decisively
/// loses to anything further out (clustering fix).
pub const IMP_SUPPRESS: f64 = 10.0;
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
/// Radius of the keeper's positioning arc, measured from the goal centre. At the
/// central (square-on) position the keeper sits this far in front of the line.
pub const KEEPER_ARC_RADIUS: f64 = 400.0;
/// Maximum angular excursion of the keeper off straight-out (radians). Caps how
/// far along the arc toward the posts the keeper will travel.
pub const KEEPER_ARC_MAX_ANGLE: f64 = std::f64::consts::FRAC_PI_4; // 45°

// Aggressive control profile for the keeper — fed to `ControlOverride`.
/// Snappiness dial: scales the position approach gain and terminal braking.
pub const KEEPER_AGGRESSIVENESS: f64 = 1.5;
/// Explicit terminal active-braking gain (decoupled from aggressiveness).
pub const KEEPER_BRAKE_GAIN: f64 = 2.0;
/// Keeper speed cap (mm/s). Tight, predictable line motion.
pub const KEEPER_SPEED_LIMIT: f64 = 2500.0;

// Ball-clearing behaviour.
/// Clear only when the ball is essentially stopped (mm/s).
pub const CLEAR_SPEED_LIMIT: f64 = 300.0;
/// Ball must be inside the penalty area by at least this margin before the keeper
/// commits to a clear, so the whole pickup maneuver stays inside the box.
pub const CLEAR_INNER_MARGIN: f64 = 250.0;
/// Abort the clear if the keeper body comes within this distance of the box edge.
pub const CLEAR_EXIT_MARGIN: f64 = 120.0;
/// Minimum |y| of the clear target, so the keeper never kicks straight up the
/// middle in front of our own goal.
pub const CLEAR_TARGET_MIN_Y: f64 = 1500.0;
/// Downfield x of the clear target (toward the opponent half).
pub const CLEAR_TARGET_X: f64 = 0.0;
