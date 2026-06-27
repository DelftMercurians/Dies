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
/// Carrier rel-x beyond which we're in the attacking final third. Here the strict
/// "supporter must be forward" gate no longer fits (there is little field left
/// ahead), so we also accept a wide, roughly-level supporter as a cross/cutback
/// outlet — see [`best_kickahead_target`].
pub const FINAL_THIRD_X: f64 = 2000.0;
/// Minimum lateral (|Δy|) separation between carrier and supporter for the
/// supporter to count as a final-third cross/cutback outlet.
pub const CROSS_MIN_LATERAL: f64 = 800.0;
/// How far *behind* the carrier (along x) a wide final-third outlet may still sit
/// and remain pass-eligible (a cutback). Strictly-forward outlets are unaffected.
pub const CROSS_BACK_MARGIN: f64 = 600.0;
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
/// Distance to the ball below which the capture is "committing": the pickup
/// skill is making its final drive, so the tight `PICKUP_TIMEOUT` applies
/// instead of the generous `APPROACH_TIMEOUT` used while still traversing.
pub const CAPTURE_PICKUP_DIST: f64 = 500.0;
/// Distance to the target area at which a dribble is considered arrived. Kept
/// below the correction step so a corrective dribble is a real move, not an
/// instant "already there" that would spin the replan loop.
pub const DRIBBLE_ARRIVE_DIST: f64 = 150.0;
/// Ball displacement from the engagement point that aborts any waypoint.
pub const BALL_MOVED_DIST: f64 = 2000.0;
/// Capture timeouts: generous while traversing toward the ball, tight once
/// within `CAPTURE_PICKUP_DIST` (the pickup skill's final committing drive).
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
/// Markers never cross to the opponent half: their target x is capped here
/// (team-relative, so 0 = midfield). Keeps defenders home instead of chasing
/// opponents up-field.
pub const MARK_MAX_X: f64 = 0.0;
/// Don't generate a mark role for opponents below this threat (0..1) to our goal
/// — a zero-importance mark is still assignable and would lure an up-field robot
/// off its defensive duty.
pub const MARK_MIN_THREAT: f64 = 0.05;
/// Corridor width for the marking lane-openness term.
pub const MARK_LANE_CORRIDOR: f64 = 500.0;
/// Number of offensive support roles (split across flanks).
pub const SUPPORT_COUNT: usize = 2;
/// When the ball is in the opponent half and our own goal is not threatened we
/// commit more bodies forward as supporters (keeper + one shadow stay home).
pub const SUPPORT_ATTACK_COUNT: usize = 3;
/// Ball rel-x beyond which support placement/staffing switches to the aggressive
/// attacking mode (0 = ball in the opponent half).
pub const SUPPORT_ATTACK_BALL_X: f64 = 0.0;
/// Own-goal threat below which it is safe to commit the extra attacking
/// supporters (above this we keep the conservative `SUPPORT_COUNT`).
pub const SUPPORT_ATTACK_MAX_THREAT: f64 = 0.35;
/// Lateral bands (fraction of half-width) for attacking support placement. Wider
/// than the conservative grid so the outer candidates clear the box keepout and
/// flank the opponent goal for a cross/cutback.
pub const SUPPORT_FLANK_Y_FRACS: [f64; 3] = [0.46, 0.62, 0.78];
/// How far in front of the opponent goal line the most advanced attacking flank
/// candidate sits, so a deep carrier still gets a supporter level with / ahead.
pub const SUPPORT_GOAL_LINE_SETBACK: f64 = 400.0;
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
/// Support importance while attacking (ball in the opponent half, own goal safe).
/// Above the weak-mark / spread range so surplus bodies win the assignment as
/// supporters and push up instead of loitering at midfield.
pub const IMP_SUPPORT_ATTACK: f64 = 5.0;
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
/// far along the arc toward the posts the keeper will travel. Also bounds the
/// guard zone's angular span.
pub const KEEPER_ARC_MAX_ANGLE: f64 = std::f64::consts::FRAC_PI_4; // 45°
/// Radial breathing room (mm) added beyond `KEEPER_ARC_RADIUS` for the guard
/// zone's outer edge, so position noise doesn't peg the keeper at the boundary.
/// The aggressive control profile itself (gains, speed/accel caps, ORCA-off)
/// lives in the `GoToBounded` executor skill.
pub const KEEPER_ZONE_RADIUS_SLACK: f64 = 50.0;

// Ball-clearing behaviour.
/// Clear only when the ball is essentially stopped (mm/s).
pub const CLEAR_SPEED_LIMIT: f64 = 300.0;
/// A stopped ball this far inside the penalty area's field-facing edges (front +
/// sides) is the keeper's to clear. No field robot may enter the box (they're
/// held ~200mm clear of every edge by the keepout), so the margin is small and
/// uniform — a larger one would leave the box corners/edges in a dead zone where
/// neither the keeper nor a field robot can reach the ball.
pub const CLEAR_BALL_MARGIN: f64 = 50.0;
/// Abort the clear if the keeper charges out past the *front* (field-facing) edge
/// by this much — that's the one direction where leaving means abandoning the
/// goal. The side/goal-line edges are unconstrained so the keeper can reach a
/// ball in the box corners.
pub const CLEAR_EXIT_MARGIN: f64 = 120.0;
/// Minimum |y| of the clear target, so the keeper never kicks straight up the
/// middle in front of our own goal.
pub const CLEAR_TARGET_MIN_Y: f64 = 1500.0;
/// Downfield x of the clear target (toward the opponent half).
pub const CLEAR_TARGET_X: f64 = 0.0;
