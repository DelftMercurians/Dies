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
/// Maximum distance to goal from which we attempt a direct shot.
pub const SHOOT_RANGE: f64 = 4000.0;

// ── Open-goal shot aiming ───────────────────────────────────────────────────
// Instead of only ever shooting at the goal *centre* (which a keeper tracking the
// bisector always covers), `geometry::best_shot` finds the widest open window in
// the goal mouth — every opponent projected as a shadow on the goal line — and
// aims at its midpoint. The decision to shoot is gated on the window's *angular*
// width, which is naturally distance-aware (a far shot needs a wider gap for the
// same aiming tolerance).
/// Ball radius (mm). Aim points are inset off the posts and opponent shadows by
/// this so the struck ball physically clears.
pub const BALL_RADIUS: f64 = 21.5;
/// Effective radius (mm) of a field opponent when projecting its goal-line shadow.
/// Robot radius (~90mm) plus a little slack for position noise.
pub const SHOT_ROBOT_RADIUS: f64 = 110.0;
/// Effective radius (mm) of the opponent *keeper* shadow. Inflated well beyond a
/// field robot: the keeper actively dives/slides to cover during the ball's flight,
/// so it blocks far more of the mouth than its static footprint suggests.
pub const SHOT_KEEPER_RADIUS: f64 = 280.0;
/// Weight on aiming *away* from the keeper when scoring open windows. Score is in
/// goal-line mm: a window this many mm farther from the keeper's projected position
/// is worth `+1mm` of open width per `1/bias` mm of distance. Picks the corner the
/// keeper must travel furthest to reach when both corners are open.
pub const SHOT_KEEPER_BIAS: f64 = 0.35;
/// Minimum angular width (radians) of the open window for the carrier to take a
/// direct shot. Below this we fall through to the kick-ahead / pass logic. The
/// angular gate scales with distance for free: ~0.10 rad is a ~130mm gap at 1.3m
/// but needs a ~400mm gap at 4m.
pub const SHOT_MIN_ANGLE: f64 = 0.10;
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
/// Backstop timeout (s) for a snatch attempt. The `snatch` skill has its own
/// internal timeout that normally fires first; this is a driver-level safety net
/// so a wedged attempt can't run forever before the planner gives up.
pub const SNATCH_TIMEOUT: f64 = 4.0;
/// Lateral step a contested carrier strafes off the squeeze axis to break a pin
/// while keeping the ball (the keep-possession escape). Short — it only needs to
/// move the opponent out from between us and goal, after which we replan normally.
pub const ESCAPE_STEP: f64 = 300.0;
/// How long a robot is excluded from re-selection after a NoProgress failure.
pub const NOPROGRESS_TTL: f64 = 1.0;

// ── Advancement hoof (kick to open space) ─────────────────────────────────────
// A full-power `DribbleShoot` kick fires at a fixed ~4000 mm/s and the ball rolls
// ~`KICK_SPEED / ball_damping` ≈ 4000 / 0.8 = 5000 mm before stopping. Aiming an
// advancement kick "at open space" a short distance ahead therefore overshoots and
// drives the ball out of bounds — a self-inflicted stoppage that hands the opponent
// a free kick. We aim instead at the kick's *resting point* so it stays in play.
/// Estimated distance the ball rolls after a full-power advancement kick.
pub const HOOF_TRAVEL: f64 = 5000.0;
/// Margin (mm) inside the field-of-play edge the resting point must stay within.
/// The sim flags the ball out ~100 mm inside the line; the extra slack absorbs
/// travel-estimate error so we don't graze the boundary.
pub const HOOF_BOUNDARY_MARGIN: f64 = 300.0;
/// Minimum forward (goalward) progress for an advancement hoof to be worth taking
/// instead of keeping the ball; below this the planner prefers a corrective dribble.
pub const HOOF_MIN_PROGRESS: f64 = 800.0;
/// Weight on landing-spot openness (distance to nearest opponent) when scoring hoof
/// directions, and the distance beyond which extra openness no longer helps.
pub const HOOF_OPEN_WEIGHT: f64 = 0.5;
pub const HOOF_OPEN_CAP: f64 = 2000.0;
/// A loose ball within this distance of a touchline or goal line is treated as a
/// boundary "rescue": the capture heading is biased to dribble the ball back
/// *inward* (into the field) instead of along/over the line. Set comfortably
/// above the ball radius + capture contact so a ball pinned on the line still
/// trips it. Pairs with the pickup skill's in-field staging clamp.
pub const BOUNDARY_RESCUE_MARGIN: f64 = 300.0;
/// During a boundary rescue, how strongly the toward-goal direction is blended
/// into the inward push (0 = straight inward, larger = more goalward). Keeps a
/// forward component while guaranteeing the dominant pull is away from the line.
pub const RESCUE_GOAL_BIAS: f64 = 0.6;

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
/// Lateral spacing between adjacent shadows in the goal-coverage wall. Kept just
/// under a robot width (~180mm) so neighbours overlap and leave no central gap —
/// the wall is built centre-out on the direct ball→goal line, so an even count
/// straddles the centre rather than splitting around it.
pub const SHADOW_SPACING: f64 = 170.0;
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
/// Minimum spacing between two supporters' target positions. Support roles are
/// placed greedily, each excluding candidates within this radius of an already-placed
/// supporter, so two never converge on the same outlet. Sized so distinct grid cells
/// survive the constraint while genuinely co-located spots are forbidden.
pub const SUPPORT_MIN_SEPARATION: f64 = 1500.0;
// ── Central box-runner (striker) ────────────────────────────────────────────
// v0 (and most simple defences) cover the centre of the mouth with only a 2-robot
// wall + an arc keeper that bias toward the ball — so the highest-value attacking
// position is central, just in front of the box: a cutback target and rebound
// crasher. When attacking we stage ONE supporter there (replacing a wide one, to
// keep the forward body-count balanced) so the planner has a central outlet to
// pass into and finish from close range instead of shooting from acute wing angles.
/// How far in front of the opponent penalty-area front edge the box-runner sits
/// (mm). Small + positive keeps it just outside the box (legal, and a step from a
/// point-blank finish) rather than parked on the goal line.
pub const BOX_RUNNER_FRONT_MARGIN: f64 = 250.0;
/// Lateral offset (mm) of the box-runner from centre, toward the side *away* from
/// the ball — it opens a cutback angle across the keeper rather than standing on
/// the ball's own line. Kept small so it stays within the high-value central zone.
pub const BOX_RUNNER_Y_OFFSET: f64 = 350.0;
/// Importance of the box-runner role while attacking. At/above `IMP_SUPPORT_ATTACK`
/// so the central outlet is reliably staffed when we commit forward.
pub const IMP_STRIKER: f64 = 5.5;

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
/// Capture worth-it gate: a robot more than this redirect time from the ball is
/// barred from the Capture slot, so the matcher never sends a robot that would
/// arrive far too late, and leaves the ball to Formation when no one can reach it.
/// Tuned via self-play + snapshot rollouts: the dominant knob trading possession
/// (longer → pursue more loose balls) against defensive shape (shorter → keep
/// bodies home). 1.5s recovers near-baseline ball ownership.
pub const CAPTURE_TIME_HORIZON: f64 = 1.5;
/// Importance of the Capture slot in the capturer assignment. Weighed against the
/// defensive roles a robot would otherwise fill: too low and we never commit
/// anyone in our own half, too high and we strip defenders. Decides *whether* to
/// send a capturer; the time horizon + opportunity cost decide *who*.
pub const CAPTURE_IMPORTANCE: f64 = 13.0;
/// Seconds the capturer leads the ball along its velocity (aim at an intercept,
/// not a stale point).
pub const CAPTURE_LEAD_TAU: f64 = 0.3;
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

// Shot-line intercept.
/// Ball speed (mm/s) above which, if the ball is heading toward our goal across
/// the goal mouth, the keeper switches from the cone-bisector position to sitting
/// on the shot line (the incoming trajectory) so it is in the ball's path
/// whichever corner the shot is aimed at. Below this the keeper tracks the cone
/// bisector as usual.
pub const KEEPER_INTERCEPT_SPEED: f64 = 800.0;
/// Lateral slack (mm) added beyond each post when deciding whether a fast ball's
/// trajectory counts as "on target" for the shot-line intercept. Covers shots
/// aimed just inside/around a post plus tracking noise.
pub const KEEPER_INTERCEPT_MOUTH_MARGIN: f64 = 300.0;

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

// ── Planner: in-field clamping ───────────────────────────────────────────────
/// Margin (mm) by which a kick-ahead pass/cross lead target is kept inside the
/// field boundary. A lead aims a pass into the space ahead of a receiver (toward
/// the opponent goal); for a deep or wide receiver that space can fall past the
/// goal line or a touchline, sending the ball out (stoppage + lost possession).
/// See [`crate::planner::Planner::clamp_in_field`].
pub const FIELD_LEAD_MARGIN: f64 = 250.0;
