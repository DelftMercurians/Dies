//! Tunable parameters for the Concerto strategy.
//!
//! All physical constants live here so behaviour can be tuned in one place.
//! Distances are in mm, times in seconds, speeds in mm/s, accelerations in mm/s².

// ── Role master switches ─────────────────────────────────────────────────────
// AND'd with the UI's live per-role toggles (`role_*` params): a role runs only
// if both this and the UI switch are on.
pub const ROLE_WALL_ENABLED: bool = true;
pub const ROLE_MARK_ENABLED: bool = true; // TODO: only for ITA
pub const ROLE_SUPPORT_ENABLED: bool = true;
pub const ROLE_STRIKER_ENABLED: bool = true;
pub const ROLE_BALANCE_ENABLED: bool = false;
pub const ROLE_OUTLET_ENABLED: bool = false;

/// Master switch for the goalkeeper. When `false` the keeper robot is not driven
/// by the Guard/Clear machine and is no longer excluded from Formation — it folds
/// into the assignable pool and takes a regular field role like any other robot.
/// Purely a concerto-side knob (testing/scrimmage); the GC keeper designation and
/// executor-side keeper handling are unaffected. Route every role-purpose keeper
/// lookup through [`crate::our_keeper_id`] rather than `world.our_keeper_id()` so
/// this stays the single gate.
pub const GOALIE_ENABLED: bool = true;

/// Global pass-propensity knob — a single scalar multiplied into **every**
/// pass-success estimate concerto computes (forward-pass quality and
/// forward-outlet eligibility). It is the one global lever for "how much to
/// pass vs just shoot":
/// - `1.0` (default) — neutral; behaviour is unchanged from an ungated build.
/// - `(0, 1)` — progressively discount every pass's assessed success, so more
///   forward balls fall below [`SUPPORTER_MIN_OPENNESS`] and are downgraded to a
///   strike-finish or hoof. Biases toward shooting.
/// - `0.0` — passing is fully disabled: no forward pass ever clears its gate
///   (the corner one-timer has its own geometry gates and is unaffected). The
///   ball is always advanced by a strike.
/// - `> 1.0` — biases toward passing (marginal lanes clear their gate more often).
pub const PASS_SUCCESS_BASE: f64 = 1.0;
/// ── Rest defense (don't over-commit on the counter) ─────────────────────────
/// While attacking, all field robots but the keeper + box anchor pile forward,
/// so a turnover leaves the goal to the keeper + anchor alone and the wing wall
/// has nobody recoverable to staff it. The Balance role keeps ONE body home as
/// rest defense: it is placed on the ball→own-goal ray this far (mm) in front of
/// our goal — own-half/midfield, on the most likely counter lane — so on
/// turnover it is already there to take a wing of the wall while the deep
/// attackers recover.
pub const BALANCE_STANDOFF: f64 = 3500.0;
/// Importance of the rest-defense Balance role while attacking. Set above the
/// forward block (IMP_SUPPORT_ATTACK = 5.0, IMP_STRIKER = 5.5) so it wins exactly
/// one body, but below the box anchor (8.0 × 0.8 = 6.4), so it never outranks
/// the genuine last line. Sits above the discounted Mark (≤4), which is
/// intended: rest-defense shape beats man-marking in the current tuning. Only
/// emitted when `attacking`.
pub const IMP_BALANCE: f64 = 5.0;

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

/// Range within which a carrier that has the ball but no *clean* aimed-shot
/// window will fall back to a one-touch Strike at the goal instead of recycling
/// the ball backward. Tighter than [`SHOOT_RANGE`] so the reflex finish only
/// fires from genuine shooting distance (final third / edge of the box), never as
/// a long midfield hoof. The aimed Shoot's kick gate refuses a `lane_blocked`
/// corridor and orbits forever from wide/covered angles; the Strike fires through
/// it, turning stalled attacking possessions into shots.
pub const STRIKE_FINISH_RANGE: f64 = 3600.0;

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
pub const SHOT_ROBOT_RADIUS: f64 = 90.0;
/// Effective radius (mm) of the opponent *keeper* shadow. Inflated well beyond a
/// field robot: the keeper actively dives/slides to cover during the ball's flight,
/// so it blocks far more of the mouth than its static footprint suggests.
pub const SHOT_KEEPER_RADIUS: f64 = 150.0;
// Keeper-charge-aware shot model. The inflated keeper shadow above assumes the
// keeper sits on its line and can dive/slide across the mouth during the ball's
// flight. That stops being true once the keeper has *committed forward* off its
// line (charging a loose ball, smothering a carrier): a keeper several hundred mm
// up-field physically cannot recover to a corner, so the static 280mm shadow
// over-states the cover and makes us *pass up* a shot at a net the keeper has
// abandoned. When the inferred opponent keeper is advanced off its goal line we
// shrink its effective shadow toward a plain field-robot radius, converting a
// high-quality chance we'd otherwise decline into a shot. This is net-asymmetric
// (only the team in possession facing the over-committed keeper benefits — a
// frozen mirror of ourselves can't copy it without also charging its keeper), it
// moves none of our own players, and it is self-limiting: the shrink reverts the
// instant the keeper recovers its line.
/// Keeper advancement (mm off its goal line, along x toward the ball) below which
/// the full inflated [`SHOT_KEEPER_RADIUS`] shadow applies (keeper still on its line).
pub const KEEPER_CHARGE_NEAR: f64 = 700.0;
/// Keeper advancement (mm off its goal line) at/beyond which the keeper is treated
/// as fully committed forward — its shadow shrinks all the way to [`SHOT_ROBOT_RADIUS`]
/// (it can no longer dive to cover the corners). Linearly interpolated in between.
pub const KEEPER_CHARGE_FAR: f64 = 1800.0;
/// Weight on aiming *away* from the keeper when scoring open windows. Score is in
/// goal-line mm: a window this many mm farther from the keeper's projected position
/// is worth `+1mm` of open width per `1/bias` mm of distance. Picks the corner the
/// keeper must travel furthest to reach when both corners are open.
pub const SHOT_KEEPER_BIAS: f64 = 0.35;
/// Minimum angular width (radians) of the open window for the carrier to take a
/// direct shot. Below this we fall through to the kick-ahead / pass logic. The
/// angular gate scales with distance for free: ~0.10 rad is a ~130mm gap at 1.3m
/// but needs a ~400mm gap at 4m.
pub const SHOT_MIN_ANGLE: f64 = 0.03;
// ── Restart kicker aiming (kickoff / free kick) ──────────────────────────────
// A restart kicker (our kickoff / free kick) always RELEASES with a strike-
// through (a reflex `Strike`, never a hold-and-aim — that would double-touch).
// It shoots directly only when the goal is *actually* clear, held to the same
// standard as an open-play shot ([`SHOT_MIN_ANGLE`], shared via `clear_shot`).
// The old Div-B "benign sliver" gamble — shoot at a ~70mm window from the kickoff
// spot and hope for a wall deflection — is retired: with no clear shot a kickoff
// instead deflects off a distant side opponent (below) and a free kick makes a
// clean forward advance, both strictly better than a hopeful sliver that usually
// just concedes possession.
/// Maximum distance to the opponent goal for the restart shot — a coarse reach
/// gate so we don't strike at goal from our own half (the ball would arrive dead).
/// Covers the kickoff spot (4500mm) with margin.
pub const KICKER_SHOT_RANGE: f64 = 5600.0;

// ── Never fire into a close opponent ─────────────────────────────────────────
// A veto on any ball release (restart shot / deflection / advance, open-play
// hoof) whose ray drives the ball straight INTO an opponent at point-blank
// range. Tuned VERY aggressively: it fires only on a *direct shadow* of the
// kick ray — collision-imminent, ball radius + robot radius, nothing wider.
// A wider bubble seals every forward lane whenever a marker stands near (the
// t=1125 free-kick failure), forcing the very hoof-into-the-marker it was meant
// to prevent. Lane *preference* is `clean_advance`'s openness scoring, not this.
/// An opponent within this range of the ball, ahead on the aim ray, vetoes the aim.
pub const CLOSE_OPP_VETO_DIST: f64 = 300.0;
/// Perpendicular half-width (mm) of the veto corridor — a direct shadow only
/// (ball 21 + robot 90, essentially touching the ray).
pub const CLOSE_OPP_VETO_CORRIDOR: f64 = 115.0;

// ── Clean advance (safe-release fallback) ────────────────────────────────────
// The fallback when the naive release aim fails `release_ok` (a direct-shadow
// opponent, or the ray would leave the field almost immediately): sweep a fan of
// forward-to-lateral headings (never backward toward our own goal), reject any
// that fail the same release gate, and take the one down the most open lane.
/// Half-angle (rad) of the heading fan, measured off straight up-field (+x).
/// `FRAC_PI_2` = ±90°, so the extremes are pure-lateral (toward a touchline, at
/// worst a throw-in) and never point back toward our own goal.
pub const ADVANCE_FAN_HALF: f64 = std::f64::consts::FRAC_PI_2;
/// Number of headings swept across the fan.
pub const ADVANCE_FAN_STEPS: usize = 24;
/// Aim-point distance (mm) down the chosen heading (heading is what the strike
/// uses; the exact landing point is not modelled).
pub const ADVANCE_REACH: f64 = 2500.0;
/// A release whose ray leaves the field sooner than this (mm) is rejected — it
/// would put the ball out almost immediately. Applied to BOTH the fan candidates
/// and (new) every naive release via `safe_release`, so no kick concedes the
/// ball over a line; rays exiting through the opponent goal mouth are exempt
/// (that exit is a shot).
pub const ADVANCE_MIN_TRAVEL: f64 = 1500.0;
/// Corridor (mm) for the lane-openness score of each candidate heading.
pub const ADVANCE_CORRIDOR: f64 = 500.0;
/// Forward-progress tiebreak weight: added as `w · dir.x` (dir.x ∈ [0,1]) to the
/// openness ∈ [0,1]. Kept well below 1 so a genuinely cleaner lane always wins —
/// "prioritise clean lanes heavily", forward only breaks near-ties.
pub const ADVANCE_FWD_W: f64 = 0.35;

// ── Harass (block a stable opponent holder) ──────────────────────────────────
/// Centre-to-centre standoff (mm) of the harasser's block point, placed on the
/// holder's facing axis directly in front of its dribbler. ~2 robot radii + a
/// little slack: pressed right up against the holder's mouth, blocking its
/// advancement without ORCA fighting the contact (the driver disables robot
/// avoidance for the harass drive).
pub const HARASS_STANDOFF: f64 = 250.0;

// ── Kickoff deflection ───────────────────────────────────────────────────────
// With no clear shot from the kickoff spot (the mouth is never clear from centre),
// aim at a distant opponent off to the side: the ball deflects off it and most
// likely runs out of play, and an opponent last-touch gives us the restart deep in
// their half. No deflection physics is modelled — we simply aim at a suitable far,
// wide opponent, and the close-opponent veto keeps the lane to it clean.
/// A deflection target must be at least this far downfield (team-relative x) —
/// keeps the deflection in their half, never bounced back toward our goal.
pub const KICKOFF_DEFLECT_MIN_X: f64 = 500.0;
/// A deflection target must be at least this far from the ball ("distant").
pub const KICKOFF_DEFLECT_MIN_DIST: f64 = 2000.0;
/// A deflection target must be at least this far off centre ("to the side"), so
/// the deflection runs toward a touchline rather than back into central play.
pub const KICKOFF_DEFLECT_MIN_LATERAL: f64 = 800.0;
/// A forward-pass receiver must be at least this far forward of the ball.
pub const SUPPORTER_FWD_MARGIN: f64 = 400.0;
/// Corridor width for the ball→supporter lane-openness check.
pub const KICK_LANE_CORRIDOR: f64 = 400.0;
/// Minimum lane openness (0..1) for a supporter to be a viable pass target.
pub const SUPPORTER_MIN_OPENNESS: f64 = 0.5;
/// Lead the pass this far past the supporter toward goal (into space to run onto).
pub const SUPPORTER_LEAD: f64 = 350.0;
/// Ball rel-x beyond which we're in the attacking final third. Here the strict
/// "supporter must be forward" gate no longer fits (there is little field left
/// ahead), so we also accept a wide, roughly-level supporter as a cross/cutback
/// outlet — see the planner's `forward_pass`.
pub const FINAL_THIRD_X: f64 = 2000.0;
/// Minimum lateral (|Δy|) separation between ball and supporter for the
/// supporter to count as a final-third cross/cutback outlet.
pub const CROSS_MIN_LATERAL: f64 = 800.0;
/// How far *behind* the ball (along x) a wide final-third outlet may still sit
/// and remain pass-eligible (a cutback). Strictly-forward outlets are unaffected.
pub const CROSS_BACK_MARGIN: f64 = 600.0;

// ── Corner one-timer (the 1-2 finish) ────────────────────────────────────────
// A ball deep and wide of the opponent box — no shot window from that acute
// angle — is crossed to a teammate posted at the box edge, which redirects it
// into the goal with the pre-armed reflex kick (`ctx.pass(..).forward_to(..)`,
// zero handling latency). These gates are deliberately independent of every
// generic pass-quality estimate ([`PASS_SUCCESS_BASE`] does not scale them):
// the play is judged on its own deflection geometry.
/// Master switch for the whole play. `false` removes it from the destination
/// ladder — only goal shots, forward passes, and advance strikes remain.
pub const CORNER_OT_ENABLED: bool = false;
/// How far upfield of the opponent box front the trigger zone extends: the
/// carrier must be at `x > box_front − CORNER_OT_ZONE_DEPTH`. Deep enough to
/// cover the whole corner pocket where the strike-finish fallback fires from
/// near-zero angle, shallow enough that a wide midfielder doesn't spray
/// crosses from halfway.
pub const CORNER_OT_ZONE_DEPTH: f64 = 500.0;
/// How far outside the box's lateral edge (|y| > pen_half_width + this) the
/// carrier must be. Inside this band a direct strike still has workable
/// geometry; the one-timer is for genuinely acute angles.
pub const CORNER_OT_MIN_WIDE: f64 = 200.0;
/// Commit bar on the deflection angle (degrees; 180° = head-on = ideal, same
/// convention as the executor's `ONE_TIMER_MIN_DEFLECTION`). Must stay above
/// the coordinator's 120° pre-commit gate with slack, so staging drift and a
/// moving ball can't turn a committed pass into an `InfeasibleGeometry` abort.
pub const CORNER_OT_MIN_DEFLECTION: f64 = 135.0;
/// Minimum openness (0..1, [`KICK_LANE_CORRIDOR`]-width) of the cross lane from
/// the ball to the pocket. Below this the cross is a giveaway into a blocker —
/// hard gate, NOT scaled by [`PASS_SUCCESS_BASE`].
pub const CORNER_OT_MIN_LANE: f64 = 0.5;
/// Cap on the nominated receiver's momentum-aware travel time to the pocket
/// (s). Cross-field pulls are intended (the play is worth a body), but a robot
/// that can't arrive inside the coordinator's 8s stage budget with margin
/// would just burn the possession staging.
pub const CORNER_OT_MAX_TRAVEL: f64 = 4.0;
/// Standoff of the pocket candidates outside the opponent defense area edge.
/// Attackers may not touch the ball inside the area, so the receive point
/// presses to the edge but keeps a robot radius + legal/intercept margin out.
pub const CORNER_OT_POCKET_MARGIN: f64 = 300.0;
/// Room the passer needs BEHIND the ball on the cross axis (mm): the pass
/// coordinator stages the passer behind the free ball and drives through it,
/// so the robot-centre point this far behind the ball (away from the pocket)
/// must be inside the field. A deep-corner ball fails this for pockets whose
/// axis points out of the corner — those pockets are skipped, not the whole
/// play (a different pocket bends the axis back into the field). Without the
/// gate the release wedges against the boundary and burns the whole stage
/// budget (observed in the corner_one_timer scenario: 7 s Release timeout,
/// 0 mm/s kick). 250 matches the release backoff measured on clean releases
/// (~235 mm behind the ball at kick).
pub const CORNER_OT_STAGE_ROOM: f64 = 250.0;
/// Weight of lane openness from the pocket to the goal-mouth aim when ranking
/// feasible aims, in degrees of equivalent deflection margin. Feasibility is
/// the primary pull; a fully open shooting lane is worth this many extra
/// degrees of margin, no more.
pub const CORNER_OT_AIM_OPENNESS_WEIGHT: f64 = 15.0;
/// Floor of the pocket-clearance factor (0..1). An opponent parked on the
/// pocket discounts it (adjacent samples win — the "slide along the box edge"
/// behavior) but never vetoes it outright: pressing into a crowded box is part
/// of the play.
pub const CORNER_OT_CLEARANCE_FLOOR: f64 = 0.35;

// ── Penalty kick (rules §5.3.5) ──────────────────────────────────────────────
// The taker is driven directly (not through the planner): the shot distance
// (goal_line_to_penalty_mark, 6m Div B) exceeds [`KICKER_SHOT_RANGE`], and the
// planner's fallbacks (kick-ahead / recycle / pass) are all illegal with every
// teammate parked ≥1m behind the ball. The executor enforces all positioning
// rules; concerto only stages the taker and takes the single shot.
/// How far behind the ball (away from the opponent goal) the taker stands during
/// `PreparePenalty` — it may approach but must NOT touch the ball before normal
/// start. More generous than the generic 300mm prep standoff as a margin against
/// control jitter; closing 500mm is trivially inside the 10s kick budget.
pub const PENALTY_PREP_STANDOFF: f64 = 500.0;
/// Aim hysteresis: the latched aim point only moves when the freshly computed
/// best target shifts more than this along the goal line. Prevents the Strike
/// aiming loop from re-orbiting on per-frame keeper jitter.
pub const PENALTY_AIM_STICK: f64 = 200.0;
/// Fallback aim (no shot window resolvable): this far inside the post, on the
/// side farther from the keeper.
pub const PENALTY_AIM_POST_MARGIN: f64 = 150.0;

// ── Advancement hoof (ray-based, travel-calibration-free, goal-mouth cone) ───
// The hoof scores *directions*, not landing points, so no roll-distance model
// enters (the old resting-point scorer broke whenever kick speed or carpet
// friction differed from the model). Candidates are restricted to the cone from
// the ball to the opponent GOAL MOUTH (posts inset by [`HOOF_MOUTH_INSET`]) —
// not the full back line. Rationale: the Div-B aimless-kick rule awards the
// opponent a free kick AT OUR KICK POSITION whenever a ball kicked from our half
// crosses halfway and then their goal line outside the goal untouched, so a wide
// back-line exit is the *penalized* outcome, not a benign one. A mouth-bound ray
// has no aimless case: untouched it's a goal; blocked it's a keeper/defender
// touch (rebound, or a corner for us off the keeper). 2026-07-02 match data:
// 11 aimless resets in ~55min from wide-cone hoofs, while every kick aimed
// inside the mouth either scored or drew a keeper touch.
/// Bonus for a ray whose exit falls inside an *open* goal-mouth window
/// (keeper-shadow aware) — prefer the ray that can actually score.
pub const HOOF_GOAL_W: f64 = 3000.0;
/// Bonus for a ray passing near one of our players in the opponent half (the
/// permanent outlet / supporters), decaying to zero over [`HOOF_MATE_RADIUS`] of
/// perpendicular distance. With the mouth cone this is a within-mouth tiebreak
/// (a mate near the goal-bound ray helps rebound recovery), no longer a wide
/// target-man ball.
pub const HOOF_MATE_W: f64 = 2000.0;
pub const HOOF_MATE_RADIUS: f64 = 1200.0;
/// Maximum penalty for opponents sitting on the ray (they intercept en route).
/// Kept mild: every ray is goal-bound now, so an en-route touch is rule-safe
/// (defuses aimless) and a deflection near their goal is a cheap gamble.
pub const HOOF_OPEN_PENALTY: f64 = 1500.0;
/// Corridor width (mm) for the ray-openness term.
pub const HOOF_RAY_CORRIDOR: f64 = 600.0;
/// Weight on the centre-bias tiebreak (mm of |exit y| off the touchline): all
/// rays exit at the same depth, so ties break toward the dangerous central zone.
pub const HOOF_CENTER_W: f64 = 0.3;
/// Inset (mm) from the goal posts bounding the candidate cone — execution margin
/// so a yaw error doesn't turn a near-post aim into a wide (aimless) miss. Real
/// long kicks land ~100–150mm off the intended line at full field range; today's
/// near-post aimless misses were exactly this band (534–1100mm past the post).
pub const HOOF_MOUTH_INSET: f64 = 150.0;

// ── Driver ──────────────────────────────────────────────────────────────────
/// Ball displacement from the engagement point that aborts any waypoint. This is
/// the driver's only remaining ball-handling guard — a premise-void replan
/// trigger, not an execution timeout (those now live inside the skills).
pub const BALL_MOVED_DIST: f64 = 2000.0;

// ── Formation ─────────────────────────────────────────────────────────────
/// Threat ramp: distance to our goal at which threat is maximal / negligible.
pub const THREAT_GOAL_NEAR: f64 = 1500.0;
pub const THREAT_GOAL_FAR: f64 = 6000.0;
/// Number of shadow (goal-coverage) roles the wall always emits; the *staffed*
/// count emerges from threat-scaled importance.
pub const SHADOW_MAX: usize = 2; // TODO: CHECk
/// Fix A: per-arc-step importance falloff for the always-emitted shadow wall.
/// The central shadow (on the ball→goal ray) keeps full importance; each step out
/// toward a wing multiplies by this, so at low threat only the centre is staffed
/// and the wings light up continuously as threat rises. No count step → no blink.
/// Lever 1: raised 0.6→0.8 so the wing wall fades less toward the flanks — at
/// moderate threat the wings (which cover the open *post angles* a counter shoots
/// into) now light up and outbid mark/capture for the nearest bodies, instead of
/// only the central anchor being staffed. Goal openness is angular, so lateral
/// wall coverage (not a deeper central body) is what shrinks the open angle.
pub const SHADOW_STAGGER: f64 = 0.8;
/// How far in front of our goal the shadow line sits.
pub const SHADOW_STANDOFF: f64 = 1000.0;
/// Lateral spacing between adjacent shadows in the goal-coverage wall. Kept just
/// under a robot width (~180mm) so neighbours overlap and leave no central gap —
/// the wall is built centre-out on the direct ball→goal line, so an even count
/// straddles the centre rather than splitting around it.
pub const SHADOW_SPACING: f64 = 170.0;
/// Lateral spacing for a **compacted 2-body wall** (anchor + one wing brought onto
/// the same band, side by side). Separate knob from [`SHADOW_SPACING`] so the 2-wall
/// pair can be tuned independently; defaults to the same tight, no-gap wall spacing.
pub const COMPACT_SPACING: f64 = SHADOW_SPACING;
/// How far outside the defense-area box (mm) the shadow **wing** curve sits. The
/// wall now slides along a rounded-rectangle offset of the box rather than a radial
/// line: this is the wings' offset, and [`ANCHOR_BOX_MARGIN`] the (tighter) anchor's.
/// Together they replace the goal-centre-radial `SHADOW_STANDOFF`. Sits ~one
/// robot-width-plus outside the anchor so a 2-body wall never stacks. Tuning surface
/// — changes wall depth/coverage; validate in sim.
pub const SHADOW_BOX_OFFSET: f64 = 100.0;
/// Half-width (mm) of the bounded band around each shadow offset curve. The
/// no-overshoot envelope keeps a waller within `offset ± this` of the box. Must
/// exceed the executor's BOUNDS_MARGIN (30) or the target sits on an unreachable
/// band edge and the robot chatters.
pub const WALL_BAND_SLACK: f64 = 80.0;
/// Max along-curve translation speed (mm/s of arc-length) of the wall as a rigid
/// unit. Kept below the GoToBounded speed cap so no member is asked to translate
/// faster than it can track → relative spacing stays pinned.
pub const WALL_SLEW_RATE: f64 = 1800.0;
/// Along-curve tracking-error tolerance (mm). When the furthest-behind wall member
/// lags its slot target by this much, group translation freezes (paced to
/// [`PACE_FLOOR`]) until it catches up, so the wall never shears apart mid-swing.
pub const WALL_LAG_TOL: f64 = 260.0;
/// Floor on the rigid-wall pace factor (0..1) so a single persistently-slow /
/// handicapped member can never fully stall the whole wall's tracking.
pub const PACE_FLOOR: f64 = 0.1;
/// A waller further than `max_offset + this` (mm) from its band drives in with the
/// planner+ORCA (`go_to`) instead of the bounded direct drive (which bypasses both
/// and would cut straight through obstacles); it hands off to bounded once inside.
pub const WALL_HANDOFF_SLACK: f64 = 250.0;
/// The central shadow (on the ball→goal ray) is promoted to an "anchor": a sticky
/// last-line field defender, distinct from the keeper at the mouth. It is pulled
/// in from the wall standoff to the penalty-area front edge plus this margin (mm)
/// so it hugs the box — layering keeper(~400) → anchor(box edge) → wings(1500).
/// Only applied in open play; on set-piece defense the centre stays in the
/// contiguous wall (leg-2 corner-free-kick fix).
pub const ANCHOR_BOX_MARGIN: f64 = 150.0;
/// Importance floor (on the threat-scaling factor, normally 0.5..1.0) for the
/// anchor. Its importance never collapses with low ball-threat the way the wing
/// wall does, so it stays staffed above the striker/support roles even while we
/// commit forward — guaranteeing one body at the box. `IMP_SHADOW_BASE * this`
/// (8.0 × 0.8 = 6.4) sits above IMP_STRIKER/IMP_SUPPORT_ATTACK/IMP_MARK_BASE but
/// below CAPTURE/IMP_SUPPRESS, so a genuine ball-winner or a plan robot already
/// contesting at the box can still pull it.
pub const ANCHOR_THREAT_FLOOR: f64 = 0.8;
/// Marking standoff in front of the marked opponent (toward our goal).
pub const MARK_STANDOFF: f64 = 400.0;
/// Markers never cross to the opponent half: their target x is capped here
/// (team-relative, so 0 = midfield). Keeps defenders home instead of chasing
/// opponents up-field.
pub const MARK_MAX_X: f64 = 0.0;
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
/// Fix A: the `attacking` boolean is replaced by a continuous attack fraction
/// `af ∈ [0,1]`. `af` ramps up as the ball advances past `SUPPORT_ATTACK_BALL_X`
/// toward `..._FULL`, and is gated down as our-goal threat rises from
/// `SUPPORT_ATTACK_MIN_THREAT` to `SUPPORT_ATTACK_MAX_THREAT`. Support importance
/// lerps IMP_SUPPORT→IMP_SUPPORT_ATTACK with `af`, and the striker importance
/// scales with `af` (≈0 when not attacking) — so the support/striker block fades
/// in/out smoothly across midfield instead of snapping at a boolean threshold.
pub const SUPPORT_ATTACK_BALL_X_FULL: f64 = 2500.0;
pub const SUPPORT_ATTACK_MIN_THREAT: f64 = 0.15;
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
/// Minimum LATERAL (|Δy|) spacing between two supporters' targets, applied on top
/// of the radial separation: supporters must occupy distinct lateral bands, so
/// multiple outlets always spread across the field width instead of stacking on
/// one flank (two bodies on one wing are one pass option, not two).
pub const SUPPORT_MIN_LATERAL_SEP: f64 = 1200.0;
/// Weight of the shot-quality term in supporter placement. The candidate score is
/// `lane_openness + receiver_clearance + SUPPORT_SHOT_W · shot_quality`, where
/// shot_quality is the planner's own keeper-charge-aware shot model
/// (`geometry::best_shot_from`) normalized by [`SUPPORT_SHOT_ANGLE_NORM`]. At 1.0
/// a fully open shooting window is worth as much as a fully open pass lane — the
/// supporter stands where a received ball is a real chance, not merely reachable.
pub const SUPPORT_SHOT_W: f64 = 1.0;
/// Shot-window angular width (rad) that counts as a "full" shooting window in
/// supporter scoring; wider windows saturate at 1. ~0.25 rad is a ~500mm gap at
/// 2m — a comfortably makeable shot.
pub const SUPPORT_SHOT_ANGLE_NORM: f64 = 0.25;
/// Forward (+x) preference in supporter scoring, per mm of candidate x — pushes
/// the outlets into the opponent half. At 2e-4 a 3000mm-deeper spot is worth
/// +0.6 in the 0..3 score space: it decides among comparably open spots, but a
/// genuinely open + shootable spot (a full term's worth) still beats raw depth.
pub const SUPPORT_FWD_X_W: f64 = 2e-4;
/// Incumbency bonus that stabilises a supporter's target against frame-to-frame
/// thrashing. `best_support_pos` is a discrete argmax over a grid of candidate
/// outlets; with opponent positions jittering under Kalman noise, the top two
/// candidates are often near-tied and the winner flips every tick — yanking the
/// target across a grid cell (or the whole field, on a flank flip) and leaving
/// the supporter dithering in place instead of arriving. The candidate nearest
/// last tick's target for this slot gets up to `+SUPPORT_STICKINESS` added to its
/// score, so a challenger must be *decisively* better (by this margin in the
/// 0..3 score space) to displace the incumbent — hysteresis in argmax form.
/// Deliberately VERY high: a challenger must beat the incumbent by a full
/// term's worth (a whole open lane / clear spot / shot window) before the
/// target moves. A stable, predictable outlet beats chasing the marginally
/// best spot (oscillation is the failure mode, not slightly-suboptimal spots).
pub const SUPPORT_STICKINESS: f64 = 1.0;
/// Distance over which the incumbency bonus decays to zero (mm). Around the
/// grid spacing (~1000mm in x, wider across flanks) so only the candidate
/// essentially *at* last tick's target keeps the full bonus and its neighbours get
/// little — the bonus favours holding the current outlet, not a whole region.
/// Also absorbs the small per-tick translation of the ball-relative grid so the
/// incumbent stays sticky through jitter while still advancing with the ball.
pub const SUPPORT_STICKY_RADIUS: f64 = 1000.0;
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
/// Incumbency bonus for the box-runner's finishing pocket, the analogue of
/// `SUPPORT_STICKINESS` for `best_finishing_pocket`. That pocket is also a discrete
/// argmax (6 candidates near the box front), so it flips between near-tied pockets
/// under opponent jitter. Unlike the supporters' additive `open+clearance` score
/// (0..2), the pocket is scored `shot_angle × receiver_clearance` — a much smaller,
/// input-dependent scale — so the bonus is applied *multiplicatively* here: the
/// candidate nearest last tick's pocket has its score scaled by up to
/// `1 + BOX_RUNNER_STICKINESS`, i.e. a challenger must be this fraction better to
/// take the pocket. Scale-free, so it holds whatever the absolute angles are.
pub const BOX_RUNNER_STICKINESS: f64 = 0.15;
/// Decay distance (mm) for the box-runner incumbency bonus. Tighter than the wide
/// supporters' radius because the finishing pockets are only ~200–500mm apart — this
/// keeps the bonus discriminating between adjacent pockets (holds *this* pocket, not
/// the whole box front) rather than smearing across all of them.
pub const BOX_RUNNER_STICKY_RADIUS: f64 = 250.0;
/// Importance of the box-runner role while attacking. At/above `IMP_SUPPORT_ATTACK`
/// so the central outlet is reliably staffed when we commit forward.
pub const IMP_STRIKER: f64 = 5.5;
/// Rebound poacher: when a shot is in flight at the opponent goal (the ball is
/// moving goalward at or above this speed in the attacking third), the box-runner
/// stops holding its front-of-box cutback pocket and *crashes the goal mouth* to
/// pounce on a keeper save / parry / rebound — "follow your shot in". A save in
/// this low-event sim is the single most common way a finish dies without a second
/// chance; putting a body on the loose ball turns it into another shot. Position
/// only (same Support slot), so no role churn, and it never touches our keeper.
pub const REBOUND_CRASH_BALL_SPEED: f64 = 1500.0;
/// How far out from the opponent goal line (mm) the poacher waits for the rebound —
/// inside the goal area, where parries and rebounds land, but not on the line.
pub const REBOUND_CRASH_DEPTH: f64 = 700.0;
/// Lateral offset (mm) of the poacher from goal centre while crashing, toward the
/// side away from the incoming ball (the side the keeper, having shifted to cover
/// the shot, tends to leave open). Small — it must still cover a central rebound.
pub const REBOUND_CRASH_Y: f64 = 220.0;

/// Importance ladder (points; converted to seconds via SEC_PER_IMPORTANCE).
/// Lever 1: raised 8.0→10.0 to give the threat-scaled wing wall enough headroom
/// to outbid Mark for the nearest bodies when the goal is threatened, so the
/// posts get covered on a counter. Also lifts the anchor floor (×0.8 → 8.0); both
/// stay below Capture (13), so a genuine ball-winner can still pull them.
pub const IMP_SHADOW_BASE: f64 = 12.0;
/// Discounted 6.0→4.0 (current tuning: prefer shadows over man-marking). At 6 a
/// high-threat open-lane mark could tie a mid-threat wing (10 × 0.75 × 0.8) and
/// outbid Outlet/SupportAttack (5) for surplus bodies. At 4 the wall always wins,
/// and marks sit below the forward-presence tier — staffed only by bodies with
/// nothing better, on genuinely dangerous unmarked opponents (threat·openness
/// > 0.75 to beat plain Support at 3).
pub const IMP_MARK_BASE: f64 = 4.0;
pub const IMP_SUPPORT: f64 = 3.0;
/// Support importance while attacking (ball in the opponent half, own goal safe).
/// Above the weak-mark / spread range so surplus bodies win the assignment as
/// supporters and push up instead of loitering at midfield.
pub const IMP_SUPPORT_ATTACK: f64 = 5.0;
pub const IMP_SPREAD: f64 = 0.0;
// ── Counter outlet (anti-press swing target) ─────────────────────────────────
// When under genuine own-goal threat (being pressed) the attack fraction `af`
// collapses and every supporter retreats, leaving no advanced target to clear or
// counter to — the "limited presence in the opponent half when pressed, so we
// can't swing back" failure. To fix it we hold ONE body high on the far flank
// whenever own-goal threat exceeds OUTLET_THREAT_LO, ramping to full at
// OUTLET_THREAT_HI. Gated on THREAT (not ball-x) so it fires only while defending
// and never during our own low-threat buildup, leaving the carrier's short
// recycle outlets intact (extending the ball-x attack ramp into our half instead
// tanked buildup possession). Suppressed on set-piece defense (full wall, leg-2).
pub const OUTLET_THREAT_LO: f64 = 0.30;
pub const OUTLET_THREAT_HI: f64 = 0.60;
/// Floor on the outlet's threat gate (Div-B): `max(smoothstep(LO, HI, threat),
/// floor)`, so at 1.0 the outlet is a PERMANENTLY deployed forward — the target
/// man every hoof is biased toward and the standing pressure on their buildup.
/// 0.0 restores the pure threat-gated behaviour. Note the floor must be applied
/// via `max`, not by lowering `OUTLET_THREAT_LO` — a negative LO only shifts the
/// smoothstep (LO = -1 yields gate ≈ 0.68 at zero threat, not 1). Ladder check:
/// at full importance (5.0) the outlet only outbids the *fading* outer wall wings
/// at low threat, and is the last defensive-tier role staffed at high threat.
pub const OUTLET_FLOOR: f64 = 1.0;
/// Field-robot count below which the floor is NOT applied (the threat-gated
/// behaviour remains): a short-handed team can't pin 1 of 3 defenders forward.
pub const OUTLET_MIN_BOTS: usize = 4;
/// Contest gate on the outlet floor: possession latched with an opponent still
/// on the ball in our own half is not secure, so the permanent-forward floor is
/// faded out and no body is pinned at [`OUTLET_X`] mid-scrum. The fade is the
/// product of two smoothsteps: fully contested when the nearest opponent is
/// within `..._OPP_NEAR` mm of the ball, fully clear beyond `..._OPP_FAR`.
pub const OUTLET_CONTEST_OPP_NEAR: f64 = 500.0;
pub const OUTLET_CONTEST_OPP_FAR: f64 = 1000.0;
/// Ball-x fade for the same gate: the contest discount applies fully with the
/// ball in our half (x ≤ 0) and washes out by this far into the opponent half.
pub const OUTLET_CONTEST_BALL_X_FADE: f64 = 1000.0;
/// Field-absolute x of the counter outlet (mm, team-relative; +x = opponent
/// half). Deep enough into the opponent half to press their buildup and chase
/// hoofs landing in their third (was 800 — a clearance target, not a presser —
/// when the outlet existed only under threat).
pub const OUTLET_X: f64 = 2200.0;
/// Lateral placement of the outlet as a fraction of half-width, on the flank away
/// from the ball (where a switch/clearance naturally goes).
pub const OUTLET_Y_FRAC: f64 = 0.55;
/// Importance of the counter outlet at full threat. Below Balance (6) and the
/// Shadow wall (10) so it never strips the goal wall — it claims at most the
/// least-critical surplus body — but above plain Support (3)/Spread and the
/// discounted Mark (≤4, deliberate: forward presence over man-marking) so that
/// body holds high instead of collapsing all the way home.
pub const IMP_OUTLET: f64 = 5.0;
// ── Opponent-kickoff guard ───────────────────────────────────────────────────
/// A single body that shadows the opponent kicker's aim just outside the centre
/// circle on our side, ready for an instant catch of the kickoff. Active only
/// while the opponent prepares/takes a kickoff.
///
/// Importance: below the (kickoff) shadow wall so the wall + keeper always staff
/// first — at a centred ball the wings sit at `IMP_SHADOW_BASE·(0.5+0.5·threat)·
/// SHADOW_STAGGER ≈ 12·0.66·0.8 ≈ 6.4` — but above every other role (Outlet 5.0,
/// Striker 5.5, Support/Mark ≤4). So the guard claims the next body after the
/// wall and yields a wing only when short-handed.
pub const IMP_KICKOFF_GUARD: f64 = 6.0;
/// How far outside the centre circle (mm) the guard rides — just clear of it, so
/// it can step onto a rolling kickoff. (The lib.rs kickoff-staging push-out, which
/// forces other robots to `radius + KICKOFF_CIRCLE_MARGIN`, is bypassed for this
/// role; see the `kickoff_guard` exemption there.)
pub const KICKOFF_GUARD_CIRCLE_MARGIN: f64 = 150.0;
/// Margin (mm) past the halfway line onto our side: the guard's centre never
/// crosses to `x > -this`, keeping the whole body on our half.
pub const KICKOFF_GUARD_HALF_MARGIN: f64 = 150.0;
// ── Wall clear ──────────────────────────────────────────────────────────────
// A ball loose in front of our wall (the "apron" — near our goal, outside the
// keeper's box) is the wall's to deal with, and the wall alone: while the ball
// is in the apron the open capture role is suppressed *unconditionally* so no
// outside capturer stern-chases it into the corridor (crowding the wall,
// splitting it, or stalling the game). A *fast* ball is blocked by the standing
// wall — nobody steps out. A *slow / settled* ball is cleared: the wall robot
// nearest it (by redirect_time) steps out and hoofs it straight forward
// (`Strike { acquire_first: false }` — never holds it), but only when we win the
// race to the ball against the closest opponent (else we hold and defend the
// wall — losing the race would open a gap AND hand them a dead ball). While it
// clears, its shadow slot is held by its own body (see SHADOW_HOLD_*) so no
// backfill body is pulled across the field.
/// Master switch for the wall clear. `false` restores pure open-capture
/// behaviour (every loose ball, including in the apron, goes to the capturer).
pub const WALL_CLEAR_ENABLED: bool = true;
/// Max ball speed (mm/s) to elect a clearer — at/below this the ball is settled
/// or gently rolling and worth stepping out for; a faster ball is a delivery/shot
/// and is blocked by the standing wall, never chased out of formation.
pub const WALL_CLEAR_MAX_SPEED: f64 = 500.0;
/// Race slack (seconds of redirect_time): commit to the clear iff the nearest
/// waller's time-to-ball ≤ closest opponent's + this. 0 = go on a coin-flip
/// (harass); negative = only when clearly ahead; positive = more aggressive.
/// Tune on the real field.
pub const WALL_CLEAR_MARGIN: f64 = 0.0;
/// Hysteresis (mirror of CAPTURE_COMMIT): once committed, abort only if the
/// opponent is winning the race by more than this, so a marginally-closer
/// opponent mid-approach doesn't flicker the wall gap open.
pub const WALL_CLEAR_COMMIT: f64 = 0.4;
/// Apron depth (mm): the ball is in the wall's zone when it is within this of the
/// own goal line (`ball.x < -field_length/2 + this`). Covers the wall standoff
/// (SHADOW_STANDOFF) plus a reach margin.
pub const WALL_CLEAR_APRON_DEPTH: f64 = 2600.0;
/// Apron lateral half-width (mm): balls wider than this (near our backline
/// corners) are out of the wall's zone. ≈ penalty_area half-width + a margin.
pub const WALL_CLEAR_APRON_HALF_WIDTH: f64 = 1600.0;
/// Max distance (mm) from the clearer to the ball for an armed clear to stay
/// valid — a grown reach so approach jitter doesn't drop the episode.
pub const WALL_CLEAR_REACH: f64 = 900.0;
/// Slot-hold: a reserved plan robot (e.g. the wall striker working at its own
/// post, or a carrier standing in the wall) fades the importance of a Shadow
/// slot it is physically covering — `imp *= smoothstep(NEAR, FAR, dist)` — so
/// the matcher never pulls a backfill body across the field for a slot that has
/// a body in it, and the slot fades back in continuously if the body leaves.
pub const SHADOW_HOLD_NEAR: f64 = 300.0;
pub const SHADOW_HOLD_FAR: f64 = 900.0;
/// Coverage accounting: radius around a plan robot's ball contest within which
/// formation roles are de-prioritised (soft suppression, avoids clustering).
/// Shadow roles are exempt — the wall must never step aside because a scrum
/// arrived at it (its post is most valuable exactly then); Shadow staffing is
/// instead moderated by the body-proximity slot-hold above.
pub const SUPPRESS_RADIUS: f64 = 800.0;
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
/// Fix D: commitment discount (seconds of redirect time) the incumbent pursuer
/// gets on the Capture slot, so a challenger must be this much faster to the ball
/// to take over the chase. Bounded well below the capture importance so it only
/// settles near-ties (who is closest as the lead point jitters), never overrides a
/// decisively better challenger. Keeps a chase committed to one robot.
pub const CAPTURE_COMMIT: f64 = 0.4;
/// Wall-membership stickiness (seconds of redirect time). A robot currently serving
/// a Shadow slot gets this discount on *any* Shadow slot, so a challenger (typically
/// a supporter) must be decisively faster to pull it off the wall. This damps the
/// shadow↔support ping-pong that makes the wall shuffle — the wall-membership analogue
/// of [`CAPTURE_COMMIT`]. Bounded below the wall importance so a genuine threat drop
/// (wing importance fading) still releases the body; it only settles near-ties.
pub const WALL_COMMIT: f64 = 0.7;
/// Assignment recalculation cadence.
pub const RECALC_COOLDOWN: f64 = 0.18;
pub const RECALC_BG_PERIOD: f64 = 0.4;
/// A plan-context area is "moved" (a trigger) when it shifts more than this.
pub const PLAN_CTX_MOVE_EPS: f64 = 300.0;

// ── Goalkeeper ──────────────────────────────────────────────────────────────
/// Radius of the keeper's positioning arc, measured from the goal centre. At the
/// central (square-on) position the keeper sits this far in front of the line.
pub const KEEPER_ARC_RADIUS: f64 = 450.0;
/// Maximum angular excursion of the keeper off straight-out (radians). A pure
/// *sanity* bound now (the lateral mouth clamp is the primary limit): keeps the
/// keeper a little in front of the line rather than drifting onto it at extreme
/// angles. Widened from the old 45° — that clamp parked the keeper near goal-centre
/// and left the near post wide open against oblique (corner) shots. Also bounds the
/// guard zone's angular span.
pub const KEEPER_ARC_MAX_ANGLE: f64 = 90.0 * std::f64::consts::PI / 180.0; // 80°
/// How far inside each post (mm) the keeper centre may travel laterally — the
/// "stay in front of your net" clamp (Option B). With `KEEPER_ARC_RADIUS` < half the
/// goal width this is slack (the arc caps lateral reach first); it becomes the
/// binding limit if the radius is ever grown past the mouth half-width.
pub const KEEPER_MOUTH_MARGIN: f64 = 0.0;
/// Modest bias of the guard target toward the *near* post (the post on the ball's
/// side), scaled by how oblique the ball is. Conceding the far post (a very hard
/// shot from a tight angle) to better deny the near post, which is the makeable
/// shot. 0 = pure cone bisector; 1 = aim straight at the near post.
///
/// Kept small: on a *symmetric* shooter the bias is net-negative (it opens the far
/// lane wider than it closes the near — see `examples/goalie_bench.rs`), and the
/// mouth clamp alone already covers the near post on oblique shots. This small
/// value hedges only toward a realistic shooter that prefers the near post on tight
/// angles, at a bounded coverage cost.
pub const KEEPER_NEARPOST_BIAS: f64 = 0.10;
/// Radial breathing room (mm) added beyond `KEEPER_ARC_RADIUS` for the guard
/// zone's outer edge, so position noise doesn't peg the keeper at the boundary.
/// The aggressive control profile itself (gains, speed/accel caps, ORCA-off)
/// lives in the `GoToBounded` executor skill.
pub const KEEPER_ZONE_RADIUS_SLACK: f64 = 50.0;
/// Angular breathing room (radians) added beyond `KEEPER_ARC_MAX_ANGLE` for the
/// guard zone's ends, so the keeper can actually reach a near-post target without
/// the no-overshoot envelope braking it at the boundary.
pub const KEEPER_ZONE_ANGLE_SLACK: f64 = 0.12;

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

// Cross-shot lateral damping. A ball moving fast *across* the goal mouth (large
// |vy|, not heading goalward — so the shot-line intercept above does NOT fire) is
// a cross/switch, not a shot. Chasing the cone bisector to the ball's current
// lateral position over-commits the keeper to the ball's side; a quick strike back
// across the face then beats it into the open corner (the worked-ball-across-the-
// box concession). When the ball is crossing the face fast in our defensive third,
// pull the keeper's guard angle back toward square-on so it stays compact and set
// for the shot from either side instead of following the cross all the way. This
// keeps the keeper MORE central/home — it never advances or chases off the line —
// so it is strictly safer than the bisector. Damping scales with lateral ball
// speed (a slow ball dribbled across is tracked normally) and is capped well below
// 1 so the keeper still shades to the ball's side, just less far.
/// Lateral ball speed (mm/s) below which no cross damping applies (a slow ball is
/// tracked normally); and the speed at which damping is full.
pub const KEEPER_CROSS_DAMP_SPEED_LO: f64 = 1000.0;
pub const KEEPER_CROSS_DAMP_SPEED_HI: f64 = 3000.0;
/// Maximum fraction the guard angle is pulled back toward square-on for a full-speed
/// cross. Capped below 1 so the keeper still shades toward the ball's side.
pub const KEEPER_CROSS_DAMP_MAX: f64 = 0.55;
/// Only damp when the ball is this deep in our half (rel-x, negative = our side):
/// a cross at midfield is no immediate shot threat. ~ -1500 mm = our defensive third.
pub const KEEPER_CROSS_DAMP_BALL_X: f64 = -1500.0;

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
pub const CLEAR_EXIT_MARGIN: f64 = 20.0;
// The clear is aimed by scoring rays (see `geometry::best_clear_ray`), not a
// fixed target: candidates in the opponent GOAL-MOUTH cone (Div-B aimless-safe,
// same rationale as the hoof), with openness measured against ALL robots — own
// wall included, which is what the old fixed-target clear kicked straight into.
// The wall simultaneously vacates a corridor (see `CLEAR_CORRIDOR_HALF_WIDTH`),
// the keeper holds a pre-strike pose until a ray is actually open, and a panic
// deadline tied to the Keeper-Held-Ball rule guarantees the ball always leaves.
/// Half-width (mm) of the corridor (ball → opponent goal centre) the wall
/// vacates while the keeper clears. Must cover the keeper's whole goal-bound
/// mouth cone at the wall's standoff (~±90mm at 1.7m) plus robot + ball radii
/// and execution margin, so any mouth-bound ray clears the parted wall.
pub const CLEAR_CORRIDOR_HALF_WIDTH: f64 = 450.0;
/// Minimum lane openness ([0,1], see `geometry::lane_openness`) of the chosen
/// ray for the keeper to commit to the strike; below it the keeper holds its
/// pre-strike pose and waits for the wall to part.
pub const CLEAR_FIRE_OPENNESS: f64 = 0.7;
/// Hysteresis on the fire gate: once striking, keep striking unless openness
/// falls this far below [`CLEAR_FIRE_OPENNESS`] (a wall robot transiting the
/// corridor must not flip the keeper back into hold mid-approach).
pub const CLEAR_FIRE_HYSTERESIS: f64 = 0.2;
/// Corridor width (mm) for the clear-ray openness term. Tighter than the hoof's
/// (600): the clear only needs the ball to physically miss bodies, and the
/// corridor the wall vacates is itself only 450mm each side.
pub const CLEAR_RAY_CORRIDOR: f64 = 300.0;
/// Range (mm from the ball) over which clear-ray openness is measured. Only
/// near blockers matter to the fire gate: a robot well downfield on the ray is
/// a rule-safe en-route touch (defuses aimless), not a bounce-back-at-our-goal
/// risk, and must not keep the keeper holding. Sized to cover our half's
/// congestion around the box (wall + pressing opponents).
pub const CLEAR_OPEN_RANGE: f64 = 3000.0;
/// Ball-in-our-box time (s, world clock) after which the keeper stops waiting
/// for a clean ray and blasts the least-bad one. The Keeper-Held-Ball rule
/// (Div B) is a foul-counter foul at 10s of ball-in-defense-area; 6s leaves
/// ample margin for the approach + strike + ball exit. An aimless/throw-in
/// outcome (no foul counter) always beats the held-ball foul.
pub const CLEAR_PANIC_T: f64 = 4.0;
/// Panic-mode bonus per unit of |ray dir y| — prefer sideways shots when
/// nothing is clean: a blocked central ray ricochets back toward our goal,
/// a blocked sideways ray deflects laterally, and a clean sideways exit over
/// the touch line is only a throw-in (no foul counter, never aimless).
pub const CLEAR_PANIC_SIDE_W: f64 = 0.3;
/// Hard "never clear into a nearby opponent" veto for the keeper — the strongest
/// application of the close-opponent rule (a rebound off an opponent at the
/// keeper's feet goes into our OWN goal). Wider than the restart veto
/// ([`CLOSE_OPP_VETO_DIST`] / [`CLOSE_OPP_VETO_CORRIDOR`]): a vetoed ray makes the
/// keeper hold (or, in panic, pick any lane that misses the opponent).
pub const CLEAR_VETO_DIST: f64 = 1500.0;
pub const CLEAR_VETO_CORRIDOR: f64 = 550.0;
/// Bonus for a clear ray exiting through an *open* opponent goal-mouth window
/// (an untouched clear scores). Sized against openness ∈ [0,1] and the panic
/// side term so an open goal-bound ray always outranks an open sideways one.
pub const CLEAR_GOAL_W: f64 = 0.5;
/// How far behind the ball (mm, along the chosen ray) the keeper holds its
/// pre-strike pose while waiting for the corridor to open.
pub const CLEAR_HOLD_STANDOFF: f64 = 250.0;
/// Radius (mm) of the keeper's work disc around the ball that the wall also
/// vacates while clearing. The forward corridor only frees the strike ray; the
/// keeper must additionally swing *around* the ball to line up (acquire side is
/// `Fastest`), and for a box-corner ball the wall shadows bunch right at the
/// keepout line next to it — observed in self-play as the keeper hovering
/// ~180mm short of a corner ball for 4s, boxed in by its own teammates.
pub const CLEAR_BALL_CLEARANCE: f64 = 700.0;

// ── Planner: in-field clamping ───────────────────────────────────────────────
/// Margin (mm) by which a kick-ahead pass/cross lead target is kept inside the
/// field boundary. A lead aims a pass into the space ahead of a receiver (toward
/// the opponent goal); for a deep or wide receiver that space can fall past the
/// goal line or a touchline, sending the ball out (stoppage + lost possession).
/// See [`crate::planner::Planner::clamp_in_field`].
pub const FIELD_LEAD_MARGIN: f64 = 250.0;

// ── Count-aware offensive stance ─────────────────────────────────────────────
/// Runtime-tunable stance parameters that make role allocation scale with the
/// number of field robots (3–6) and let the offence/defence balance be tuned
/// without recompiling. Loaded once from environment variables at strategy
/// startup (`StanceConfig::from_env`); each falls back to a default that
/// reproduces sensible behaviour at full strength.
///
/// The defensive wall is restructured into one always-present home **anchor**
/// plus 0..=`SHADOW_MAX-1` threat-scaled **wings** whose count scales with the
/// field-robot count, so a short-handed team thins its wall instead of pinning
/// every robot home. `balance` (the rest defender) and the forward block are
/// likewise gated/scaled by count and an aggression multiplier.
#[derive(Debug, Clone, Copy)]
pub struct StanceConfig {
    /// Bodies reserved for non-wall duty when sizing the wing wall. The number of
    /// staffed wings is `clamp(field_n - forward_reserve - 1, 0, SHADOW_MAX-1)`
    /// (the `-1` accounts for the always-present anchor). Lower → wider wall;
    /// higher → more bodies freed forward. Default 2.0:
    /// walls = {n3:1, n4:2, n5:3, n6:3} (anchor + wings).
    pub forward_reserve: f64,
    /// Minimum field-robot count at which the dedicated rest defender (`Balance`)
    /// is emitted. Below this the rest-defense body is freed forward instead.
    /// Default 5.0 → only 5- and 6-robot teams keep a dedicated rest defender.
    pub balance_min_bots: f64,
    /// Multiplier on forward-role importances (support, striker, pivot — NOT the
    /// counter outlet, which stays a deliberate fixture). >1 commits more bodies
    /// forward at every count; 1.0 = neutral. Below ~0.5 the forward block sinks
    /// to spread-tier importance and surplus robots loiter at midfield.
    pub aggression: f64,
    /// Override for [`SEC_PER_IMPORTANCE`] — seconds of robot travel one
    /// importance point is worth. Higher → importance outweighs travel, so robots
    /// commit to high-value (forward) roles from farther away. Default = 0.4.
    pub sec_per_importance: f64,
    /// Override for [`RECALC_COOLDOWN`] — minimum seconds between assignment
    /// recomputes. Default = `RECALC_COOLDOWN`.
    pub recalc_cooldown: f64,
    /// Override for [`RECALC_BG_PERIOD`] — background recompute period when no
    /// event forces one. Raising it holds the assignment longer → less role
    /// churn (at the cost of slower adaptation). Default = `RECALC_BG_PERIOD`.
    pub recalc_bg_period: f64,
}

impl Default for StanceConfig {
    fn default() -> Self {
        // Tuned by headless sweep vs the frozen baseline concerto (cvc goal_diff
        // fitness, counts 3–6, 20 seeds; see `.analysis/stance_sweep.py` and the
        // [[dynamic-robot-count]] memory). This point is net-positive at 4–6
        // robots with markedly higher offensive presence, while keeping the
        // rest-defender so a short-handed team doesn't over-extend and get
        // countered. (3v3 stays bunker-hard — the original defensive concerto
        // edges it there, but that extreme is rare.)
        StanceConfig {
            // Div-B conservative: max out the wing wall (wings = field_n - 2 - 1,
            // capped at 2 → anchor + 2 wings at 5 field robots). Peer-level
            // default was 3.0 (anchor + 1 wing at full strength).
            forward_reserve: 2.0,
            // Keep the dedicated rest defender at every count (gate effectively
            // off): the single biggest fix against being countered short-handed.
            balance_min_bots: 0.0,
            // Div-B conservative: soften the forward push further (peer-level
            // default was 0.9) — fewer bodies committed, the permanent outlet
            // (un-scaled by this) carries the forward presence instead.
            aggression: 0.9,
            sec_per_importance: SEC_PER_IMPORTANCE,
            recalc_cooldown: RECALC_COOLDOWN,
            recalc_bg_period: RECALC_BG_PERIOD,
        }
    }
}

impl StanceConfig {
    /// Load from environment, falling back to defaults. Env keys:
    /// `CONCERTO_FORWARD_RESERVE`, `CONCERTO_BALANCE_MIN_BOTS`,
    /// `CONCERTO_AGGRESSION`, `CONCERTO_SEC_PER_IMP`, `CONCERTO_RECALC_CD`,
    /// `CONCERTO_RECALC_BG`.
    pub fn from_env() -> Self {
        let d = StanceConfig::default();
        StanceConfig {
            forward_reserve: env_f64("CONCERTO_FORWARD_RESERVE", d.forward_reserve),
            balance_min_bots: env_f64("CONCERTO_BALANCE_MIN_BOTS", d.balance_min_bots),
            aggression: env_f64("CONCERTO_AGGRESSION", d.aggression),
            sec_per_importance: env_f64("CONCERTO_SEC_PER_IMP", d.sec_per_importance),
            recalc_cooldown: env_f64("CONCERTO_RECALC_CD", d.recalc_cooldown),
            recalc_bg_period: env_f64("CONCERTO_RECALC_BG", d.recalc_bg_period),
        }
    }
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(default)
}
