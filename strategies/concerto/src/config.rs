//! Tunable parameters for the Concerto strategy.
//!
//! All physical constants live here so behaviour can be tuned in one place.
//! Distances are in mm, times in seconds, speeds in mm/s, accelerations in mm/s².

// ── Attacking-half recycle pivot ─────────────────────────────────────────────
// When the carrier has no good *forward* option in the opponent half, it lays the
// ball back to a deep recycle pivot to keep possession, rather than forcing a
// low-percentage forward pass or hoofing into space (the two dominant turnover
// sources). Gated to the attacking half so we never circulate the ball in our own
// half, where a turnover becomes a goal. In headless self-play vs v0 (32 paired
// seeds) this lifted pass completion ~2.4×, cut turnovers ~40%, lengthened
// possession chains, and *raised* shots/SOT — with goals-for and goals-against vs
// v0 unchanged. Earlier variants that recycled everywhere (own half too) conceded
// significantly more and lost; the attacking-half gate is what makes it pay.
//
/// Forward-pass quality (lane openness × receiver clearance, 0..1) at/above which a
/// forward pass is taken outright instead of recycling. Below it, a weak forward
/// pass (or a hoof into space) is recycled when a safe outlet exists.
pub const FORWARD_OK_BAR: f64 = 0.50;

/// Global pass-propensity knob — a single scalar multiplied into **every**
/// pass-success estimate concerto computes (forward-pass quality, forward-outlet
/// eligibility, and recycle openness). It is the one global lever for "how much to
/// pass vs just shoot":
/// - `1.0` (default) — neutral; behaviour is unchanged from an ungated build.
/// - `(0, 1)` — progressively discount every pass's assessed success, so more
///   forward balls fall below [`FORWARD_OK_BAR`] / [`SUPPORTER_MIN_OPENNESS`] and
///   are downgraded to a strike-finish, hoof, or short carry. Biases toward
///   shooting/carrying.
/// - `0.0` — passing is fully disabled: no pass ever clears its gate, so the
///   planner never emits a `Waypoint::Pass`. The carrier always shoots, carries,
///   or hoofs; a set-piece kicker still releases forward into open space.
/// - `> 1.0` — biases toward passing (marginal lanes clear their gate more often).
///
/// Because every pass gate scales linearly with it, `0.0` disabling passing falls
/// out of the arithmetic — no special-casing needed.
///
/// Div-B conservative mode: 0.0 — passes are the dominant self-inflicted turnover
/// source against lower-tier opponents, so the carrier always shoots, carries, or
/// hoofs. Restore 1.0 for peer-level play.
pub const PASS_SUCCESS_BASE: f64 = 0.0;
/// How far behind the ball (toward our own half) the recycle pivot sits, on the
/// ball's flank. Deep enough to be a safe lay-back outlet, never up in the box
/// where an extra central body would invite the defence to collapse the shot zone.
pub const PIVOT_SETBACK: f64 = 1400.0;
/// Lateral band (|y|) the pivot holds — ball-side half-space, clamped so it stays
/// central enough to switch play but off the exact ball line.
pub const PIVOT_Y_MIN: f64 = 300.0;
pub const PIVOT_Y_MAX: f64 = 1800.0;
/// Importance of the pivot. Kept at the conservative-support level so a *surplus*
/// body takes it; it never outbids a shadow/mark and pulls a defender out of shape
/// (the defensive cost that sank the more prominent-pivot variants).
pub const IMP_PIVOT: f64 = 3.0;
/// ── Rest defense (lever 2: don't over-commit on the counter) ────────────────
/// While attacking, all field robots but the keeper + box anchor pile forward
/// (2 wide support + box-runner + pivot), so a turnover leaves the goal to the
/// keeper + anchor alone and the wing wall has nobody recoverable to staff it.
/// The Balance role keeps ONE body home as rest defense: it is placed on the
/// ball→own-goal ray this far (mm) in front of our goal — own-half/midfield, on
/// the most likely counter lane — so on turnover it is already there to take a
/// wing of the wall while the deep attackers recover.
pub const BALANCE_STANDOFF: f64 = 3500.0;
/// Importance of the rest-defense Balance role while attacking. Set above the
/// forward block (IMP_SUPPORT_ATTACK = 5.0, IMP_STRIKER = 5.5) so it wins exactly
/// one body — displacing the *pivot* (IMP_PIVOT = 3.0, the least-critical forward
/// outlet), not a support — but below the box anchor (8.0 × 0.8 = 6.4), so it
/// never outranks the genuine last line. Sits above the discounted Mark (≤4),
/// which is intended: rest-defense shape beats man-marking in the current
/// tuning. Only emitted when `attacking`.
pub const IMP_BALANCE: f64 = 5.0;
/// A recycle outlet must be at least this far from the carrier (a real switch of
/// play, not a tap) and its lane this open, to be worth laying the ball back to.
pub const RECYCLE_MIN_DIST: f64 = 1200.0;
pub const RECYCLE_MIN_OPENNESS: f64 = 0.6;

// ── Fast-break commit (transient counter-attack) ─────────────────────────────
// The patient recycle (above) keeps possession but is wrong on a clean
// counter-attack: in the 1–2 s after we win the ball the opponent is still
// recovering, so the open lane is *forward*, not backward. For a short window
// after a turnover, when we have a genuine numerical break — strictly more of our
// own bodies goal-side of the ball than the opponent has (its keeper excluded) —
// we commit the ball forward (kick-ahead to a supporter, or a hoof into the space
// behind their line) instead of laying it back. This is net-asymmetric: it can
// only fire for the team that just recovered the ball, so a frozen mirror of
// ourselves can't symmetrically copy it; and it is self-limiting — once the
// defence recovers its numbers goal-side, the break gate closes and play reverts
// to the patient shape. The numerical gate keeps it safe: we never gamble the
// direct ball when we are outnumbered at the back.
/// Seconds after winning the ball during which a numerical break is committed
/// forward rather than recycled. Outside this window play is always patient.
pub const FAST_BREAK_WINDOW: f64 = 1.5;

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

/// First-time finish: a *contested* ball in the attacking strike zone (within
/// [`STRIKE_FINISH_RANGE`] of the opponent goal) is struck goalward on contact
/// rather than settled. Trying to settle a genuine 50/50 in the final third loses
/// the ball and concedes a counter — the recurring finishing-battery failure
/// (ball sits in the attacking third, the 50/50 is lost, the opponent counters
/// the length of the field). A reflex strike-through forces a save/rebound/corner
/// and pins the ball in the attacking third instead. The contest is "genuine" when
/// an opponent is within [`FINISH_CONTEST_OPP_DIST`] of the ball AND our actor is
/// within [`FINISH_CONTEST_OUR_DIST`] — i.e. a reflex strike is imminent, not a
/// long traverse during which the picture changes. If we'd win the ball clean (no
/// opponent near), we settle and build instead of hacking at goal.
pub const FINISH_CONTEST_OPP_DIST: f64 = 600.0;
pub const FINISH_CONTEST_OUR_DIST: f64 = 750.0;

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
pub const SHOT_MIN_ANGLE: f64 = 0.10;
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

// ── Never fire a restart into a close opponent ───────────────────────────────
// A universal veto on any restart aim (direct shot / kickoff deflection / forward
// advance) whose ray drives the ball almost immediately into a nearby opponent —
// worst of all in our own half, where a blocked restart hands them a shot on our
// goal. Reach and corridor are widened when the ball is in our half.
/// An opponent within this range of the ball, ahead on the aim ray, vetoes the aim.
pub const CLOSE_OPP_VETO_DIST: f64 = 1200.0;
/// Perpendicular half-width (mm) of the veto corridor around the aim ray.
pub const CLOSE_OPP_VETO_CORRIDOR: f64 = 400.0;
/// Multiplier on the veto reach and corridor when the ball is in our own half.
pub const CLOSE_OPP_VETO_OWN_HALF_MUL: f64 = 1.5;

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
/// Lateral step a contested carrier strafes off the squeeze axis to break a pin
/// while keeping the ball (the keep-possession escape). Short — it only needs to
/// move the opponent out from between us and goal, after which we replan normally.
pub const ESCAPE_STEP: f64 = 300.0;
/// |y| of the pressed contest-escape clear's aim point on the opponent goal
/// line, on the ball's flank side — inside the mouth (goal half-width 500) but
/// off centre, so a full-length roll-out is a goal or a keeper touch, never a
/// Div-B aimless-kick reset. (The keeper's own clear no longer uses a fixed
/// aim point — see `geometry::best_clear_ray`.)
pub const ESCAPE_CLEAR_AIM_Y: f64 = 250.0;

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

// ── Kick-ahead (hoof toward a supporter) ─────────────────────────────────────
// When an open-play hoof is aimed near a forward teammate, we don't fire-and-
// forget: that teammate is pinned as the ball-winner and, once the kick clears,
// collects the ball in flight (a plain `Hold` acquire — no receive prediction,
// just pick it up whichever way it comes). Distinct from a coordinated `Pass`:
// the carrier never waits, it just hoofs, and the runner chases.
/// Maximum perpendicular distance (mm) of a forward teammate from the hoof ray
/// for it to be treated as the kick-ahead receiver. Tighter than the hoof's own
/// mate-bias radius so we only commit a runner when the ball is genuinely on its
/// path.
pub const KICKAHEAD_MAX_PERP: f64 = 1000.0;
/// How far (mm) the ball must travel from the kick origin for the kick to count
/// as "cleared" — once cleared, the runner stays committed to the chase.
pub const KICKAHEAD_CLEAR_DIST: f64 = 400.0;
/// If the ball hasn't cleared [`KICKAHEAD_CLEAR_DIST`] within this many seconds
/// of the kick releasing (kick whiffed / blocked / stuck on the kicker), abort
/// the kick-ahead and replan the loose ball normally. World-time, not wall-clock.
pub const KICKAHEAD_CLEAR_TIMEOUT: f64 = 0.6;

// ── Driver ──────────────────────────────────────────────────────────────────
/// Ball displacement from the engagement point that aborts any waypoint. This is
/// the driver's only remaining ball-handling guard — a premise-void replan
/// trigger, not an execution timeout (those now live inside the skills).
pub const BALL_MOVED_DIST: f64 = 2000.0;

// ── Formation ─────────────────────────────────────────────────────────────
/// Threat ramp: distance to our goal at which threat is maximal / negligible.
pub const THREAT_GOAL_NEAR: f64 = 1500.0;
pub const THREAT_GOAL_FAR: f64 = 6000.0;
/// Shadow (goal-coverage) role count scales with threat between these bounds.
/// (Fix A: SHADOW_MIN is retained for reference but the goal wall now always
/// emits `SHADOW_MAX` shadows; the *staffed* count emerges from importance.)
pub const SHADOW_MIN: usize = 1;
pub const SHADOW_MAX: usize = 3;
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
pub const SHADOW_STANDOFF: f64 = 1600.0;
/// Lateral spacing between adjacent shadows in the goal-coverage wall. Kept just
/// under a robot width (~180mm) so neighbours overlap and leave no central gap —
/// the wall is built centre-out on the direct ball→goal line, so an even count
/// straddles the centre rather than splitting around it.
pub const SHADOW_SPACING: f64 = 170.0;
/// How far outside the defense-area box (mm) the shadow **wing** curve sits. The
/// wall now slides along a rounded-rectangle offset of the box rather than a radial
/// line: this is the wings' offset, and [`ANCHOR_BOX_MARGIN`] the (tighter) anchor's.
/// Together they replace the goal-centre-radial `SHADOW_STANDOFF`. Sits ~one
/// robot-width-plus outside the anchor so a 2-body wall never stacks. Tuning surface
/// — changes wall depth/coverage; validate in sim.
pub const SHADOW_BOX_OFFSET: f64 = 550.0;
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
/// Incumbency bonus that stabilises a supporter's target against frame-to-frame
/// thrashing. `best_support_pos` is a discrete argmax over a grid of candidate
/// outlets scored by `lane_openness + receiver_clearance` (both in [0, 1]); with
/// opponent positions jittering under Kalman noise, the top two candidates are
/// often near-tied and the winner flips every tick — yanking the target across a
/// grid cell (or the whole field, on a flank flip) and leaving the supporter
/// dithering in place instead of arriving. The candidate nearest last tick's
/// target for this slot gets up to `+SUPPORT_STICKINESS` added to its score, so a
/// challenger must be *decisively* more open (by this margin in the 0..2 score
/// space) to displace the incumbent — hysteresis in argmax form, the same
/// commitment pattern as the capture-role discount. Small enough that a genuinely
/// better outlet (openness swing ≥ this) still wins and the supporter follows the
/// ball as the whole grid advances.
pub const SUPPORT_STICKINESS: f64 = 0.18;
/// Distance over which the incumbency bonus decays to zero (mm). Sized well below
/// the grid spacing (~1000mm in x, wider across flanks) so only the candidate
/// essentially *at* last tick's target keeps the full bonus and its neighbours get
/// none — the bonus favours holding the current outlet, not a whole region. Also
/// absorbs the small per-tick translation of the ball-relative grid so the
/// incumbent stays sticky through jitter while still advancing with the ball.
pub const SUPPORT_STICKY_RADIUS: f64 = 600.0;
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

/// A ball contest with at least this much threat to our goal relieves one shadow
/// (the contesting plan robot stands in for it). Keeps shadows off the snatcher.
pub const SHADOW_RELIEF_THREAT: f64 = 0.4;
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
pub const SUPPRESS_RADIUS: f64 = 500.0;
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
pub const CLEAR_PANIC_T: f64 = 6.0;
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
