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
/// outlet), not a support — but below the box anchor (8.0 × 0.8 = 6.4) and Mark,
/// so it never outranks the genuine last line. Only emitted when `attacking`.
pub const IMP_BALANCE: f64 = 6.0;
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
// A full-power `Shoot` kick fires at a fixed ~4000 mm/s and the ball rolls
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
pub const SHADOW_STANDOFF: f64 = 1500.0;
/// Lateral spacing between adjacent shadows in the goal-coverage wall. Kept just
/// under a robot width (~180mm) so neighbours overlap and leave no central gap —
/// the wall is built centre-out on the direct ball→goal line, so an even count
/// straddles the centre rather than splitting around it.
pub const SHADOW_SPACING: f64 = 170.0;
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
/// to outbid Mark (≤6) for the nearest bodies when the goal is threatened, so the
/// posts get covered on a counter. Also lifts the anchor floor (×0.8 → 8.0); both
/// stay below Capture (13), so a genuine ball-winner can still pull them.
pub const IMP_SHADOW_BASE: f64 = 10.0;
pub const IMP_MARK_BASE: f64 = 6.0;
pub const IMP_SUPPORT: f64 = 3.0;
/// Support importance while attacking (ball in the opponent half, own goal safe).
/// Above the weak-mark / spread range so surplus bodies win the assignment as
/// supporters and push up instead of loitering at midfield.
pub const IMP_SUPPORT_ATTACK: f64 = 5.0;
pub const IMP_SPREAD: f64 = 0.5;
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
/// Field-absolute x of the counter outlet (mm, team-relative; +x = opponent
/// half). Just into the opponent half so a clear/switch has an advanced target and
/// the body is a genuine counter springboard rather than another midfielder.
pub const OUTLET_X: f64 = 800.0;
/// Lateral placement of the outlet as a fraction of half-width, on the flank away
/// from the ball (where a switch/clearance naturally goes).
pub const OUTLET_Y_FRAC: f64 = 0.55;
/// Importance of the counter outlet at full threat. Below Mark/Balance (6) and the
/// Shadow wall (10) so it never strips the goal wall — it claims at most the
/// least-critical surplus body — but above plain Support (3)/Spread so that body
/// holds high instead of collapsing all the way home.
pub const IMP_OUTLET: f64 = 5.0;
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
pub const KEEPER_ARC_RADIUS: f64 = 400.0;
/// Maximum angular excursion of the keeper off straight-out (radians). A pure
/// *sanity* bound now (the lateral mouth clamp is the primary limit): keeps the
/// keeper a little in front of the line rather than drifting onto it at extreme
/// angles. Widened from the old 45° — that clamp parked the keeper near goal-centre
/// and left the near post wide open against oblique (corner) shots. Also bounds the
/// guard zone's angular span.
pub const KEEPER_ARC_MAX_ANGLE: f64 = 80.0 * std::f64::consts::PI / 180.0; // 80°
/// How far inside each post (mm) the keeper centre may travel laterally — the
/// "stay in front of your net" clamp (Option B). With `KEEPER_ARC_RADIUS` < half the
/// goal width this is slack (the arc caps lateral reach first); it becomes the
/// binding limit if the radius is ever grown past the mouth half-width.
pub const KEEPER_MOUTH_MARGIN: f64 = 80.0;
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
    /// Multiplier on forward-role importances (support, striker, pivot, outlet).
    /// >1 commits more bodies forward at every count; 1.0 = neutral. Default 1.0.
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
            // Thin the wing wall (wings = field_n - 3 - 1, so 0 at ≤4 field
            // robots, 1 at 5) — frees bodies forward at higher counts.
            forward_reserve: 3.0,
            // Keep the dedicated rest defender at every count (gate effectively
            // off): the single biggest fix against being countered short-handed.
            balance_min_bots: 2.0,
            // Slightly soften the forward push so committed bodies don't
            // over-extend into turnovers.
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
