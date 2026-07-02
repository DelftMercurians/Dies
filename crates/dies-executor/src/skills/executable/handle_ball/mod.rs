//! HandleBall — unified acquire + carry + terminal-action ball handling.
//!
//! Acquire *and* the terminal action are one [`SkillCommand`] **variant**, so the
//! [`SkillExecutor`](crate::control::skill_executor::SkillExecutor) updates this
//! skill's params in place when the strategy swaps the action (e.g. once the ball
//! is secured) rather than tearing it down and rebuilding — so the delicate
//! acquire→act transition no longer discards capture-phase state.
//!
//! ## Module layout
//! The skill is split by phase, with every tuning constant in this `mod.rs` so
//! there is one tuning site:
//! - [`acquire`] — the capture front-end (side selection, staging, commit drive)
//!   and the shared ball-geometry helpers.
//! - [`aim`] — the `Shoot` path: launch-point selection, orbit-to-aim, kick, and
//!   the post-kick departure verify.
//! - [`strike`] — the `Strike` reflex strike-through.
//! - `Hold`/`Carry` are tiny drives and live here.
//!
//! ## Terminal contract (keeps the seam removal sound)
//! The skill owns its whole execution lifecycle — every timeout lives here, not in
//! the caller:
//! - `Hold` **never** returns `Done` — it runs until the caller swaps the action
//!   (each in-possession action swap is a live param update, not a teardown).
//! - `Carry` self-completes: `Done(Success)` on arrival ([`DRIBBLE_ARRIVE_DIST`]),
//!   `Done(Failure)` on its own timeout ([`DRIBBLE_TIMEOUT`]).
//! - A kick (`Shoot`/`Strike`) is `Done(Success)` on the verified departure; the aim
//!   stage fails on [`AIM_BACKSTOP`].
//! - The acquire front-end fails on its own [`APPROACH_TIMEOUT`] / [`PICKUP_TIMEOUT`].
//! - Internal capture completion (breakbeam) is an internal stage edge, not a
//!   `Done` — that is what makes the acquire→act seam disappear.
//!
//! ## Silent re-acquire
//! Losing the ball mid-aim/carry returns to `Acquire` (debounced) instead of
//! failing up to the strategy. Bounded by [`MAX_REACQUIRE`] and the acquire timeouts
//! so a persistent loss still surfaces as `Done(Failure)` and the caller can re-elect
//! a capturer. `Strike` never re-acquires (double-touch safe).

mod acquire;
mod aim;
mod strike;

use std::time::Duration;

use dies_core::{Angle, DebugColor, Vector2, PLAYER_RADIUS};
use dies_strategy_protocol::{AcquirePosition, BallAction, SkillCommand, SkillStatus};
use dies_tunables_macro::tunables;

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::PlayerControlInput;

// ── structural / derived constants (NOT runtime-tunable) ─────────────────────
// These stay compile-time const: counts, fixed sample fans, hardware timeouts, and
// values derived from physical radii. Only the behaviour knobs below are exposed.
/// Max ball losses before the skill gives up instead of re-acquiring forever.
const MAX_REACQUIRE: u32 = 6;
/// Number of candidate push directions sampled around the ball.
const N_APPROACH_SAMPLES: usize = 24;
/// Ego clearance required at a candidate staging point (robot radius + buffer).
const APPROACH_EGO_RADIUS: f64 = PLAYER_RADIUS + 20.0;
/// Settle band around the staging point inside which the anti-stiction feed is
/// off. Must stay below COMMIT_PERP so a robot resting anywhere inside the band
/// is already within the commit gate (matches the follower's arrive deadband).
const STAGING_SETTLE_DIST: f64 = 15.0;
/// Clearance required along the commit corridor (staging point → ball). Slightly
/// tighter so only a real blocker between us and the ball rejects an approach.
const APPROACH_COMMIT_RADIUS: f64 = PLAYER_RADIUS;
/// How long to wait after arming the reflex for the ball to depart before
/// declaring a whiff and failing (so a dribbler-on hold can't linger and
/// accumulate double-touch contact).
const REFLEX_TIMEOUT: Duration = Duration::from_millis(5000);
/// How long to wait for the ball to leave after commanding a kick before
/// declaring the kick a whiff and failing.
const VERIFY_WINDOW: Duration = Duration::from_millis(200);
/// Clearance required at a candidate kicking pose (robot behind the launch point).
const LAUNCH_POSE_EGO: f64 = PLAYER_RADIUS;
/// Carry distances (mm) tried when searching for a reachable launch point. Short
/// — repositioning is a small nudge off a bad spot, not a dribble across field.
const CARRY_STEPS: [f64; 3] = [300.0, 500.0, 700.0];
/// Lateral fan (radians) tried around the inward direction at each carry step.
const CARRY_FAN: [f64; 5] = [0.0, 0.5, -0.5, 1.0, -1.0];

tunables! {
    // ── episode timing / thrash backstops ────────────────────────────────────
    section "HandleBall timing";

    /// Debounce before a ball loss during an act stage triggers a re-acquire — a
    /// pass-style ride on the dribbler edge routinely drops the breakbeam for a frame
    /// or two. Mirrors the pass coordinator's `SETUP_BALL_LOST_GRACE`.
    #[tunable(unit = "s", min = 0.0, max = 1.0, step = 0.05)]
    BALL_LOST_GRACE: f64 = 0.2;
    /// Possession-loss debounce when the ball is still NEAR (inside
    /// [`LOST_BALL_DISTANCE`]): a ball resting just in front of the dribbler
    /// (~110–220 mm centre-to-centre) is invisible to the breakbeam yet not "far",
    /// so without this the act stages held forever with no ball (the 2026-07-02
    /// GreenTea freeze). Long enough to ride out dribbler-contact flicker, short
    /// enough that a bounced-off ball is re-acquired promptly.
    #[tunable(unit = "s", min = 0.1, max = 2.0, step = 0.05)]
    NEAR_LOST_GRACE: f64 = 0.5;
    /// Approach timeout: while still traversing toward the ball (not yet committed to
    /// the final capture drive), fail this long after entering the acquire stage so a
    /// robot that can't reach the ball surfaces a failure and the caller re-decides.
    /// Re-armed on every re-acquire (attempts bounded by `MAX_REACQUIRE`).
    #[tunable(unit = "s", min = 1.0, max = 8.0, step = 0.5)]
    APPROACH_TIMEOUT: f64 = 3.0;
    /// Pickup timeout: once committed to the final capture drive (pressed against the
    /// ball) fail this long after the commit latch engages if the breakbeam never
    /// latches — the fast bail for a ball pinned/wedged so we can't take it.
    #[tunable(unit = "s", min = 0.5, max = 6.0, step = 0.5)]
    PICKUP_TIMEOUT: f64 = 2.0;
    /// Aim hold cap: this long after entering the aim stage the shot is FORCED —
    /// reposition is abandoned, the orbit ignores `lane_blocked`, and the kick
    /// commits on the loose [`AIM_FORCED_YAW_DEG`] gate instead of the precise
    /// one. Failing here instead (the old behaviour) just re-engaged the same
    /// robot with a fresh clock, so a covered lane produced an endless hold-and-
    /// spin (observed 9–17 s of continuous dribbler contact — excessive dribbling,
    /// and a ball long lost IRL). Booting the ball through the covered corridor
    /// is strictly better than holding it. Sized to leave room for the capture-
    /// settle window (SETTLE_TIME) that delays the start of the orbit.
    #[tunable(unit = "s", min = 1.0, max = 12.0, step = 0.5)]
    AIM_BACKSTOP: f64 = 2.5;
    /// Yaw-error gate for a forced (past-backstop) shot. Loose on purpose: after
    /// the cap the priority is releasing the ball roughly forward, not precision.
    #[tunable(unit = "deg", min = 5.0, max = 60.0, step = 5.0)]
    AIM_FORCED_YAW_DEG: f64 = 20.0;
    /// Hard failure deadline for the aim stage (multiple of [`AIM_BACKSTOP`]): if
    /// even the forced shot can't commit by then (ball never seated, alignment
    /// unreachable), give up so the caller re-decides.
    #[tunable(unit = "x", min = 1.5, max = 4.0, step = 0.5)]
    AIM_GIVE_UP_FACTOR: f64 = 2.0;
    /// Carry timeout: fail this long after entering the carry stage if the target is
    /// still not reached (e.g. a blocked path).
    #[tunable(unit = "s", min = 1.0, max = 12.0, step = 0.5)]
    DRIBBLE_TIMEOUT: f64 = 4.0;
    /// Distance to the carry target at which the carry self-completes (arrived). Kept
    /// below the planner's correction step so a corrective carry is a real move, not an
    /// instant "already there".
    #[tunable(unit = "mm", min = 30.0, max = 400.0, step = 10.0)]
    DRIBBLE_ARRIVE_DIST: f64 = 150.0;

    // ── capture settle (post-breakbeam glide) ─────────────────────────────────
    section "HandleBall settle";

    /// Deceleration cap for the settle window right after capture. Stopping dead
    /// at contact bounces the ball off the dribbler (the ball still carries the
    /// commit-drive speed); instead the last command glides down at this rate —
    /// from a 300 mm/s contact that is ~0.4 s and ~55 mm of drive-through while
    /// the dribbler seats the ball. The glide direction is the commit axis (the
    /// controller ramps down its previous command), so ball keep-in/defense
    /// margins from the commit still cover it.
    #[tunable(unit = "mm/s²", min = 200.0, max = 4000.0, step = 100.0)]
    SETTLE_DECEL: f64 = 800.0;
    /// Length of the settle window after the acquire→act transition.
    #[tunable(unit = "s", min = 0.0, max = 1.5, step = 0.05)]
    SETTLE_TIME: f64 = 0.4;
    /// Yaw-rate cap during the settle window: spinning toward the act heading in
    /// the same tick as capture rips the ball off the dribbler just like the
    /// hard stop does. Plumbed to the firmware heading controller (max_yaw_rate).
    #[tunable(unit = "rad/s", min = 0.2, max = 6.0, step = 0.1)]
    SETTLE_YAW_RATE: f64 = 1.5;

    // ── dribbler ──────────────────────────────────────────────────────────────
    section "HandleBall dribbler";

    /// Dribbler speed used while acquiring, holding, aiming, and reflex-striking.
    #[tunable(min = 0.0, max = 1.0, step = 0.05)]
    DRIBBLER_SPEED: f64 = 0.6;
    /// Dribbler speed while carrying the ball to a `Carry` target (matches `Dribble`).
    #[tunable(min = 0.0, max = 1.0, step = 0.05)]
    CARRY_DRIBBLER_SPEED: f64 = 0.5;

    // ── magnet capture ──────────────────────────────────────────────────────────
    section "HandleBall magnet";

    /// Master enable for firmware magnet capture during the commit drive (`>0.5` =
    /// on). ANDed with the per-command strategy flag and the robot's ToF
    /// capability. A no-op on robots without a working ToF sensor and in sim.
    #[tunable(min = 0.0, max = 1.0, step = 1.0)]
    MAGNET_ENABLED: f64 = 1.0;

    // ── commit-drive containment / safety ─────────────────────────────────────
    section "HandleBall commit safety";

    /// Wall margin scale during a commit drive (which runs walls-only ORCA with
    /// `avoid_robots = false`). `0` = robot edge may reach the wall, so we can
    /// strike/capture a ball hard against the boundary; `1` = the full margin.
    #[tunable(min = 0.0, max = 1.0, step = 0.1)]
    COMMIT_WALL_CARE: f64 = 0.0;
    /// Commit drives drop defense-box ORCA, so they must bail rather than wander
    /// into a defense area. Bail if the robot centre is within this of a defense
    /// area (≈ robot radius + a small buffer, so we bail before the body crosses).
    #[tunable(unit = "mm", min = 0.0, max = 400.0, step = 10.0)]
    DEFENSE_BAIL_MARGIN: f64 = 120.0;
    /// Bail if the ball we would touch is within this of a defense area — touching
    /// a ball inside the opponent area (or our own, keeper's job) is a foul.
    #[tunable(unit = "mm", min = 0.0, max = 300.0, step = 10.0)]
    BALL_IN_BOX_MARGIN: f64 = 30.0;

    // ── acquire (capture front-end) ───────────────────────────────────────────
    section "HandleBall acquire";

    /// Distance to the ball at which the approach begins decelerating in.
    #[tunable(unit = "mm", min = 50.0, max = 500.0, step = 10.0)]
    APPROACH_DISTANCE: f64 = 200.0;
    /// Distance to the ball below which the capture commits to its final drive.
    #[tunable(unit = "mm", min = 50.0, max = 500.0, step = 10.0)]
    COMMIT_DISTANCE: f64 = 280.0;
    /// Perpendicular tolerance to the commit axis while committed.
    #[tunable(unit = "mm", min = 5.0, max = 150.0, step = 5.0)]
    COMMIT_PERP: f64 = 30.0;
    /// Release (hysteresis) band for the commit latch. Once committed we keep
    /// driving while perp stays under this, and only bail if it blows past it. Kept
    /// ≤ GATE_PERP so the proportional drive is already throttled near the
    /// boundary — the robot slides back onto the axis rather than ramming the ball.
    #[tunable(unit = "mm", min = 30.0, max = 200.0, step = 5.0)]
    COMMIT_PERP_RELEASE: f64 = 70.0;
    /// Along-axis overshoot tolerated while latched: the ball estimate collapses
    /// into the hull during the final press (vision occlusion puts it at the robot
    /// centre), so `along` crossing 0 must not drop the latch and disarm the kicker.
    #[tunable(unit = "mm", min = 0.0, max = 100.0, step = 5.0)]
    COMMIT_ALONG_OVERSHOOT: f64 = 30.0;
    /// Along-axis release margin past COMMIT_DISTANCE while latched, so hovering
    /// right at the commit boundary can't flap the latch (each flap used to count
    /// a re-acquire and one stuck approach exhausted the whole budget).
    #[tunable(unit = "mm", min = 0.0, max = 150.0, step = 5.0)]
    COMMIT_ALONG_RELEASE: f64 = 60.0;
    /// Perpendicular gate width scaling the *proportional* term of the commit
    /// drive (the MIN_SPEED floor is not gated — see `commit_velocity`).
    #[tunable(unit = "mm", min = 20.0, max = 200.0, step = 5.0)]
    GATE_PERP: f64 = 80.0;
    /// Proportional gain on the approach drive toward the ball.
    #[tunable(min = 0.5, max = 8.0, step = 0.25)]
    APPROACH_GAIN: f64 = 3.0;
    /// Minimum approach speed so the final drive doesn't stall short of the ball.
    /// Applied as an ungated floor: it is also the guaranteed ball-contact speed.
    /// Live data (2026-07-02): contact ≥ ~290 mm/s → one clean breakbeam edge and
    /// the reflex fires; contact ≤ ~100 mm/s → beam flicker and whiffs.
    #[tunable(unit = "mm/s", min = 50.0, max = 600.0, step = 25.0)]
    APPROACH_MIN_SPEED: f64 = 300.0;
    /// Anti-stiction feed toward the staging point while uncommitted: added on
    /// top of the position controller whenever the robot is more than the settle
    /// band from staging, so the terminal proportional command can't decay into
    /// the drivetrain's stiction band and park the robot outside the commit gate
    /// (observed: sustained 170–260 mm/s commands left the robot stationary;
    /// breakaway needed ~400 mm/s commanded).
    #[tunable(unit = "mm/s", min = 0.0, max = 600.0, step = 25.0)]
    STAGING_MIN_SPEED: f64 = 300.0;
    /// Lateral correction gain that holds the robot on the commit axis.
    #[tunable(min = 0.5, max = 10.0, step = 0.25)]
    LATERAL_GAIN: f64 = 6.0;
    /// Care factor during the approach (negative = aggressive, less braking).
    #[tunable(min = -1.0, max = 1.0, step = 0.05)]
    APPROACH_CARE: f64 = -0.6;
    /// Ball displacement from the engagement point that fails a static capture.
    #[tunable(unit = "mm", min = 30.0, max = 400.0, step = 10.0)]
    BALL_MOVED_FAIL: f64 = 100.0;
    /// Distance driven without progress that fails the capture.
    #[tunable(unit = "mm", min = 50.0, max = 600.0, step = 10.0)]
    DRIVEN_FAIL: f64 = 250.0;
    /// Inset from the *physical* field edge (touchline/goal line **plus** the run-off
    /// `boundary_width`) that the staging point is kept inside. Robots may legally
    /// stand in the run-off, so the staging point only has to stay on the playing
    /// surface — clamping to the line itself would forbid getting *behind* a ball
    /// pinned on the line. Covers the robot radius plus a buffer.
    #[tunable(unit = "mm", min = 0.0, max = 400.0, step = 10.0)]
    STAGING_FIELD_MARGIN: f64 = 130.0;

    // ── heading-free approach-side selection ──────────────────────────────────
    section "HandleBall approach-side";

    /// Sampling step for the commit-corridor clearance check.
    #[tunable(unit = "mm", min = 20.0, max = 200.0, step = 10.0)]
    APPROACH_CLEAR_STEP: f64 = 60.0;
    /// Value (mm of approach distance) of a perfectly exit-aligned push. Trades off
    /// against "closest staging point": a well-aimed capture is worth approaching from
    /// a bit farther away. Soft — boundary safety is a hard constraint.
    #[tunable(unit = "mm", min = 0.0, max = 1500.0, step = 50.0)]
    EXIT_BIAS_WEIGHT: f64 = 600.0;
    /// The chosen side is kept unless a candidate beats it by more than this, so the
    /// approach side doesn't flip-flop on noise / orbiting opponents.
    #[tunable(unit = "mm", min = 0.0, max = 600.0, step = 25.0)]
    APPROACH_HYSTERESIS: f64 = 200.0;
    /// A ball within this distance of a field line forbids any approach whose push has
    /// an outward component there — never dribble the ball out (the touchline/goal-line
    /// rescue, folded into the generic selector).
    #[tunable(unit = "mm", min = 100.0, max = 800.0, step = 25.0)]
    BALL_KEEPIN_MARGIN: f64 = 350.0;
    /// Tolerance on the inward-push constraint (slightly negative so a near-parallel
    /// push along the line is still allowed).
    #[tunable(min = -0.5, max = 0.5, step = 0.05)]
    MIN_INWARD_PUSH: f64 = -0.05;

    // ── moving-ball tail-catch + contact offset ───────────────────────────────
    section "HandleBall moving-ball";

    /// Lateral contact offset: the ball is centred this far to the robot's LEFT of
    /// dribbler centre so it rests clear of the dribbler-drive gear on the right.
    #[tunable(unit = "mm", min = -20.0, max = 20.0, step = 0.5)]
    PICKUP_LATERAL_OFFSET: f64 = 2.5;
    /// Ball speed above which a capture switches from static side-selection to a
    /// velocity-aware tail-catch (lead the intercept, feed-forward the ball speed).
    /// Below this the ball is effectively static and the side-selector is better.
    #[tunable(unit = "mm/s", min = 100.0, max = 1000.0, step = 25.0)]
    MOVING_BALL_SPEED: f64 = 300.0;
    /// Cosine of the half-cone within which a ball rolling *toward* the robot is
    /// treated as head-on and left to `Receive` (a tail-catch would loop around).
    #[tunable(min = 0.0, max = 1.0, step = 0.05)]
    HEAD_ON_COS: f64 = 0.6;
    /// Nominal robot speed for the intercept-time estimate. Conservative (below true
    /// max) so the aim point lands slightly short rather than past the ball.
    #[tunable(unit = "mm/s", min = 1000.0, max = 4000.0, step = 100.0)]
    INTERCEPT_ROBOT_SPEED: f64 = 2500.0;
    /// Cap on the intercept look-ahead.
    #[tunable(unit = "s", min = 0.2, max = 4.0, step = 0.1)]
    MAX_INTERCEPT_TIME: f64 = 1.5;
    /// Lateral deviation of a moving ball from the commit axis that counts as the ball
    /// squirting out of the corridor (replaces `BALL_MOVED_FAIL` while rolling along).
    #[tunable(unit = "mm", min = 50.0, max = 400.0, step = 10.0)]
    BALL_STRAY_FAIL: f64 = 150.0;

    // ── reflex strike (Strike) ────────────────────────────────────────────────
    section "HandleBall strike";

    /// Ball displacement along the kick axis that counts as "the reflex connected".
    #[tunable(unit = "mm", min = 20.0, max = 300.0, step = 10.0)]
    KICK_DEPART_DIST: f64 = 100.0;
    /// Ball speed that also counts as departure (filtered estimate; confirms).
    #[tunable(unit = "mm/s", min = 200.0, max = 3000.0, step = 100.0)]
    KICK_DEPART_SPEED: f64 = 1000.0;

    // ── aim / shoot (Shoot) ───────────────────────────────────────────────────
    section "HandleBall aim";

    /// Hold radius: centre-of-robot to centre-of-ball while orbiting to aim.
    #[tunable(unit = "mm", min = 90.0, max = 200.0, step = 1.0)]
    BALL_TO_ROBOT_DISTANCE: f64 = 111.0;
    /// Tangential orbit speed cap. Kept low enough that the dribbler retains the ball
    /// through the slide — a faster orbit squeezes it out of the dribbler mouth.
    #[tunable(unit = "mm/s", min = 100.0, max = 1000.0, step = 25.0)]
    ORBIT_SPEED: f64 = 400.0;
    /// Orbit angular gain (commanded tangential speed per radian of aim error).
    #[tunable(min = 100.0, max = 1500.0, step = 50.0)]
    ORBIT_GAIN: f64 = 600.0;
    /// Floor on the orbit speed so the final approach to the aim line doesn't stall.
    #[tunable(unit = "mm/s", min = 0.0, max = 200.0, step = 5.0)]
    MIN_ORBIT_SPEED: f64 = 40.0;
    /// Radial gain pulling the robot to the hold radius while orbiting the ball.
    #[tunable(min = 0.5, max = 6.0, step = 0.25)]
    RADIUS_KP: f64 = 2.0;
    /// Robot→ball distance beyond which the ball is considered lost (well past the
    /// hold radius). Distance-based so a brief breakbeam flicker doesn't abort.
    #[tunable(unit = "mm", min = 120.0, max = 400.0, step = 10.0)]
    LOST_BALL_DISTANCE: f64 = 220.0;
    /// Heading tolerance to the shot axis before the kick is allowed.
    #[tunable(unit = "rad", min = 0.01, max = 0.3, step = 0.01)]
    YAW_TOLERANCE: f64 = 5.0 * std::f64::consts::PI / 180.0;
    /// Commanded kick speed for an aimed shot.
    #[tunable(unit = "mm/s", min = 1000.0, max = 6500.0, step = 100.0)]
    KICK_SPEED: f64 = 4000.0;
    /// Half-width of the lane-openness corridor checked along the shot.
    #[tunable(unit = "mm", min = 50.0, max = 400.0, step = 10.0)]
    LANE_HALF_WIDTH: f64 = 120.0;
    /// Range over which the shot-lane openness is evaluated.
    #[tunable(unit = "mm", min = 1000.0, max = 6000.0, step = 100.0)]
    LANE_RANGE: f64 = 3000.0;

    // ── launch-point selection (Shoot) ────────────────────────────────────────
    section "HandleBall launch";

    /// Inset from the physical field edge a kicking pose must keep (robot stays on the
    /// surface). Matches the acquire staging clamp.
    #[tunable(unit = "mm", min = 0.0, max = 400.0, step = 10.0)]
    LAUNCH_SURFACE_MARGIN: f64 = 130.0;
    /// A ball this close to a field line is "jammed" — repositioned inward before
    /// aiming, so we never try to aim/kick with the ball pinned on the boundary.
    #[tunable(unit = "mm", min = 100.0, max = 800.0, step = 25.0)]
    LAUNCH_BOUNDARY_MARGIN: f64 = 350.0;
    /// Ball within this distance of the chosen launch point → done repositioning.
    #[tunable(unit = "mm", min = 30.0, max = 300.0, step = 10.0)]
    REPOSITION_ARRIVE: f64 = 90.0;
    /// Carry acceleration cap (reuse the proven `Dribble` law: hold the ball, drive to
    /// the kicking pose under an acceleration cap so it isn't shaken loose).
    #[tunable(unit = "mm/s²", min = 100.0, max = 2000.0, step = 50.0)]
    CARRY_ACCEL_LIMIT: f64 = 500.0;
    /// Carry angular speed cap.
    #[tunable(unit = "rad/s", min = 0.5, max = 4.0, step = 0.1)]
    CARRY_ANGULAR_LIMIT: f64 = 1.5;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// No possession yet — run the capture front-end.
    Acquire,
    /// Dribble the held ball to a `Carry` target.
    Carry,
    /// Orbit the held ball to the shot axis (`Shoot`).
    Aim,
    /// Kick commanded this tick.
    Kicking,
    /// Waiting to confirm the ball left the dribbler.
    Verifying,
    /// Possess + face a heading; never self-completes.
    Hold,
}

pub struct HandleBallSkill {
    action: BallAction,
    /// Which side of the ball to take it from during the acquire sub-phase.
    acquire: AcquirePosition,
    /// Per-command strategy opt-in for firmware magnet capture (default `true`).
    /// ANDed with the `MAGNET_ENABLED` tunable and the robot's ToF capability.
    magnet: bool,
    /// External hold-fire gate for the drive-through `Strike`: while set, the
    /// commit latch is never engaged, so the skill stages behind the ball
    /// (aimed, `avoid_ball` on, kicker disarmed) indefinitely. Set only by the
    /// pass coordinator, which drives this skill directly — it stages the passer
    /// and releases the strike once the receiver is ready. Timer-safe: the
    /// drive-through has no approach timeout and `REFLEX_TIMEOUT` only starts
    /// on the commit that the gate suppresses.
    strike_gated: bool,
    status: SkillStatus,
    stage: Stage,
    /// World time (`t_received`, s) of the first tick. Sim-clock based for
    /// deterministic faster-than-real sim (see CLAUDE.md).
    first_tick: Option<f64>,
    /// World time the current stage was entered (per-stage timers).
    stage_entered: f64,
    /// Whether possession has ever been confirmed (gates the acquire backstop).
    had_ball: bool,
    /// World time a mid-act ball loss began (re-acquire debounce); `None` when held.
    lost_since: Option<f64>,
    /// End of the capture-settle window (post-breakbeam glide). Set only on the
    /// acquire→act transition — an act→act swap must not re-trigger the glide.
    settle_until: Option<f64>,
    /// Ball losses so far this episode (bounds silent re-acquire).
    reacquires: u32,
    // ── capture state (acquire) ──
    chosen_dir: Option<Vector2>,
    commit_pos: Option<Vector2>,
    commit_ball: Option<Vector2>,
    /// Schmitt latch for the commit drive (shared by acquire + strike): set on
    /// entering the tight corridor, held through the release bands (perp *and*
    /// along — see `acquire::latched`), cleared on a real bail or re-acquire.
    commit_latched: bool,
    /// World time the commit latch first engaged this attempt (the final capture
    /// drive began). Drives the tight pickup timeout; `None` while not committed,
    /// re-armed on the next latch edge and cleared on re-acquire.
    committed_since: Option<f64>,
    // ── strike (reflex) state ──
    kick_ball_pos: Option<Vector2>,
    armed_at: Option<f64>,
    // ── aim / verify state ──
    launch: Option<Vector2>,
    kick_time: Option<f64>,
    /// Compact per-phase summary appended to [`description`](Self::description).
    /// Set by the active drive each tick (the trait's `description()` has no
    /// world access, so the phase logic stashes its salient state here).
    detail: String,
}

impl HandleBallSkill {
    pub fn new(action: BallAction, acquire: AcquirePosition, magnet: bool) -> Self {
        Self {
            action,
            acquire,
            magnet,
            strike_gated: false,
            status: SkillStatus::Running,
            stage: Stage::Acquire,
            first_tick: None,
            stage_entered: 0.0,
            had_ball: false,
            lost_since: None,
            settle_until: None,
            reacquires: 0,
            chosen_dir: None,
            commit_pos: None,
            commit_ball: None,
            commit_latched: false,
            committed_since: None,
            kick_ball_pos: None,
            armed_at: None,
            launch: None,
            kick_time: None,
            detail: String::new(),
        }
    }

    /// Reconfigure (used by the pass coordinator, which drives this skill
    /// directly rather than through the executor).
    pub fn reconfigure(&mut self, action: BallAction, acquire: AcquirePosition) {
        self.action = action;
        self.acquire = acquire;
    }

    /// Set the strike hold-fire gate (see the field doc). Coordinator-only.
    pub(crate) fn set_strike_gate(&mut self, gated: bool) {
        self.strike_gated = gated;
    }

    /// Whether to engage firmware magnet capture this tick. All gates ANDed:
    /// master tunable on, per-command strategy flag on, the robot's ToF reporting
    /// `Ok`, the ToF currently seeing the ball (so the firmware's servo will work —
    /// it holds still when engaged-but-not-acquired), and we're in the commit
    /// corridor. `tof_ok`/`tof_ball_detected` are never set in sim, so this is a
    /// no-op there and the velocity capture runs unchanged.
    pub(super) fn magnet_engaged(&self, ctx: &SkillContext<'_>, committed: bool) -> bool {
        MAGNET_ENABLED() > 0.5
            && self.magnet
            && ctx.player.tof_ok
            // && ctx.player.tof_ball_detected
            && committed
    }

    /// Exit-bias heading for the acquire sub-phase. An explicit `Heading` overrides;
    /// otherwise (`Default`/`Fastest`) it derives from the action. For `Fastest` the
    /// heading is still used as the robot's facing during the static approach, but
    /// the side selection ignores it (see [`exit_bias_weight`](Self::exit_bias_weight)).
    fn acquire_heading(&self, ball_pos: Vector2) -> Angle {
        if let AcquirePosition::Heading(a) = self.acquire {
            return a;
        }
        match self.action {
            BallAction::Shoot { target } | BallAction::Strike { target, .. } => {
                Angle::from_vector(target - ball_pos)
            }
            BallAction::Carry { heading, .. } | BallAction::Hold { heading } => heading,
        }
    }

    /// How strongly the acquire side-selection is biased toward the exit heading.
    /// `Fastest` zeroes it (rank sides purely by how close the staging point is);
    /// `Default`/`Heading` use the full [`EXIT_BIAS_WEIGHT`] tunable.
    fn exit_bias_weight(&self) -> f64 {
        match self.acquire {
            AcquirePosition::Fastest => 0.0,
            AcquirePosition::Default | AcquirePosition::Heading(_) => EXIT_BIAS_WEIGHT(),
        }
    }

    fn hold_heading(&self) -> Angle {
        match self.action {
            BallAction::Hold { heading } | BallAction::Carry { heading, .. } => heading,
            _ => Angle::from_radians(0.0),
        }
    }

    /// Whether the current act stage matches the (possibly just-swapped) action.
    fn stage_matches_action(&self) -> bool {
        matches!(
            (self.stage, self.action),
            (Stage::Hold, BallAction::Hold { .. })
                | (Stage::Carry, BallAction::Carry { .. })
                | (Stage::Aim, BallAction::Shoot { .. })
                | (
                    Stage::Aim,
                    BallAction::Strike {
                        acquire_first: true,
                        ..
                    }
                )
        )
    }

    /// Enter the act stage for the current action once the ball is held.
    fn enter_act(&mut self, now: f64) {
        self.stage = match self.action {
            BallAction::Carry { .. } => Stage::Carry,
            BallAction::Shoot { .. }
            | BallAction::Strike {
                acquire_first: true,
                ..
            } => Stage::Aim,
            // Hold, and the drive-through Strike (handled before this point).
            _ => Stage::Hold,
        };
        self.stage_entered = now;
        self.launch = None;
        self.lost_since = None;
    }

    /// Reset to a fresh capture after a ball loss (bounded silent re-acquire).
    fn reacquire(&mut self, now: f64) {
        self.stage = Stage::Acquire;
        self.stage_entered = now;
        self.chosen_dir = None;
        self.commit_pos = None;
        self.commit_ball = None;
        self.commit_latched = false;
        self.committed_since = None;
        self.launch = None;
        self.lost_since = None;
        self.settle_until = None;
        self.reacquires += 1;
    }

    fn fail(&mut self) -> SkillProgress {
        self.status = SkillStatus::Failed;
        SkillProgress::failure()
    }

    /// One-word name of the current terminal action (for the debug firehose).
    fn action_str(&self) -> &'static str {
        match self.action {
            BallAction::Shoot { .. } => "shoot",
            BallAction::Strike { .. } => "strike",
            BallAction::Carry { .. } => "carry",
            BallAction::Hold { .. } => "hold",
        }
    }

    /// Emit the phase-independent debug values every tick: the stage, the active
    /// action, the re-acquire count, possession, and the robot→ball distance, all
    /// keyed `team_<Color>.p<id>.hb.*` (see the CLAUDE.md naming convention). Each
    /// phase adds its own keys on top via [`dkey`].
    fn emit_common(&self, ctx: &SkillContext<'_>, stage: &str) {
        let tc = ctx.team_context;
        tc.debug_string(dkey(ctx, "stage"), stage);
        tc.debug_string(dkey(ctx, "action"), self.action_str());
        tc.debug_value(dkey(ctx, "reacquires"), self.reacquires as f64);
        tc.debug_value(
            dkey(ctx, "has_ball"),
            if ctx.player.has_ball { 1.0 } else { 0.0 },
        );
        if let Some(ball) = ctx.world.ball.as_ref() {
            let dist = (ctx.player.position - ball.position.xy()).norm();
            tc.debug_value(dkey(ctx, "ball_dist"), dist);
        }
    }

    fn drive_hold(&mut self, ctx: &SkillContext<'_>, heading: Angle) -> SkillProgress {
        ctx.team_context
            .debug_value(dkey(ctx, "hold_heading_deg"), heading.degrees());
        self.detail = format!("face {:.0}°", heading.degrees());
        let mut input = PlayerControlInput::new();
        input.with_dribbling(DRIBBLER_SPEED());
        input.with_yaw(heading);
        SkillProgress::Continue(input)
    }

    fn drive_carry(&mut self, ctx: &SkillContext<'_>, now: f64) -> SkillProgress {
        let (to, heading) = match self.action {
            BallAction::Carry { to, heading } => (to, heading),
            _ => return self.drive_hold(ctx, self.hold_heading()),
        };
        let tc = ctx.team_context;
        tc.debug_cross_colored(dkey(ctx, "carry_to"), to, DebugColor::Purple);
        tc.debug_line_colored(
            dkey(ctx, "carry_path"),
            ctx.player.position,
            to,
            DebugColor::Purple,
        );
        let remaining = (ctx.player.position - to).norm();
        tc.debug_value(dkey(ctx, "carry_remaining"), remaining);
        // The skill owns Carry completion: arrive on distance, fail on its own timeout.
        if remaining < DRIBBLE_ARRIVE_DIST() {
            self.status = SkillStatus::Succeeded;
            return SkillProgress::success();
        }
        if now - self.stage_entered > DRIBBLE_TIMEOUT() {
            log::warn!("handle_ball: carry timed out");
            self.detail = "carry timeout".into();
            return self.fail();
        }
        self.detail = format!("→({:.0},{:.0}) {remaining:.0}mm", to.x, to.y);
        let mut input = PlayerControlInput::new();
        input.with_dribbling(CARRY_DRIBBLER_SPEED());
        input.with_position(to);
        input.with_yaw(heading);
        input.with_acceleration_limit(CARRY_ACCEL_LIMIT());
        input.with_angular_speed_limit(CARRY_ANGULAR_LIMIT());
        SkillProgress::Continue(input)
    }
}

/// Build a debug key in this skill's `hb` namespace: `p<id>.hb.<tag>` (the
/// `team_<Color>.` prefix is added by `team_context`).
pub(super) fn dkey(ctx: &SkillContext<'_>, tag: &str) -> String {
    format!("p{}.hb.{}", ctx.player.id.as_u32(), tag)
}

impl ExecutableSkill for HandleBallSkill {
    fn matches_command(&self, command: &SkillCommand) -> bool {
        matches!(command, SkillCommand::HandleBall { .. })
    }

    fn update_params(&mut self, command: &SkillCommand) {
        if let SkillCommand::HandleBall {
            action,
            acquire,
            magnet,
        } = command
        {
            self.action = *action;
            self.acquire = *acquire;
            self.magnet = *magnet;
        }
    }

    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress {
        let now = ctx.world.t_received;
        if self.first_tick.is_none() {
            self.first_tick = Some(now);
            // Stamp the initial acquire-stage clock (re-acquire / enter_act restamp it
            // on later stage transitions) so the approach timeout has a valid origin.
            self.stage_entered = now;
        }
        let has_ball = ctx.player.has_ball;
        if has_ball {
            self.had_ball = true;
        }
        let player_pos = ctx.player.position;

        // Drive-through Strike is a one-motion reflex strike-through: handled
        // separately (never holds, never re-acquires). The `acquire_first` variant
        // instead falls through to the normal acquire → aim path below and fires via
        // reflex once aimed.
        if let BallAction::Strike {
            target,
            acquire_first: false,
        } = self.action
        {
            return self.tick_strike(&ctx, target, now);
        }

        // Thrash backstop: a persistent loss surfaces as a failure and the caller
        // re-decides. The acquire approach/pickup timeouts live in `drive_acquire`.
        if self.reacquires > MAX_REACQUIRE {
            log::warn!("handle_ball: lost the ball too many times");
            return self.fail();
        }

        let Some(ball) = ctx.world.ball.as_ref() else {
            // No ball observation: hold if we have it, otherwise idle.
            let mut input = PlayerControlInput::new();
            if has_ball {
                input.with_dribbling(DRIBBLER_SPEED());
            }
            self.status = SkillStatus::Running;
            self.detail = "no ball seen".into();
            self.emit_common(&ctx, "no_ball");
            return SkillProgress::Continue(input);
        };
        let ball_pos = ball.position.xy();
        // Planar speed: the sim ball micro-bounces in place (large transient
        // |vz| with zero ground speed), so the 3-D norm must not feed the
        // kick-departure verify.
        let ball_vel_norm = ball.velocity.xy().norm();

        // Debounced silent re-acquire if the ball is lost during an act stage.
        // Distance-dependent grace: a far blow-out re-acquires fast; a ball still
        // near (just off the dribbler, breakbeam dark) gets a longer debounce so
        // contact flicker doesn't abort — but it MUST eventually re-acquire, or a
        // ball resting 110–220 mm ahead deadlocks a zero-translation Hold.
        if matches!(self.stage, Stage::Aim | Stage::Carry | Stage::Hold) {
            let far = (player_pos - ball_pos).norm() > LOST_BALL_DISTANCE();
            if !has_ball {
                let lost = *self.lost_since.get_or_insert(now);
                let grace = if far {
                    BALL_LOST_GRACE()
                } else {
                    NEAR_LOST_GRACE()
                };
                if now - lost > grace {
                    self.reacquire(now);
                }
            } else {
                self.lost_since = None;
            }
        }

        // Route: acquire until we hold the ball, then enter / re-route the act stage.
        if matches!(self.stage, Stage::Acquire) {
            if has_ball {
                self.enter_act(now);
                // Open the capture-settle window: glide down from the commit-drive
                // speed instead of stopping dead (which bounces the ball off).
                self.settle_until = Some(now + SETTLE_TIME());
            } else {
                self.status = SkillStatus::Running;
                self.emit_common(&ctx, "acquire");
                return self.drive_acquire(&ctx, ball_pos, ball.velocity.xy(), player_pos, now);
            }
        } else if matches!(self.stage, Stage::Aim | Stage::Carry | Stage::Hold)
            && !self.stage_matches_action()
        {
            // Action swapped live (e.g. Hold -> Shoot) — re-route. Kicking/Verifying
            // are committed and never re-routed.
            self.enter_act(now);
        }

        self.status = SkillStatus::Running;
        let stage_name = match self.stage {
            Stage::Acquire => "acquire",
            Stage::Carry => "carry",
            Stage::Aim => "aim",
            Stage::Kicking => "kick",
            Stage::Verifying => "verify",
            Stage::Hold => "hold",
        };
        self.emit_common(&ctx, stage_name);
        let mut progress = match self.stage {
            Stage::Hold => self.drive_hold(&ctx, self.hold_heading()),
            Stage::Carry => self.drive_carry(&ctx, now),
            Stage::Aim => self.drive_aim(&ctx, ball_pos, player_pos, now),
            Stage::Kicking => self.drive_kick(&ctx, ball_pos, now),
            Stage::Verifying => self.drive_verify(&ctx, ball_pos, ball_vel_norm, now),
            Stage::Acquire => unreachable!("acquire handled above"),
        };
        // Capture settle: for a short window after the acquire→act transition,
        // cap deceleration and yaw rate so the robot glides through the contact
        // instead of stopping/spinning dead — a hard stop at capture bounces the
        // ball off the dribbler before it is seated. Applied on top of whatever
        // the act drive commanded (tightest limit wins).
        let settling = self.settle_until.is_some_and(|until| now < until)
            && matches!(self.stage, Stage::Hold | Stage::Carry | Stage::Aim);
        ctx.team_context
            .debug_value(dkey(&ctx, "settle"), if settling { 1.0 } else { 0.0 });
        if settling {
            if let SkillProgress::Continue(input) = &mut progress {
                let accel = input.acceleration_limit.unwrap_or(f64::INFINITY);
                input.with_acceleration_limit(accel.min(SETTLE_DECEL()));
                let yaw_rate = input.angular_speed_limit.unwrap_or(f64::INFINITY);
                input.with_angular_speed_limit(yaw_rate.min(SETTLE_YAW_RATE()));
            }
        }
        progress
    }

    fn status(&self) -> SkillStatus {
        self.status
    }

    fn skill_type(&self) -> &'static str {
        "HandleBall"
    }

    fn is_oneshot(&self) -> bool {
        true
    }

    fn description(&self) -> String {
        let act = self.action_str();
        let stage = match self.stage {
            Stage::Acquire => "acquiring",
            Stage::Carry => "carrying",
            Stage::Aim => "aiming",
            Stage::Kicking => "kicking",
            Stage::Verifying => "verifying",
            Stage::Hold => "holding",
        };
        if self.detail.is_empty() {
            format!("{act}: {stage}")
        } else {
            format!("{act}: {stage} · {}", self.detail)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::test_support::{player, team_ctx, world};
    use crate::control::{KickerControlInput, ObstacleSet, TeamContext};
    use dies_core::{PlayerData, TeamData};

    fn skill(action: BallAction) -> HandleBallSkill {
        HandleBallSkill::new(action, AcquirePosition::Default, true)
    }

    fn ctx<'a>(w: &'a TeamData, tc: &'a TeamContext, p: &'a PlayerData) -> SkillContext<'a> {
        SkillContext {
            player: p,
            world: w,
            team_context: tc,
            debug_prefix: tc.key(format!("p{}", p.id)),
            obstacles: ObstacleSet::default(),
        }
    }

    /// A robot at `pos`, holding the ball (breakbeam latched).
    fn held_player(pos: Vector2) -> PlayerData {
        let mut p = player(0, pos, 0.0);
        p.has_ball = true;
        p
    }

    /// A single-robot world with the ball at `ball`, sampled at world time `t`.
    fn world_at(p: &PlayerData, ball: Vector2, t: f64) -> TeamData {
        let mut w = world(vec![p.clone()], Some(ball), 0.016);
        w.t_received = t;
        w
    }

    #[test]
    fn carry_self_completes_on_arrival() {
        let to = Vector2::new(1000.0, 0.0);
        let mut s = skill(BallAction::Carry {
            to,
            heading: Angle::from_radians(0.0),
        });
        // Already at the target, holding the ball → the skill (not the driver) reports
        // arrival on the first tick that enters the carry stage.
        let p = held_player(to);
        let tc = team_ctx();
        let w = world_at(&p, to, 0.0);
        assert!(matches!(s.tick(ctx(&w, &tc, &p)), SkillProgress::Done(_)));
        assert_eq!(s.status(), SkillStatus::Succeeded);
    }

    #[test]
    fn carry_times_out_when_target_unreached() {
        let to = Vector2::new(3000.0, 0.0);
        let mut s = skill(BallAction::Carry {
            to,
            heading: Angle::from_radians(0.0),
        });
        let start = Vector2::new(0.0, 0.0);
        let p = held_player(start);
        let tc = team_ctx();
        // Frame 1 (t=0) enters the carry stage and keeps going.
        let w0 = world_at(&p, start, 0.0);
        assert!(matches!(
            s.tick(ctx(&w0, &tc, &p)),
            SkillProgress::Continue(_)
        ));
        // Frame 2 past the carry budget with no progress → the skill fails itself.
        let w1 = world_at(&p, start, DRIBBLE_TIMEOUT() + 1.0);
        assert!(matches!(s.tick(ctx(&w1, &tc, &p)), SkillProgress::Done(_)));
        assert_eq!(s.status(), SkillStatus::Failed);
    }

    #[test]
    fn acquire_approach_times_out() {
        let mut s = skill(BallAction::Hold {
            heading: Angle::from_radians(0.0),
        });
        // Robot far from the ball and never secures it (has_ball = false).
        let p = player(0, Vector2::new(0.0, 0.0), 0.0);
        let tc = team_ctx();
        let ball = Vector2::new(2000.0, 0.0);
        // Frame 1 (t=0) stamps the approach clock while still traversing.
        let w0 = world_at(&p, ball, 0.0);
        assert!(matches!(
            s.tick(ctx(&w0, &tc, &p)),
            SkillProgress::Continue(_)
        ));
        // Frame 2 past the approach budget, still uncommitted → the skill fails itself.
        let w1 = world_at(&p, ball, APPROACH_TIMEOUT() + 1.0);
        assert!(matches!(s.tick(ctx(&w1, &tc, &p)), SkillProgress::Done(_)));
        assert_eq!(s.status(), SkillStatus::Failed);
    }

    #[test]
    fn capture_settle_glides_then_releases_the_limits() {
        let mut s = skill(BallAction::Hold {
            heading: Angle::from_radians(1.0),
        });
        let pos = Vector2::new(500.0, 0.0);
        let p = held_player(pos);
        let tc = team_ctx();
        // First tick captures (acquire → hold) and opens the settle window: the
        // hold input must carry the gentle decel + yaw-rate caps so the robot
        // glides through the contact instead of stopping/spinning dead.
        let w0 = world_at(&p, pos, 0.0);
        match s.tick(ctx(&w0, &tc, &p)) {
            SkillProgress::Continue(input) => {
                assert_eq!(input.acceleration_limit, Some(SETTLE_DECEL()));
                assert_eq!(input.angular_speed_limit, Some(SETTLE_YAW_RATE()));
            }
            p => panic!("expected Continue, got {p:?}"),
        }
        // Past the window the hold input is unclamped again.
        let w1 = world_at(&p, pos, SETTLE_TIME() + 0.1);
        match s.tick(ctx(&w1, &tc, &p)) {
            SkillProgress::Continue(input) => {
                assert_eq!(input.acceleration_limit, None);
                assert!(input.angular_speed_limit.unwrap_or(f64::INFINITY) > SETTLE_YAW_RATE());
            }
            p => panic!("expected Continue, got {p:?}"),
        }
    }

    #[test]
    fn hold_with_ball_lost_nearby_reacquires_after_grace() {
        // Regression (2026-07-02 GreenTea freeze): a ball resting ~135 mm ahead is
        // invisible to the breakbeam yet inside LOST_BALL_DISTANCE, so the old
        // `!has_ball && far` gate never fired and the zero-translation Hold
        // deadlocked. A near loss must re-acquire after NEAR_LOST_GRACE; a brief
        // (sub-grace) dropout must ride through and stay in Hold.
        let mut s = skill(BallAction::Hold {
            heading: Angle::from_radians(0.0),
        });
        let pos = Vector2::new(500.0, 0.0);
        let tc = team_ctx();
        // Tick 1: breakbeam latched → acquire routes into Hold.
        let held = held_player(pos);
        let w0 = world_at(&held, pos, 0.0);
        assert!(matches!(
            s.tick(ctx(&w0, &tc, &held)),
            SkillProgress::Continue(_)
        ));
        assert!(matches!(s.stage, Stage::Hold));
        // Ball pops off the dribbler and settles 135 mm away, breakbeam dark.
        // The first no-ball tick (t=0.1) stamps the loss clock.
        let lost = player(0, pos, 0.0);
        let near_ball = pos + Vector2::new(135.0, 0.0);
        let t_lost = 0.1;
        let w1 = world_at(&lost, near_ball, t_lost);
        assert!(matches!(
            s.tick(ctx(&w1, &tc, &lost)),
            SkillProgress::Continue(_)
        ));
        assert!(matches!(s.stage, Stage::Hold));
        // Within the grace window: still Hold (flicker tolerance).
        let w2 = world_at(&lost, near_ball, t_lost + NEAR_LOST_GRACE() * 0.5);
        assert!(matches!(
            s.tick(ctx(&w2, &tc, &lost)),
            SkillProgress::Continue(_)
        ));
        assert!(matches!(s.stage, Stage::Hold));
        // Past the grace window: the skill must silently re-acquire.
        let w3 = world_at(&lost, near_ball, t_lost + NEAR_LOST_GRACE() + 0.05);
        assert!(matches!(
            s.tick(ctx(&w3, &tc, &lost)),
            SkillProgress::Continue(_)
        ));
        assert!(matches!(s.stage, Stage::Acquire));
    }

    #[test]
    fn shoot_releases_with_smart_kick() {
        let s = skill(BallAction::Shoot {
            target: Vector2::new(1000.0, 0.0),
        });
        assert!(matches!(s.shot(), Some((_, KickerControlInput::Kick))));
    }

    #[test]
    fn acquire_first_strike_releases_via_reflex() {
        let s = skill(BallAction::Strike {
            target: Vector2::new(1000.0, 0.0),
            acquire_first: true,
        });
        assert!(matches!(
            s.shot(),
            Some((_, KickerControlInput::ReflexKick))
        ));
    }

    #[test]
    fn drive_through_strike_is_not_an_aim_shot() {
        // The free-ball drive-through is handled by `tick_strike`, not the aim path,
        // so it must not be picked up as a `shot()` here.
        let s = skill(BallAction::Strike {
            target: Vector2::new(1000.0, 0.0),
            acquire_first: false,
        });
        assert!(s.shot().is_none());
    }

    #[test]
    fn gated_strike_stages_and_never_arms_until_ungated() {
        // The pass coordinator's hold-fire gate: a drive-through strike parked
        // in commit-ready geometry (staged 200 mm behind the ball on the strike
        // axis) must NOT latch/arm while gated — it stages with ball avoidance
        // on. Ungating the same geometry commits and arms the reflex next tick.
        let target = Vector2::new(2000.0, 0.0);
        let mut s = skill(BallAction::Strike {
            target,
            acquire_first: false,
        });
        s.set_strike_gate(true);
        let p = player(0, Vector2::new(-200.0, 0.0), 0.0);
        let ball = Vector2::new(0.0, 0.0);
        let tc = team_ctx();

        let w0 = world_at(&p, ball, 0.0);
        match s.tick(ctx(&w0, &tc, &p)) {
            SkillProgress::Continue(input) => {
                assert!(
                    !matches!(input.kicker, KickerControlInput::ReflexKick),
                    "gated strike must not arm the reflex"
                );
                assert!(input.avoid_ball, "gated strike must keep ball avoidance");
            }
            other => panic!("expected Continue while gated, got {other:?}"),
        }

        s.set_strike_gate(false);
        let w1 = world_at(&p, ball, 0.016);
        match s.tick(ctx(&w1, &tc, &p)) {
            SkillProgress::Continue(input) => {
                assert!(
                    matches!(input.kicker, KickerControlInput::ReflexKick),
                    "ungated strike from staging must commit and arm the reflex"
                );
                assert!(!input.avoid_ball);
            }
            other => panic!("expected Continue while committing, got {other:?}"),
        }
    }

    #[test]
    fn strike_departure_verifies_even_when_the_latch_breaks() {
        // Regression: the kicked ball outruns the ball filter, so the estimate
        // jumps PAST the commit-latch release band on the very frame it finally
        // moves. The departure verify used to live inside the committed branch
        // and missed its own kick — the robot re-staged after the departed ball
        // and only "succeeded" metres downfield. Once armed, the verify must
        // run regardless of the latch.
        let target = Vector2::new(2000.0, 0.0);
        let mut s = skill(BallAction::Strike {
            target,
            acquire_first: false,
        });
        let p = player(0, Vector2::new(-200.0, 0.0), 0.0);
        let tc = team_ctx();

        // Tick 1: commit-ready geometry → latch + arm (kick origin stamped).
        let w0 = world_at(&p, Vector2::new(0.0, 0.0), 0.0);
        match s.tick(ctx(&w0, &tc, &p)) {
            SkillProgress::Continue(input) => {
                assert!(matches!(input.kicker, KickerControlInput::ReflexKick));
            }
            other => panic!("expected armed Continue, got {other:?}"),
        }

        // Tick 2: the ball estimate jumps 600 mm down the axis — far outside the
        // latch release band (unlatched this same frame) — the departure must
        // still verify as success.
        let w1 = world_at(&p, Vector2::new(600.0, 0.0), 0.016);
        assert!(matches!(s.tick(ctx(&w1, &tc, &p)), SkillProgress::Done(_)));
        assert_eq!(s.status(), SkillStatus::Succeeded);
    }

    #[test]
    fn blocked_aim_force_fires_past_the_backstop() {
        // A robot aligned with its shot but with the lane covered used to hold the
        // ball indefinitely: the aim backstop FAILED the skill, the caller
        // re-engaged the same robot with a fresh clock, and the hold-and-spin
        // repeated (observed 9–17 s of continuous dribbler contact). Past the
        // backstop the shot must instead be forced through the covered lane.
        let target = Vector2::new(1000.0, 0.0);
        let mut s = skill(BallAction::Shoot { target });
        let p = held_player(Vector2::new(0.0, 0.0));
        // A teammate parked squarely in the shot corridor → lane_blocked.
        let blocker = player(1, Vector2::new(400.0, 0.0), 0.0);
        let ball = Vector2::new(100.0, 0.0);
        let tc = team_ctx();

        let mk = |t: f64| {
            let mut w = world(vec![p.clone(), blocker.clone()], Some(ball), 0.016);
            w.t_received = t;
            w
        };

        // Frame 1 (t=0): enters the aim stage; blocked lane → no kick commit.
        let w0 = mk(0.0);
        assert!(matches!(
            s.tick(ctx(&w0, &tc, &p)),
            SkillProgress::Continue(_)
        ));
        // Frame 2 past the backstop: forced — aligned within the loose gate, so the
        // commit happens despite the covered lane.
        let w1 = mk(AIM_BACKSTOP() + 0.1);
        assert!(matches!(
            s.tick(ctx(&w1, &tc, &p)),
            SkillProgress::Continue(_)
        ));
        // Frame 3: the kick stage fires the smart kick through the covered lane.
        let w2 = mk(AIM_BACKSTOP() + 0.2);
        match s.tick(ctx(&w2, &tc, &p)) {
            SkillProgress::Continue(input) => {
                assert!(
                    matches!(input.kicker, KickerControlInput::Kick),
                    "forced shot must arm the kick, got {:?}",
                    input.kicker
                );
            }
            other => panic!("expected the forced kick tick, got {other:?}"),
        }
    }
}
