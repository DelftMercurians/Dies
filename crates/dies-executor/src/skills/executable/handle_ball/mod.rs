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
//! - `Hold`/`Carry` **never** return `Done` — they run forever (the caller decides
//!   arrival). Every in-possession action swap is therefore a live param update.
//! - A kick (`Shoot`/`Strike`) is the only `Done(Success)`.
//! - Internal capture completion (breakbeam) is an internal stage edge, not a
//!   `Done` — that is what makes the acquire→act seam disappear.
//!
//! ## Silent re-acquire
//! Losing the ball mid-aim/carry returns to `Acquire` (debounced) instead of
//! failing up to the strategy. Bounded by [`MAX_REACQUIRE`] and the acquire/aim
//! backstops so a persistent loss still surfaces as `Done(Failure)` and the
//! planner can re-elect a capturer. `Strike` never re-acquires (double-touch safe).

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
/// Clearance required along the commit corridor (staging point → ball). Slightly
/// tighter so only a real blocker between us and the ball rejects an approach.
const APPROACH_COMMIT_RADIUS: f64 = PLAYER_RADIUS;
/// How long to wait after arming the reflex for the ball to depart before
/// declaring a whiff and failing (so a dribbler-on hold can't linger and
/// accumulate double-touch contact).
const REFLEX_TIMEOUT: Duration = Duration::from_millis(600);
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
    /// Backstop: if we have *never* secured the ball this long after the first tick,
    /// give up so the planner can re-elect a capturer.
    #[tunable(unit = "s", min = 1.0, max = 12.0, step = 0.5)]
    ACQUIRE_BACKSTOP: f64 = 60.0;
    /// Backstop for the aim stage once we hold the ball (a permanently blocked lane,
    /// say). In concerto the driver's per-action timeout fires well before this.
    #[tunable(unit = "s", min = 1.0, max = 12.0, step = 0.5)]
    AIM_BACKSTOP: f64 = 6.0;

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
    /// ≤ GATE_PERP so forward drive is already throttled to a crawl near the
    /// boundary — the robot slides back onto the axis rather than ramming the ball.
    #[tunable(unit = "mm", min = 30.0, max = 200.0, step = 5.0)]
    COMMIT_PERP_RELEASE: f64 = 70.0;
    /// Perpendicular gate width for entering the commit drive.
    #[tunable(unit = "mm", min = 20.0, max = 200.0, step = 5.0)]
    GATE_PERP: f64 = 80.0;
    /// Proportional gain on the approach drive toward the ball.
    #[tunable(min = 0.5, max = 8.0, step = 0.25)]
    APPROACH_GAIN: f64 = 3.0;
    /// Minimum approach speed so the final drive doesn't stall short of the ball.
    #[tunable(unit = "mm/s", min = 50.0, max = 600.0, step = 25.0)]
    APPROACH_MIN_SPEED: f64 = 200.0;
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
    /// Ball losses so far this episode (bounds silent re-acquire).
    reacquires: u32,
    // ── capture state (acquire) ──
    chosen_dir: Option<Vector2>,
    commit_pos: Option<Vector2>,
    commit_ball: Option<Vector2>,
    /// Schmitt latch for the commit drive (shared by acquire + strike): set on
    /// entering the tight corridor, held through the release band, cleared on a
    /// real bail or re-acquire.
    commit_latched: bool,
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
            status: SkillStatus::Running,
            stage: Stage::Acquire,
            first_tick: None,
            stage_entered: 0.0,
            had_ball: false,
            lost_since: None,
            reacquires: 0,
            chosen_dir: None,
            commit_pos: None,
            commit_ball: None,
            commit_latched: false,
            kick_ball_pos: None,
            armed_at: None,
            launch: None,
            kick_time: None,
            detail: String::new(),
        }
    }

    /// Reconfigure (used by the pass coordinator's Secure phase, which drives this
    /// skill directly rather than through the executor).
    pub fn reconfigure(&mut self, action: BallAction, acquire: AcquirePosition) {
        self.action = action;
        self.acquire = acquire;
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
        self.launch = None;
        self.lost_since = None;
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

    fn drive_carry(&mut self, ctx: &SkillContext<'_>) -> SkillProgress {
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
        let first = *self.first_tick.get_or_insert(now);
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

        // Acquisition / thrash backstops so a persistent loss surfaces as a
        // failure and the planner can re-elect a capturer.
        if !self.had_ball && now - first > ACQUIRE_BACKSTOP() {
            log::warn!("handle_ball: could not acquire the ball");
            return self.fail();
        }
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
        let ball_vel_norm = ball.velocity.norm();

        // Debounced silent re-acquire if the ball is lost during an act stage.
        if matches!(self.stage, Stage::Aim | Stage::Carry | Stage::Hold) {
            let far = (player_pos - ball_pos).norm() > LOST_BALL_DISTANCE();
            if !has_ball && far {
                let lost = *self.lost_since.get_or_insert(now);
                if now - lost > BALL_LOST_GRACE() {
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
        match self.stage {
            Stage::Hold => self.drive_hold(&ctx, self.hold_heading()),
            Stage::Carry => self.drive_carry(&ctx),
            Stage::Aim => self.drive_aim(&ctx, ball_pos, player_pos, now),
            Stage::Kicking => self.drive_kick(&ctx, ball_pos, now),
            Stage::Verifying => self.drive_verify(&ctx, ball_pos, ball_vel_norm, now),
            Stage::Acquire => unreachable!("acquire handled above"),
        }
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
    use crate::control::KickerControlInput;

    fn skill(action: BallAction) -> HandleBallSkill {
        HandleBallSkill::new(action, AcquirePosition::Default, true)
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
}
