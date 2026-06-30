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
use dies_strategy_protocol::{BallAction, SkillCommand, SkillStatus};

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::PlayerControlInput;

// ── episode timing / thrash backstops ────────────────────────────────────────
/// Debounce (s) before a ball loss during an act stage triggers a re-acquire — a
/// pass-style ride on the dribbler edge routinely drops the breakbeam for a frame
/// or two. Mirrors the pass coordinator's `SETUP_BALL_LOST_GRACE`.
const BALL_LOST_GRACE: f64 = 0.2;
/// Backstop (s): if we have *never* secured the ball this long after the first
/// tick, give up so the planner can re-elect a capturer.
const ACQUIRE_BACKSTOP: f64 = 6.0;
/// Backstop (s) for the aim stage once we hold the ball (a permanently blocked
/// lane, say). In concerto the driver's per-action timeout fires well before this.
const AIM_BACKSTOP: f64 = 6.0;
/// Max ball losses before the skill gives up instead of re-acquiring forever.
const MAX_REACQUIRE: u32 = 6;

// ── dribbler ─────────────────────────────────────────────────────────────────
/// Dribbler speed used while acquiring, holding, aiming, and reflex-striking.
const DRIBBLER_SPEED: f64 = 0.6;
/// Dribbler speed while carrying the ball to a `Carry` target (matches `Dribble`).
const CARRY_DRIBBLER_SPEED: f64 = 0.5;

// ── acquire (capture front-end) ──────────────────────────────────────────────
const APPROACH_DISTANCE: f64 = 200.0;
const COMMIT_DISTANCE: f64 = 280.0;
const COMMIT_PERP: f64 = 30.0;
const GATE_PERP: f64 = 80.0;
const APPROACH_GAIN: f64 = 3.0;
const APPROACH_MIN_SPEED: f64 = 200.0;
const LATERAL_GAIN: f64 = 4.0;
const APPROACH_CARE: f64 = -0.6;
const BALL_MOVED_FAIL: f64 = 100.0;
const DRIVEN_FAIL: f64 = 250.0;
/// Inset (mm) from the *physical* field edge (touchline/goal line **plus** the
/// run-off `boundary_width`) that the staging point is kept inside. Robots may
/// legally stand in the run-off, so the staging point only has to stay on the
/// playing surface — clamping to the line itself would forbid getting *behind* a
/// ball pinned on the line and the robot would stall short of it (only the ball
/// is "out" past the line; the rescue-inward heading keeps the ball in). Covers
/// the robot radius plus a buffer so the body stays on the surface.
const STAGING_FIELD_MARGIN: f64 = 130.0;

// ── heading-free approach-side selection ─────────────────────────────────────
// A normal capture is not given a fixed approach heading: the skill picks which
// side of the ball to take it from, choosing the fastest reachable, obstacle-free
// approach point, biased toward a desired *exit* direction (so the capture also
// roughly sets up the next action). `approach`/`acquire_heading` carries that soft
// exit bias.
/// Number of candidate push directions sampled around the ball.
const N_APPROACH_SAMPLES: usize = 24;
/// Ego clearance required at a candidate staging point (robot radius + buffer).
const APPROACH_EGO_RADIUS: f64 = PLAYER_RADIUS + 20.0;
/// Clearance required along the commit corridor (staging point → ball). Slightly
/// tighter so only a real blocker between us and the ball rejects an approach.
const APPROACH_COMMIT_RADIUS: f64 = PLAYER_RADIUS;
/// Sampling step (mm) for the commit-corridor clearance check.
const APPROACH_CLEAR_STEP: f64 = 60.0;
/// Value (mm of approach distance) of a perfectly exit-aligned push. Trades off
/// against "closest staging point": a well-aimed capture is worth approaching
/// from a bit farther away. Soft — boundary safety is a hard constraint below.
const EXIT_BIAS_WEIGHT: f64 = 600.0;
/// The chosen side is kept unless a candidate beats it by more than this (mm),
/// so the approach side doesn't flip-flop on noise / orbiting opponents.
const APPROACH_HYSTERESIS: f64 = 200.0;
/// A ball within this distance (mm) of a field line forbids any approach whose
/// push has an outward component there — never dribble the ball out (only the
/// ball is "out" past the line; this is the touchline/goal-line rescue, now
/// folded into the generic selector instead of a special heading).
const BALL_KEEPIN_MARGIN: f64 = 350.0;
/// Tolerance on the inward-push constraint (slightly negative so a near-parallel
/// push along the line is still allowed).
const MIN_INWARD_PUSH: f64 = -0.05;

// ── moving-ball tail-catch + contact offset ──────────────────────────────────
/// Lateral contact offset (mm): the ball is centered this far to the robot's
/// LEFT of dribbler center so it rests clear of the dribbler-drive gear on the
/// right. Applied in the robot's heading frame (for a moving catch the heading is
/// the drive direction, so heading-left and axis-left coincide).
const PICKUP_LATERAL_OFFSET: f64 = 2.5;
/// Ball speed (mm/s) above which a capture switches from static side-selection to
/// a velocity-aware tail-catch: lead the intercept and feed-forward the ball
/// speed so the robot matches pace instead of decelerating into a stern chase.
/// Below this the ball is effectively static and the side-selector is better.
const MOVING_BALL_SPEED: f64 = 300.0;
/// Cosine of the half-cone within which a ball rolling *toward* the robot is
/// treated as head-on and left to `Receive` (a tail-catch would loop around).
const HEAD_ON_COS: f64 = 0.6; // ~53°
/// Nominal robot speed (mm/s) for the intercept-time estimate. Conservative
/// (below true max) so the aim point lands slightly short rather than past the
/// ball.
const INTERCEPT_ROBOT_SPEED: f64 = 2500.0;
/// Cap (s) on the intercept look-ahead.
const MAX_INTERCEPT_TIME: f64 = 1.5;
/// Lateral deviation (mm) of a moving ball from the commit axis that counts as
/// the ball squirting out of the corridor. Replaces the static `BALL_MOVED_FAIL`
/// while the ball is legitimately rolling *along* the axis.
const BALL_STRAY_FAIL: f64 = 150.0;

// ── reflex strike (Strike) ───────────────────────────────────────────────────
/// Ball displacement along the kick axis that counts as "the reflex connected".
const KICK_DEPART_DIST: f64 = 100.0;
/// Ball speed that also counts as departure (filtered estimate; confirms).
const KICK_DEPART_SPEED: f64 = 1000.0;
/// How long to wait after arming the reflex for the ball to depart before
/// declaring a whiff and failing (so a dribbler-on hold can't linger and
/// accumulate double-touch contact).
const REFLEX_TIMEOUT: Duration = Duration::from_millis(600);

// ── aim / shoot (Shoot) ──────────────────────────────────────────────────────
const BALL_TO_ROBOT_DISTANCE: f64 = 111.0;
/// Tangential orbit speed cap. Kept low enough that the dribbler can retain the
/// ball through the slide — a faster orbit squeezes it out of the dribbler mouth.
const ORBIT_SPEED: f64 = 400.0;
const ORBIT_GAIN: f64 = 600.0;
const MIN_ORBIT_SPEED: f64 = 40.0;
/// Radial gain pulling the robot to the hold radius while orbiting the ball.
const RADIUS_KP: f64 = 2.0;
/// Robot→ball distance beyond which the ball is considered lost (well past the
/// `BALL_TO_ROBOT_DISTANCE` hold radius). Distance-based so a brief breakbeam
/// flicker during the slide doesn't abort.
const LOST_BALL_DISTANCE: f64 = 220.0;
const YAW_TOLERANCE: f64 = 5.0 * std::f64::consts::PI / 180.0;
const KICK_SPEED: f64 = 4000.0;
const LANE_HALF_WIDTH: f64 = 120.0;
const LANE_RANGE: f64 = 3000.0;
/// How long to wait for the ball to leave after commanding a kick before
/// declaring the kick a whiff and failing.
const VERIFY_WINDOW: Duration = Duration::from_millis(200);

// ── launch-point selection (Shoot) ───────────────────────────────────────────
/// Clearance required at a candidate kicking pose (robot behind the launch point).
const LAUNCH_POSE_EGO: f64 = PLAYER_RADIUS;
/// Inset from the physical field edge a kicking pose must keep (robot stays on
/// the surface). Matches the acquire staging clamp.
const LAUNCH_SURFACE_MARGIN: f64 = 130.0;
/// A ball this close to a field line is "jammed" — repositioned inward before
/// aiming, so we never try to aim/kick with the ball pinned on the boundary.
const LAUNCH_BOUNDARY_MARGIN: f64 = 350.0;
/// Carry distances (mm) tried when searching for a reachable launch point. Short
/// — repositioning is a small nudge off a bad spot, not a dribble across field.
const CARRY_STEPS: [f64; 3] = [300.0, 500.0, 700.0];
/// Lateral fan (radians) tried around the inward direction at each carry step.
const CARRY_FAN: [f64; 5] = [0.0, 0.5, -0.5, 1.0, -1.0];
/// Ball within this distance of the chosen launch point → done repositioning.
const REPOSITION_ARRIVE: f64 = 90.0;
/// Carry motion limits (reuse the proven `Dribble` law: hold the ball, drive to
/// the kicking pose under an acceleration cap so it isn't shaken loose). Shared
/// with the `Carry` action.
const CARRY_ACCEL_LIMIT: f64 = 500.0;
const CARRY_ANGULAR_LIMIT: f64 = 1.5;

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
    /// Exit-bias heading for the acquire sub-phase; `None` derives from `action`.
    approach: Option<Angle>,
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
    pub fn new(action: BallAction, approach: Option<Angle>) -> Self {
        Self {
            action,
            approach,
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
            kick_ball_pos: None,
            armed_at: None,
            launch: None,
            kick_time: None,
            detail: String::new(),
        }
    }

    /// Reconfigure (used by the pass coordinator's Secure phase, which drives this
    /// skill directly rather than through the executor).
    pub fn reconfigure(&mut self, action: BallAction, approach: Option<Angle>) {
        self.action = action;
        self.approach = approach;
    }

    /// Exit-bias heading for the acquire sub-phase.
    fn acquire_heading(&self, ball_pos: Vector2) -> Angle {
        if let Some(a) = self.approach {
            return a;
        }
        match self.action {
            BallAction::Shoot { target } | BallAction::Strike { target } => {
                Angle::from_vector(target - ball_pos)
            }
            BallAction::Carry { heading, .. } | BallAction::Hold { heading } => heading,
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
        )
    }

    /// Enter the act stage for the current action once the ball is held.
    fn enter_act(&mut self, now: f64) {
        self.stage = match self.action {
            BallAction::Carry { .. } => Stage::Carry,
            BallAction::Shoot { .. } => Stage::Aim,
            // Hold, and the unreachable Strike (handled before this point).
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
        input.with_dribbling(DRIBBLER_SPEED);
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
        input.with_dribbling(CARRY_DRIBBLER_SPEED);
        input.with_position(to);
        input.with_yaw(heading);
        input.with_acceleration_limit(CARRY_ACCEL_LIMIT);
        input.with_angular_speed_limit(CARRY_ANGULAR_LIMIT);
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
        if let SkillCommand::HandleBall { action, approach } = command {
            self.action = *action;
            self.approach = *approach;
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

        // Strike is a one-motion reflex strike-through: handled separately (never
        // holds, never re-acquires).
        if let BallAction::Strike { target } = self.action {
            return self.tick_strike(&ctx, target, now);
        }

        // Acquisition / thrash backstops so a persistent loss surfaces as a
        // failure and the planner can re-elect a capturer.
        if !self.had_ball && now - first > ACQUIRE_BACKSTOP {
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
                input.with_dribbling(DRIBBLER_SPEED);
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
            let far = (player_pos - ball_pos).norm() > LOST_BALL_DISTANCE;
            if !has_ball && far {
                let lost = *self.lost_since.get_or_insert(now);
                if now - lost > BALL_LOST_GRACE {
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
